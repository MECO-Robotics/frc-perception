"""Detector implementations used by SCOUT."""
from dataclasses import dataclass
from pathlib import Path
import time
import threading
import json
import os
import cv2
import numpy as np

COCO_NAMES = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']


def letterbox_geometry(frame_width, frame_height, input_width, input_height):
    scale = min(input_width / frame_width, input_height / frame_height)
    resized_width = round(frame_width * scale)
    resized_height = round(frame_height * scale)
    offset_x = (input_width - resized_width) // 2
    offset_y = (input_height - resized_height) // 2
    return scale, resized_width, resized_height, offset_x, offset_y

@dataclass
class Detection:
    x1: int; y1: int; x2: int; y2: int; class_id: int; confidence: float

class Detector:
    name='detector'; precision='n/a'; input_size=0
    def infer(self, frame): raise NotImplementedError

class MotionDetector(Detector):
    name='motion-baseline'; precision='cpu'
    def __init__(self): self.last=None
    def infer(self, frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if self.last is None: self.last=gray; return []
        delta=cv2.absdiff(gray,self.last); self.last=gray; _,mask=cv2.threshold(delta,28,255,cv2.THRESH_BINARY); contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        out=[]
        for c in contours:
            x,y,w,h=cv2.boundingRect(c)
            if w*h>900: out.append(Detection(x,y,x+w,y+h,0,.5))
        return out

class TensorRTDetector(Detector):
    name = 'yolo-tensorrt'
    precision = 'FP16'
    input_size = 640

    def __init__(self, engine_path, confidence=.25, iou=.45, max_candidates=0,
                 reuse_preprocess=True, class_confidences=None):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
        except ImportError as exc:
            raise RuntimeError('TensorRT detector requires working tensorrt and pycuda installations') from exc
        self.name = Path(engine_path).stem.rsplit('_fp16', 1)[0] + '-tensorrt'
        self.trt, self.cuda = trt, cuda
        self.confidence, self.iou = confidence, iou
        self.max_box_aspect_ratio = float(os.getenv('SCOUT_MAX_BOX_ASPECT_RATIO', '0'))
        if self.max_box_aspect_ratio != 0 and self.max_box_aspect_ratio < 1:
            raise ValueError('SCOUT_MAX_BOX_ASPECT_RATIO must be 0 (disabled) or at least 1')
        self.last_shape_rejected = 0
        try:
            self.class_names = json.loads(os.getenv('SCOUT_CLASS_NAMES', 'null')) or COCO_NAMES
        except json.JSONDecodeError as exc:
            raise ValueError('SCOUT_CLASS_NAMES must be a JSON array of labels') from exc
        if (not isinstance(self.class_names, list) or not self.class_names or
                any(not isinstance(label, str) or not label for label in self.class_names)):
            raise ValueError('SCOUT_CLASS_NAMES must be a non-empty JSON array of labels')
        self.class_confidences = {int(key): float(value) for key, value in
                                  (class_confidences or {}).items()}
        if any(cid < 0 or cid >= len(self.class_names) or not 0.1 <= threshold <= 0.95
               for cid, threshold in self.class_confidences.items()):
            raise ValueError('Class confidences require valid model ids and values from 0.10 to 0.95')
        self.max_candidates = max(0, int(max_candidates))
        self.reuse_preprocess = bool(reuse_preprocess)
        self.lock = threading.Lock()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        cuda.init()
        self.cuda_context = cuda.Device(0).retain_primary_context()
        self.cuda_context.push()
        try:
            with open(engine_path, 'rb') as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(f'Unable to load TensorRT engine: {engine_path}')
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs = [], []
            self.stream = cuda.Stream()
            self.profile_events = tuple(cuda.Event() for _ in range(6))
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = tuple(self.engine.get_tensor_shape(name))
                if any(d <= 0 for d in shape):
                    raise ValueError('SCOUT requires a static-shape YOLO engine')
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                device = cuda.mem_alloc(host.nbytes)
                item = (name, shape, dtype, host, device)
                target = self.inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.outputs
                target.append(item)
                self.context.set_tensor_address(name, int(device))
            if len(self.inputs) != 1 or len(self.outputs) != 1:
                raise ValueError('SCOUT requires one YOLO input and one raw detection output (nms=False)')
            self.input_name, self.input_shape = self.inputs[0][:2]
            if len(self.input_shape) != 4 or self.input_shape[:2] != (1, 3):
                raise ValueError('SCOUT requires a batch-one NCHW input')
            self.input_height, self.input_width = self.input_shape[2:]
            self.input_size = self.input_width if self.input_width == self.input_height else None
            self.input_resolution = f'{self.input_width}x{self.input_height}'
            if self.reuse_preprocess:
                _, _, input_h, input_w = self.input_shape
                self.canvas = np.empty((input_h, input_w, 3), dtype=np.uint8)
                self.rgb = np.empty_like(self.canvas)
                self.input_view = self.inputs[0][3].reshape(self.input_shape)
        finally:
            self.cuda_context.pop()

    def infer(self, frame):
        # FastAPI executes inference in worker threads. CUDA contexts are thread-local,
        # and the shared execution context/buffers must not be used concurrently.
        with self.lock:
            self.cuda_context.push()
            try:
                return self._infer(frame)
            finally:
                self.cuda_context.pop()

    def close(self):
        """Release CUDA allocations before replacing this engine at runtime."""
        self.cuda_context.push()
        try:
            for _, _, _, _, device in self.inputs + self.outputs:
                device.free()
            self.inputs.clear()
            self.outputs.clear()
        finally:
            self.cuda_context.pop()
            self.cuda_context.detach()

    def _infer(self, frame):
        started = time.perf_counter()
        h, w = frame.shape[:2]
        input_h, input_w = self.input_height, self.input_width
        scale, nw, nh, ox, oy = letterbox_geometry(w, h, input_w, input_h)
        pre_started = time.perf_counter()
        if self.reuse_preprocess:
            canvas = self.canvas
            canvas.fill(114)
            target = canvas[oy:oy + nh, ox:ox + nw]
            if (nw, nh) == (w, h):
                np.copyto(target, frame)
            else:
                cv2.resize(frame, (nw, nh), dst=target)
            cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB, dst=self.rgb)
            np.multiply(self.rgb.transpose(2, 0, 1), 1. / 255., out=self.input_view[0])
        else:
            canvas = np.full((input_h, input_w, 3), 114, np.uint8)
            if (nw, nh) == (w, h):
                canvas[oy:oy + nh, ox:ox + nw] = frame
            else:
                canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(frame, (nw, nh))
            blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(self.inputs[0][2]) / 255.
            np.copyto(self.inputs[0][3], blob.ravel())
        preprocess_ms = (time.perf_counter() - pre_started) * 1000
        h2d_start, h2d_end, trt_start, trt_end, d2h_start, d2h_end = self.profile_events
        h2d_start.record(self.stream)
        self.cuda.memcpy_htod_async(self.inputs[0][4], self.inputs[0][3], self.stream)
        h2d_end.record(self.stream)
        trt_start.record(self.stream)
        if not self.context.execute_async_v3(self.stream.handle):
            raise RuntimeError('TensorRT inference failed')
        trt_end.record(self.stream)
        d2h_start.record(self.stream)
        for _, _, _, host, device in self.outputs:
            self.cuda.memcpy_dtoh_async(host, device, self.stream)
        d2h_end.record(self.stream)
        self.stream.synchronize()
        h2d_ms = h2d_start.time_till(h2d_end)
        trt_ms = trt_start.time_till(trt_end)
        d2h_ms = d2h_start.time_till(d2h_end)
        output = self.outputs[0][3].reshape(self.outputs[0][1])
        post_started = time.perf_counter()
        detections = self._decode(output, scale, ox, oy, w, h)
        postprocess_ms = (time.perf_counter() - post_started) * 1000
        total_ms = (time.perf_counter() - started) * 1000
        self.last_profile = dict(preprocess_ms=round(preprocess_ms, 2), h2d_ms=round(h2d_ms, 2),
                                 tensorrt_ms=round(trt_ms, 2), d2h_ms=round(d2h_ms, 2),
                                 postprocess_ms=round(postprocess_ms, 2), detector_total_ms=round(total_ms, 2),
                                 nms_candidates=self.last_candidate_count, max_score=self.last_max_score,
                                 shape_rejected=self.last_shape_rejected)
        return detections, total_ms

    def _decode(self, output, scale, ox, oy, w, h):
        self.last_shape_rejected = 0
        output = np.asarray(output)
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        if output.ndim == 2 and output.shape[1] == 6:
            # YOLO26 exports an end-to-end head as up to 300 xyxy, score, class
            # rows. Boxes are already decoded, but still apply the configured
            # class-wise NMS: the exported head can leave highly overlapping
            # duplicate boxes in its output.
            rows = output[np.isfinite(output).all(axis=1)]
            valid_class = (rows[:, 5] >= 0) & (rows[:, 5] < len(self.class_names))
            rows = rows[valid_class]
            scores = rows[:, 4]
            ids = rows[:, 5].astype(np.int32)
            thresholds = np.asarray([self.class_confidences.get(int(cid), self.confidence)
                                     for cid in ids], dtype=np.float32)
            self.last_max_score = float(scores.max()) if len(scores) else 0.
            rows = rows[scores >= thresholds]
            self.last_candidate_count = len(rows)
            detections = []
            for x1, y1, x2, y2, confidence, class_id in rows:
                x1 = int(np.clip((x1 - ox) / scale, 0, w))
                y1 = int(np.clip((y1 - oy) / scale, 0, h))
                x2 = int(np.clip((x2 - ox) / scale, 0, w))
                y2 = int(np.clip((y2 - oy) / scale, 0, h))
                if x2 > x1 and y2 > y1:
                    detections.append(Detection(x1, y1, x2, y2, int(class_id),
                                                float(confidence)))
            keep = []
            for cid in sorted({detection.class_id for detection in detections}):
                indices = [i for i, detection in enumerate(detections)
                           if detection.class_id == cid]
                boxes = [[detections[i].x1, detections[i].y1,
                          detections[i].x2 - detections[i].x1,
                          detections[i].y2 - detections[i].y1] for i in indices]
                scores = [detections[i].confidence for i in indices]
                selected = cv2.dnn.NMSBoxes(
                    boxes, scores, self.class_confidences.get(cid, self.confidence), self.iou)
                keep.extend(indices[int(i)] for i in np.asarray(selected).reshape(-1))
            detections = [detections[i] for i in keep]
            detections.sort(key=lambda detection: detection.confidence, reverse=True)
            return self._filter_box_shapes(detections)
        channels = 4 + len(self.class_names)
        if output.ndim != 2:
            raise ValueError(f'Unsupported YOLO output shape: {output.shape}')
        if output.shape[0] == channels:
            rows = output.T
        elif output.shape[1] == channels:
            rows = output
        else:
            raise ValueError(f'Expected raw COCO-80 YOLO output, got {output.shape}')
        # Filter the dense head in NumPy before iterating candidate boxes.
        finite = np.isfinite(rows).all(axis=1)
        scores = rows[:, 4:].max(axis=1)
        self.last_max_score = float(scores[finite].max()) if finite.any() else 0.
        min_threshold = min([self.confidence] + list(self.class_confidences.values()))
        rows = rows[finite & (scores >= min_threshold)]
        max_candidates = getattr(self, 'max_candidates', 0)
        if max_candidates and len(rows) > max_candidates:
            scores = rows[:, 4:].max(axis=1)
            rows = rows[np.argpartition(scores, -max_candidates)[-max_candidates:]]
        self.last_candidate_count = len(rows)
        boxes, confidences, ids = [], [], []
        for row in rows:
            class_scores = row[4:]
            cid = int(np.argmax(class_scores))
            confidence = float(class_scores[cid])
            if confidence < self.class_confidences.get(cid, self.confidence):
                continue
            cx, cy, bw, bh = row[:4]
            x1 = int(np.clip((cx - bw / 2 - ox) / scale, 0, w))
            y1 = int(np.clip((cy - bh / 2 - oy) / scale, 0, h))
            x2 = int(np.clip((cx + bw / 2 - ox) / scale, 0, w))
            y2 = int(np.clip((cy + bh / 2 - oy) / scale, 0, h))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(confidence)
            ids.append(cid)
        # Suppress duplicates within each class, retaining overlapping different objects.
        keep = []
        for cid in sorted(set(ids)):
            indices = [i for i, value in enumerate(ids) if value == cid]
            selected = cv2.dnn.NMSBoxes([boxes[i] for i in indices],
                                       [confidences[i] for i in indices],
                                       self.class_confidences.get(cid, self.confidence), self.iou)
            keep.extend(indices[int(i)] for i in np.asarray(selected).reshape(-1))
        keep.sort(key=lambda i: confidences[i], reverse=True)
        detections = [Detection(boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2],
                                boxes[i][1] + boxes[i][3], ids[i], confidences[i]) for i in keep]
        return self._filter_box_shapes(detections)

    def _filter_box_shapes(self, detections):
        limit = self.max_box_aspect_ratio
        if not limit:
            return detections
        kept = []
        for detection in detections:
            width = detection.x2 - detection.x1
            height = detection.y2 - detection.y1
            if max(width / height, height / width) <= limit:
                kept.append(detection)
        self.last_shape_rejected = len(detections) - len(kept)
        return kept


class ConfidenceCascadeDetector(Detector):
    """Run a fast engine first and use a larger engine on ambiguous output."""
    name = 'yolo-tensorrt-confidence-cascade'
    precision = 'FP16'

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, value):
        self._confidence = float(value)
        for name in ('fast', 'fallback'):
            detector = getattr(self, name, None)
            if detector is not None:
                detector.confidence = self._confidence

    def __init__(self, fast_engine_path, fallback_engine_path, confidence=.25,
                 iou=.45, max_candidates=0, reuse_preprocess=True,
                 class_confidences=None):
        self.fast = TensorRTDetector(fast_engine_path, confidence, iou,
                                     max_candidates, reuse_preprocess,
                                     class_confidences)
        try:
            self.fallback = TensorRTDetector(fallback_engine_path, confidence, iou,
                                             max_candidates, reuse_preprocess,
                                             class_confidences)
        except Exception:
            self.fast.close()
            raise
        self.name = (f'{Path(fast_engine_path).stem}-cascade-'
                     f'{Path(fallback_engine_path).stem}-tensorrt')
        self.confidence = confidence
        self.iou = iou
        self.class_names = self.fast.class_names
        self.class_confidences = self.fast.class_confidences
        self.max_candidates = self.fast.max_candidates
        self.max_box_aspect_ratio = self.fast.max_box_aspect_ratio
        self.reuse_preprocess = self.fast.reuse_preprocess
        self.input_resolution = (f"{self.fast.input_resolution} primary; "
                                 f"{self.fallback.input_resolution} fallback")
        self.min_confidence_fallback = float(os.getenv(
            'SCOUT_CASCADE_MIN_CONFIDENCE', '0.06'))
        if not 0 <= self.min_confidence_fallback <= 1:
            self.close()
            raise ValueError('SCOUT_CASCADE_MIN_CONFIDENCE must be between 0 and 1')
        self.count_agnostic = os.getenv(
            'SCOUT_CASCADE_COUNT_AGNOSTIC', '0').lower() in ('1', 'true', 'yes')
        self.lock = threading.RLock()
        self.last_profile = {}
        self.total_frames = 0
        self.fallback_frames = 0

    def infer(self, frame):
        with self.lock:
            fast_detections, fast_ms = self.fast.infer(frame)
            reason = None
            if self.count_agnostic and not fast_detections:
                reason = 'empty_candidate'
            elif not self.count_agnostic and len(fast_detections) != 2:
                reason = 'candidate_count'
            elif min(item.confidence for item in fast_detections) < self.min_confidence_fallback:
                reason = 'weak_candidate'

            fallback_ms = 0.0
            if reason:
                detections, fallback_ms = self.fallback.infer(frame)
                selected_profile = self.fallback.last_profile
                self.fallback_frames += 1
            else:
                detections = fast_detections
                selected_profile = self.fast.last_profile
            self.total_frames += 1

            self.last_profile = dict(selected_profile)
            self.last_profile.update(
                detector_total_ms=round(fast_ms + fallback_ms, 2),
                cascade_fast_ms=round(fast_ms, 2),
                cascade_fallback_ms=round(fallback_ms, 2),
                cascade_fallback=bool(reason),
                cascade_fallback_reason=reason or 'none',
                cascade_count_agnostic=self.count_agnostic,
                cascade_total_frames=self.total_frames,
                cascade_fallback_frames=self.fallback_frames,
                cascade_fallback_fraction=round(
                    self.fallback_frames / self.total_frames, 5),
                cascade_fast_candidate_count=len(fast_detections),
                cascade_fast_min_confidence=round(
                    min((item.confidence for item in fast_detections), default=0.0), 4))
            return detections, fast_ms + fallback_ms

    def close(self):
        fast = getattr(self, 'fast', None)
        fallback = getattr(self, 'fallback', None)
        if fast is not None:
            fast.close()
            self.fast = None
        if fallback is not None:
            fallback.close()
            self.fallback = None


def create_detector(mode='motion', engine_path='models/yolo11n_640_fp16.engine', confidence=.25,
                    iou=.45, max_candidates=0, reuse_preprocess=True, class_confidences=None):
    if mode == 'tensorrt':
        fast_engine_path = os.getenv('SCOUT_CASCADE_FAST_ENGINE')
        if fast_engine_path:
            fallback_engine_path = os.getenv('SCOUT_CASCADE_FALLBACK_ENGINE', engine_path)
            return ConfidenceCascadeDetector(
                fast_engine_path, fallback_engine_path, confidence, iou,
                max_candidates, reuse_preprocess, class_confidences)
        return TensorRTDetector(engine_path, confidence, iou, max_candidates, reuse_preprocess,
                                class_confidences)
    if mode == 'motion':
        return MotionDetector()
    raise ValueError(f'Unknown detector mode: {mode}')
