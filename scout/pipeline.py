"""One hardware video path and one bounded, independent inference worker."""
import json
import gc
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from .gst_runtime import Gst, GstVideo, GstPbutils, GLib
import cv2
import numpy as np
from .detectors import COCO_NAMES, Detection, create_detector
from .stereo import StereoDepth, CharucoCollector
from .tracking import FuelTracker
from .nt4 import NT4Publisher

LOG = logging.getLogger(__name__)


class LatestFrame:
    """A single pending frame. A slow consumer never creates a frame backlog."""
    def __init__(self):
        self.condition = threading.Condition()
        self.value = None
        self.closed = False
        self.replaced = 0

    def put(self, value):
        with self.condition:
            if self.closed:
                return
            if self.value is not None:
                self.replaced += 1
            self.value = value
            self.condition.notify()

    def get(self):
        with self.condition:
            self.condition.wait_for(lambda: self.closed or self.value is not None)
            if self.closed:
                return None
            value, self.value = self.value, None
            return value

    def clear(self):
        with self.condition:
            self.value = None

    def close(self):
        with self.condition:
            self.closed = True
            self.value = None
            self.condition.notify_all()


def rtp_timestamp(pts):
    return (int(pts) * 90000 // Gst.SECOND) & 0xffffffff


class VideoPipeline:
    def __init__(self, source, mode='tensorrt', engine='models/yolo11n_640_fp16.engine',
                 source_id=None, source_name=None, nt_table=None,
                 stereo_calibration=None, stereo_baseline_m=None,
                 stereo_focal_px=None, stereo_use_global_config=True):
        self.source, self.mode, self.engine_path = str(source), mode, engine
        self.source_id, self.source_name = source_id, source_name
        self.output_width = int(os.getenv('SCOUT_OUTPUT_WIDTH', '640'))
        self.output_height = int(os.getenv('SCOUT_OUTPUT_HEIGHT', '360'))
        self.inference_width = int(os.getenv('SCOUT_INFERENCE_WIDTH', '2560'))
        self.flow_width = int(os.getenv('SCOUT_FLOW_WIDTH', '1280'))
        self.inference_max_fps = float(os.getenv('SCOUT_INFERENCE_MAX_FPS', '5'))
        if not 0 <= self.inference_max_fps <= 60:
            raise ValueError('SCOUT_INFERENCE_MAX_FPS must be between 0 (unlimited) and 60')
        self.last_inference_started = 0.0
        self.playback_rate = float(os.getenv('SCOUT_PLAYBACK_RATE', '1.0'))
        self.target_fps = float(os.getenv('SCOUT_TARGET_FPS', '60'))
        self.replay_sync = os.getenv('SCOUT_REPLAY_SYNC', '1').lower() not in (
            '0', 'false', 'no')
        self.stream_enabled = os.getenv('SCOUT_STREAM_ENABLED', '1').lower() not in (
            '0', 'false', 'no')
        self.effective_playback_rate = self.playback_rate
        if self.playback_rate <= 0:
            raise ValueError('Playback rate must be greater than zero')
        self.lock = threading.RLock()
        self.pause_lock = threading.Lock()
        self.source_lock = threading.RLock()
        self.slot = LatestFrame()
        self.flow_slot = LatestFrame()
        self.flow_gate = threading.Condition()
        self.flow_busy = False
        self.right_frames = {}
        self.flow_right_frames = {}
        self.right_frame_condition = threading.Condition()
        self.stopped = threading.Event()
        self.paused = threading.Event()
        self.switching = threading.Event()
        self.inference_gate = threading.Condition()
        self.inference_busy = False
        self.ready = threading.Event()
        self.clock = Gst.SystemClock.obtain()
        self.origin = None
        self.epoch = 0
        self.new_epoch = True
        self.capture_times = {}
        self.capture_times_lock = threading.Lock()
        self.sequence = 0
        self.peers = set()
        self.rtp_caps = None
        self.detections = None
        self.detector = None
        self.stereo = StereoDepth(
            stereo_calibration, baseline_m=stereo_baseline_m,
            focal_px=stereo_focal_px,
            use_environment_defaults=stereo_use_global_config)
        self.stereo_lock = threading.Lock()
        self.stereo_executor = ThreadPoolExecutor(max_workers=1,
                                                  thread_name_prefix='stereo-map')
        self.stereo_pair_wait_s = max(
            0.0, float(os.getenv('SCOUT_STEREO_PAIR_WAIT_MS', '2')) / 1000.0)
        self.flow_right_cache_size = max(
            2, int(os.getenv('SCOUT_FLOW_RIGHT_CACHE_FRAMES', '4')))
        self.last_depth_boxes = []
        self.fuel_tracker = FuelTracker()
        self.nt_publisher = (NT4Publisher(nt_table) if os.getenv(
            'FRC_NT_ENABLED', '0').lower() in ('1', 'true', 'yes') else None)
        self.inference_always = os.getenv('SCOUT_INFERENCE_ALWAYS', '0').lower() in ('1', 'true', 'yes')
        self.normal_confidence = float(os.getenv('SCOUT_CONFIDENCE', '.60'))
        self.low_light_gamma = float(os.getenv('SCOUT_LOW_LIGHT_GAMMA', '0'))
        self.low_light_enabled = self.low_light_gamma > 0
        self.low_light_luma_threshold = float(os.getenv('SCOUT_LOW_LIGHT_LUMA_THRESHOLD', '30'))
        self.low_light_confidence = float(os.getenv('SCOUT_LOW_LIGHT_CONFIDENCE', '0.02'))
        if self.low_light_gamma < 0 or self.low_light_gamma > 1:
            raise ValueError('SCOUT_LOW_LIGHT_GAMMA must be zero (disabled) or in (0, 1]')
        if not 0 <= self.low_light_luma_threshold <= 255:
            raise ValueError('SCOUT_LOW_LIGHT_LUMA_THRESHOLD must be between 0 and 255')
        if (self.low_light_enabled and
                not 0 < self.low_light_confidence <= self.normal_confidence):
            raise ValueError('SCOUT_LOW_LIGHT_CONFIDENCE must be positive and no greater than SCOUT_CONFIDENCE')
        self.low_light_lut = (np.clip((np.arange(256, dtype=np.float32) / 255.0) **
                                      self.low_light_gamma * 255, 0, 255).astype(np.uint8)
                              if self.low_light_enabled else None)
        self.low_light_frame_count = 0
        self.low_light_last_mean_luma = None
        self.adaptive_confidence_trigger = int(os.getenv('SCOUT_ADAPTIVE_CONFIDENCE_TRIGGER', '3'))
        self.adaptive_confidence_high = float(os.getenv('SCOUT_ADAPTIVE_CONFIDENCE_HIGH', '0.20'))
        if self.adaptive_confidence_trigger < 0:
            raise ValueError('SCOUT_ADAPTIVE_CONFIDENCE_TRIGGER must be zero or positive')
        if not 0.0 < self.adaptive_confidence_high <= 0.95:
            raise ValueError('SCOUT_ADAPTIVE_CONFIDENCE_HIGH must be greater than zero and at most 0.95')
        self.calibrator = CharucoCollector(self.stereo.calibration_path)
        self.stereo_source = False
        self.pipeline = None
        self.worker = None
        self.flow_worker = None
        self.mainloop = None
        self.mainloop_thread = None
        self.counters = dict(capture=0, inference=0, inference_skipped=0,
                             flow=0, encoded=0)
        self.previous = self.counters.copy()
        self.rate_started = time.monotonic()
        self.stats = dict(status='starting', error=None,
                          transport='WebRTC / VP9' if self.stream_enabled else 'disabled',
                          stream_enabled=self.stream_enabled,
                          encoder='nvv4l2vp9enc' if self.stream_enabled else 'disabled',
                          capture_fps=0., inference_fps=0.,
                          encoded_fps=0., inference_latency_ms=0., encode_latency_ms=0.,
                          frame_copy_ms=0., stereo_copy_ms=0., preprocess_ms=0., h2d_ms=0., tensorrt_ms=0.,
                          d2h_ms=0., postprocess_ms=0., detector_total_ms=0.,
                          input_resolution='unknown',
                          engine_input_resolution='unknown',
                          output_resolution=f'{self.output_width}x{self.output_height}',
                          source_fps=0., playback_rate=self.playback_rate, loops=0,
                          stereo=False, stereo_calibrated=False, stereo_approximate=False,
                          stereo_rectified=False,
                          stereo_match_scale=self.stereo.match_scale,
                          stereo_backend=self.stereo.backend,
                          stereo_num_disparities=self.stereo.num_disparities,
                          stereo_block_size=self.stereo.block_size,
                          stereo_active_num_disparities=self.stereo.last_num_disparities,
                          stereo_horizontal_roi=self.stereo.horizontal_roi_enabled,
                          stereo_adaptive_disparity_width=self.stereo.adaptive_disparity_width,
                          stereo_local_rectify=self.stereo.local_rectify_enabled,
                          stereo_min_disparity=self.stereo.last_min_disparity,
                          stereo_vertical_roi_pad=self.stereo.vertical_roi_pad,
                          stereo_size_gate_max_relative_error=self.stereo.size_gate_max_relative_error,
                          stereo_size_gate_rejected=0,
                          stereo_size_gate_duplicate_iou=self.stereo.size_gate_duplicate_iou,
                          stereo_size_gate_duplicate_rejected=0,
                          stereo_flow_frames=0, stereo_flow_fresh_tracks=0,
                          stereo_flow_missing_pairs=0,
                          stereo_flow_estimate_ms=0.,
                          adaptive_confidence_trigger=self.adaptive_confidence_trigger,
                          adaptive_confidence_high=self.adaptive_confidence_high,
                          adaptive_confidence_frames=0,
                          adaptive_confidence_removed_boxes=0,
                          tracking_max_age_ms=round(self.fuel_tracker.max_age_s * 1000.0, 1),
                          tracking_flow_grace_ms=round(self.fuel_tracker.flow_grace_s * 1000.0, 1),
                          tracking_decay_ms=round(self.fuel_tracker.decay_s * 1000.0, 1),
                          tracked_instances=0, predicted_instances=0,
                          nt_enabled=self.nt_publisher is not None,
                          nt_connected=False, nt_status='disabled', nt_frame=0,
                          low_light_gamma=self.low_light_gamma,
                          low_light_luma_threshold=self.low_light_luma_threshold,
                          low_light_confidence=self.low_light_confidence,
                          low_light_frames=0,
                          low_light_mean_luma=None,
                          source_id=source_id, source_name=source_name)

    def _source_description(self):
        self.loop_file = False
        self.width, self.height = 1920, 1080
        if self.source.startswith('test://'):
            fps = int(self.source.split('://', 1)[1])
            if fps < 1 or fps > 120:
                raise ValueError('Test source FPS must be between 1 and 120')
            self.stats['decoder'] = 'synthetic'
            return (f'videotestsrc is-live=true pattern=ball ! '
                    f'video/x-raw,format=NV12,width=1920,height=1080,framerate={fps}/1 ! '
                    'nvvidconv ! video/x-raw(memory:NVMM),format=NV12')
        if self.source.isdigit() or self.source.startswith('/dev/video'):
            device = '/dev/video' + self.source if self.source.isdigit() else self.source
            camera_format = os.getenv('SCOUT_CAMERA_FORMAT', 'YUY2').upper()
            camera_width = int(os.getenv('SCOUT_CAMERA_WIDTH', '640'))
            camera_height = int(os.getenv('SCOUT_CAMERA_HEIGHT', '240'))
            camera_fps = int(os.getenv('SCOUT_CAMERA_FPS', '60'))
            if camera_format not in ('YUY2', 'MJPG'):
                raise ValueError('SCOUT_CAMERA_FORMAT must be YUY2 or MJPG')
            if camera_width < 2 or camera_height < 2 or camera_fps < 1:
                raise ValueError('Camera width, height, and FPS must be positive')
            self.width, self.height = camera_width, camera_height
            self.stereo_source = camera_width % 2 == 0 and camera_width >= 4
            self.stats.update(decoder='raw camera', camera_device=device,
                              camera_format=camera_format, stereo=self.stereo_source,
                              stereo_calibrated=self.stereo.calibrated,
                              stereo_approximate=self.stereo.approximate,
                              stereo_rectified=self.stereo.rectify,
                              stereo_baseline_m=self.stereo.baseline_m,
                              stereo_focal_px=self.stereo.focal_px,
                              stereo_disparity_scale=self.stereo.disparity_scale)
            media_type = 'image/jpeg' if camera_format == 'MJPG' else 'video/x-raw'
            decoder = ' ! nvjpegdec' if camera_format == 'MJPG' else ''
            return (f'v4l2src device={json.dumps(device)} io-mode=2 ! '
                    f'{media_type},width={camera_width},height={camera_height},framerate={camera_fps}/1'
                    f'{decoder} ! '
                    'nvvidconv ! video/x-raw(memory:NVMM),format=NV12')
        self.stats['decoder'] = 'nvv4l2decoder'
        if self.source.startswith('rtsp://'):
            return (f'rtspsrc location={json.dumps(self.source)} protocols=tcp latency=100 ! '
                    'rtph264depay ! h264parse ! nvv4l2decoder')
        if self.source.startswith('tcp://'):
            parsed = urlparse(self.source)
            if not parsed.hostname or not parsed.port:
                raise ValueError('TCP source requires a host and port')
            return (f'tcpclientsrc host={json.dumps(parsed.hostname)} port={parsed.port} ! '
                    'tsdemux ! h264parse ! nvv4l2decoder')
        if self.source.startswith('mjpeg-tcp://'):
            parsed = urlparse(self.source.replace('mjpeg-tcp://', 'tcp://', 1))
            if not parsed.hostname or not parsed.port:
                raise ValueError('MJPEG TCP source requires a host and port')
            self.width = int(os.getenv('SCOUT_CAMERA_WIDTH', '3200'))
            self.height = int(os.getenv('SCOUT_CAMERA_HEIGHT', '1200'))
            camera_fps = int(os.getenv('SCOUT_CAMERA_FPS', '60'))
            self.stereo_source = (self.source_id or '').startswith('stereo-')
            self.stats.update(decoder='multipart MJPEG TCP relay',
                              camera_device=f'{parsed.hostname}:{parsed.port}',
                              stereo=self.stereo_source,
                              stereo_calibrated=self.stereo.calibrated,
                              stereo_approximate=self.stereo.approximate,
                              stereo_rectified=self.stereo.rectify,
                              stereo_baseline_m=self.stereo.baseline_m,
                              stereo_focal_px=self.stereo.focal_px,
                              stereo_disparity_scale=self.stereo.disparity_scale)
            return (f'tcpclientsrc host={json.dumps(parsed.hostname)} port={parsed.port} ! '
                    'multipartdemux ! '
                    f'image/jpeg,width={self.width},height={self.height},framerate={camera_fps}/1 ! '
                    'nvjpegdec ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12')
        path = Path(self.source).resolve(strict=True)
        info = GstPbutils.Discoverer.new(5 * Gst.SECOND).discover_uri(path.as_uri())
        streams = info.get_video_streams()
        if not streams:
            raise ValueError('Source has no video stream')
        stream = streams[0]
        self.width, self.height = stream.get_width(), stream.get_height()
        # Stereo catalog clips are synchronized left|right side-by-side views.
        # Keep the same split used for side-by-side USB camera input.
        self.stereo_source = ((self.source_id or '').startswith('stereo-'))
        if self.stereo_source:
            self.stats.update(stereo=True, stereo_calibrated=self.stereo.calibrated,
                              stereo_approximate=self.stereo.approximate,
                              stereo_rectified=self.stereo.rectify,
                              stereo_baseline_m=self.stereo.baseline_m,
                              stereo_focal_px=self.stereo.focal_px,
                              stereo_disparity_scale=self.stereo.disparity_scale)
        caps = stream.get_caps().get_structure(0)
        if caps.get_name() != 'video/x-h264':
            raise ValueError('File playback currently requires H.264 video in an MP4 container')
        ok, numerator, denominator = caps.get_fraction('framerate')
        if ok and denominator:
            source_fps = numerator / denominator
            if source_fps > 0:
                self.effective_playback_rate = min(self.playback_rate, self.target_fps / source_fps)
                self.stats.update(source_fps=source_fps,
                                  playback_rate=round(self.effective_playback_rate, 4))
        self.loop_file = True
        return (f'filesrc location={json.dumps(str(path))} ! qtdemux name=demux '
                'demux.video_0 ! queue ! h264parse ! nvv4l2decoder')

    def _build_description(self):
        source = self._source_description()
        width = min(self.inference_width, self.width)
        height = max(2, round(self.height * width / self.width / 2) * 2)
        view_width = width // 2 if self.stereo_source else width
        flow_width = min(self.flow_width, self.width)
        flow_height = max(2, round(self.height * flow_width / self.width / 2) * 2)
        flow_view_width = flow_width // 2 if self.stereo_source else flow_width
        self.inference_view_width, self.inference_view_height = view_width, height
        self.flow_view_width, self.flow_view_height = flow_view_width, flow_height
        self.stats['inference_resolution'] = f'{view_width}x{height}'
        self.stats['tracking_resolution'] = f'{flow_view_width}x{flow_height}'
        display_prefix = 'nvvidconv'
        display_caps = f'video/x-raw(memory:NVMM),format=NV12,width={self.output_width},height={self.output_height}'
        if self.stereo_source:
            # The USB camera delivers left|right. Stream the left sensor only;
            # the right sensor remains available to the depth worker below.
            display_prefix = (f'nvvidconv ! video/x-raw,format=I420,width={self.width},height={self.height} ! '
                              f'videocrop right={self.width // 2} ! nvvidconv')
        if self.stereo_source:
            inference_branch = (
                'video. ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! '
                f'nvvidconv ! video/x-raw,format=BGRx,width={width},height={height} ! tee name=inference-views '
                'inference-views. ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! '
                f'videocrop right={view_width} ! videoconvert ! video/x-raw,format=BGR ! '
                'appsink name=inference emit-signals=true sync=false max-buffers=1 drop=true enable-last-sample=false '
            )
            flow_branch = (
                'video. ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! '
                f'nvvidconv ! video/x-raw,format=BGRx,width={flow_width},height={flow_height} ! '
                'videoconvert ! video/x-raw,format=GRAY8 ! '
                'appsink name=flow-stereo emit-signals=true sync=false max-buffers=1 drop=true enable-last-sample=false '
            )
        else:
            inference_branch = (
                'video. ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! '
                f'nvvidconv ! video/x-raw,format=BGRx,width={width},height={height} ! '
                'videoconvert ! video/x-raw,format=BGR ! '
                'appsink name=inference emit-signals=true sync=false max-buffers=1 drop=true enable-last-sample=false '
            )
            flow_branch = ''
        recording_branch = ''
        recording_path = os.getenv('SCOUT_RECORD_PATH')
        if recording_path:
            recording_frames = int(os.getenv('SCOUT_RECORD_FRAMES', '600'))
            recording_bitrate = int(os.getenv('SCOUT_RECORD_BITRATE', '30000000'))
            if recording_frames < 1 or recording_bitrate < 1:
                raise ValueError('SCOUT_RECORD_FRAMES and SCOUT_RECORD_BITRATE must be positive')
            Path(recording_path).resolve().parent.mkdir(parents=True, exist_ok=True)
            recording_branch = (
                'video. ! queue max-size-buffers=16 max-size-bytes=0 max-size-time=0 ! '
                'videorate ! '
                f'video/x-raw(memory:NVMM),format=NV12,width={self.width},height={self.height},framerate=60/1 ! '
                f'identity eos-after={recording_frames} ! '
                f'nvv4l2h264enc bitrate={recording_bitrate} iframeinterval=60 '
                'insert-sps-pps=true maxperf-enable=true ! h264parse ! '
                'capssetter caps="video/x-h264,framerate=60/1" replace=false ! qtmux faststart=true ! '
                f'filesink location={json.dumps(str(Path(recording_path).resolve()))} sync=false '
            )
        self.description = (
            source + f' ! identity name=pace sync={str(self.replay_sync).lower()} signal-handoffs=true single-segment=true ! tee name=video '
            + inference_branch + flow_branch + recording_branch
        )
        if self.stream_enabled:
            self.description += (
                'video. ! queue max-size-buffers=4 max-size-bytes=0 max-size-time=0 leaky=downstream ! '
                f'{display_prefix} ! {display_caps} ! '
                'nvv4l2vp9enc name=encoder bitrate=2500000 iframeinterval=60 '
                'maxperf-enable=true ! rtpvp9pay name=pay pt=96 '
                'timestamp-offset=0 seqnum-offset=0 ssrc=1396911957 ! '
                'appsink name=rtp emit-signals=true sync=false max-buffers=32 drop=false enable-last-sample=false')
        else:
            self.description += (
                'video. ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 '
                'leaky=downstream ! fakesink name=stream-disabled sync=false')

    def start(self):
        try:
            required = ('nvv4l2decoder', 'nvvidconv')
            if self.stream_enabled:
                required += ('nvv4l2vp9enc', 'rtpvp9pay', 'h264parse',
                             'webrtcbin', 'nicesrc', 'nicesink')
            missing = [name for name in required if Gst.ElementFactory.find(name) is None]
            if missing:
                raise RuntimeError('Missing GStreamer plugins: ' + ', '.join(missing))
            reuse_preprocess = os.getenv('SCOUT_REUSE_PREPROCESS', '1').lower() not in ('0', 'false', 'no')
            try:
                class_confidences = json.loads(os.getenv('SCOUT_CLASS_CONFIDENCES', '{}'))
                if not isinstance(class_confidences, dict):
                    raise ValueError('SCOUT_CLASS_CONFIDENCES must be a JSON object keyed by COCO class id')
            except json.JSONDecodeError as exc:
                raise ValueError('SCOUT_CLASS_CONFIDENCES must be valid JSON') from exc
            self.detector = create_detector(
                self.mode, self.engine_path,
                confidence=(min(self.normal_confidence, self.low_light_confidence)
                            if self.low_light_enabled else self.normal_confidence),
                iou=float(os.getenv('SCOUT_NMS_IOU', '.10')),
                max_candidates=int(os.getenv('SCOUT_MAX_CANDIDATES', '0')),
                reuse_preprocess=reuse_preprocess,
                class_confidences=class_confidences)
            base_confidence = self.detector.confidence
            if self.adaptive_confidence_trigger and self.adaptive_confidence_high < base_confidence:
                raise ValueError('SCOUT_ADAPTIVE_CONFIDENCE_HIGH must be at least the detector confidence')
            self.stats.update(detector=self.detector.name, precision=self.detector.precision)
            if self.mode == 'tensorrt':
                self.stats.update(confidence=self.normal_confidence,
                                  nms_iou=getattr(self.detector, 'iou', None),
                                  class_confidences=self.detector.class_confidences,
                                  engine_input_resolution=self.detector.input_resolution,
                                  max_candidates=self.detector.max_candidates or None,
                                  max_box_aspect_ratio=self.detector.max_box_aspect_ratio or None,
                                  reuse_preprocess=self.detector.reuse_preprocess)
            self._build_description()
            self.pipeline = self._new_pipeline()
            self.mainloop = GLib.MainLoop()
            self.mainloop_thread = threading.Thread(target=self.mainloop.run, name='gst-main', daemon=True)
            self.mainloop_thread.start()
            self.worker = threading.Thread(target=self._infer, name='yolo-worker', daemon=True)
            self.worker.start()
            self.flow_worker = threading.Thread(target=self._track_flow,
                                                 name='fuel-flow-worker', daemon=True)
            self.flow_worker.start()
            if not self._play_source(self.pipeline):
                raise RuntimeError('Unable to start hardware video pipeline')
            if not self.ready.wait(20):
                raise RuntimeError(self.stats.get('error') or 'Video pipeline produced no frames within 20 seconds')
            if self.stats.get('error'):
                raise RuntimeError(self.stats['error'])
            self.stats['status'] = 'running'
        except Exception:
            self.stop()
            raise

    @staticmethod
    def _discard_pipeline(pipeline):
        if pipeline is None:
            return
        pipeline.set_state(Gst.State.NULL)
        pipeline.get_bus().remove_signal_watch()

    def reconnect_camera(self):
        """Recreate the live camera pipeline while retaining active WebRTC peers."""
        if not (self.source.isdigit() or self.source.startswith('/dev/video')):
            raise RuntimeError('Reconnect is only available for a live camera source')
        with self.pause_lock:
            if self.paused.is_set():
                raise RuntimeError('Resume the camera before reconnecting')
            with self.source_lock:
                if self.stopped.is_set() or self.pipeline is None:
                    raise RuntimeError('Video pipeline is not running')
                self.switching.set()
                try:
                    with self.inference_gate:
                        if not self.inference_gate.wait_for(lambda: not self.inference_busy, timeout=5):
                            raise RuntimeError('Inference did not stop before reconnecting the camera')
                    with self.flow_gate:
                        if not self.flow_gate.wait_for(lambda: not self.flow_busy, timeout=5):
                            raise RuntimeError('Optical flow did not stop before reconnecting the camera')
                    old_pipeline = self.pipeline
                    self.pipeline = None
                    self._discard_pipeline(old_pipeline)
                    self.slot.clear()
                    self.flow_slot.clear()
                    self._clear_right_frames()
                    self.ready.clear()
                    self.detections = None
                    self.fuel_tracker.reset()
                    self.stats.update(status='reconnecting', error=None)
                    self._build_description()
                    self.pipeline = self._new_pipeline()
                    self.new_epoch = True
                    if not self._play_source(self.pipeline):
                        raise RuntimeError('Unable to restart the camera pipeline')
                    if not self.ready.wait(20):
                        raise RuntimeError(self.stats.get('error') or 'Camera produced no video after reconnecting')
                    if self.stats.get('error'):
                        raise RuntimeError(self.stats['error'])
                    with self.lock:
                        self.previous = self.counters.copy()
                        self.rate_started = time.monotonic()
                        self.stats.update(status='running', error=None)
                except Exception as exc:
                    with self.lock:
                        self.stats.update(status='video_error', error=str(exc))
                    raise
                finally:
                    self.switching.clear()
        return self.snapshot()

    def switch_source(self, source, source_id, source_name):
        """Switch the local MP4 while keeping the detector and WebRTC peers alive."""
        if self.paused.is_set():
            raise RuntimeError('Resume video before switching sources')
        with self.pause_lock:
            if self.paused.is_set():
                raise RuntimeError('Resume video before switching sources')
            return self._switch_source_locked(source, source_id, source_name)

    def _switch_source_locked(self, source, source_id, source_name):
        path = Path(source).resolve(strict=True)
        with self.source_lock:
            if self.stopped.is_set() or self.pipeline is None:
                raise RuntimeError('Video pipeline is not running')
            if self.source_id == source_id and Path(self.source).resolve() == path:
                return self.snapshot()

            old_source, old_id, old_name = self.source, self.source_id, self.source_name
            self.switching.set()
            try:
                with self.inference_gate:
                    if not self.inference_gate.wait_for(lambda: not self.inference_busy, timeout=5):
                        raise RuntimeError('Inference did not stop before switching videos')
                with self.flow_gate:
                    if not self.flow_gate.wait_for(lambda: not self.flow_busy, timeout=5):
                        raise RuntimeError('Optical flow did not stop before switching videos')
                self._discard_pipeline(self.pipeline)
                self.pipeline = None
                self.slot.clear()
                self.flow_slot.clear()
                self._clear_right_frames()
                self.ready.clear()
                self.detections = None
                self.fuel_tracker.reset()

                def start_selected(selected_path, selected_id, selected_name):
                    self.source = str(selected_path)
                    self.source_id, self.source_name = selected_id, selected_name
                    self._build_description()
                    self.pipeline = self._new_pipeline()
                    self.new_epoch = True
                    self.stats.update(status='switching', error=None, input_resolution='unknown',
                                      source_fps=0., loops=0, source_id=selected_id,
                                      source_name=selected_name)
                    if not self._play_source(self.pipeline):
                        raise RuntimeError('Unable to start selected video')
                    if not self.ready.wait(20):
                        raise RuntimeError(self.stats.get('error') or 'Selected video produced no RTP within 20 seconds')
                    if self.stats.get('error'):
                        raise RuntimeError(self.stats['error'])

                try:
                    start_selected(path, source_id, source_name)
                except Exception as switch_error:
                    self._discard_pipeline(self.pipeline)
                    self.pipeline = None
                    self.slot.clear()
                    self._clear_right_frames()
                    self.ready.clear()
                    try:
                        start_selected(Path(old_source).resolve(), old_id, old_name)
                    except Exception as restore_error:
                        self.stats.update(status='video_error', error=str(restore_error))
                        raise RuntimeError('Video switch failed and the previous source could not be restored') from switch_error
                    self.stats.update(status='running')
                    raise RuntimeError(f'Video switch failed; restored {old_name or old_source}: {switch_error}') from switch_error

                with self.lock:
                    self.previous = self.counters.copy()
                    self.rate_started = time.monotonic()
                    self.stats.update(status='running', error=None)
                return self.snapshot()
            finally:
                self.switching.clear()

    def _on_capture(self, identity, buffer, owner=None):
        if owner is not None and owner is not self.pipeline:
            return
        with self.lock:
            if self.origin is None:
                self.origin = self.pipeline.get_base_time()
                self.rate_started = time.monotonic()
            if self.new_epoch:
                self.epoch = self.pipeline.get_base_time() - self.origin
                self.new_epoch = False
            pts_key = int(round((self.epoch + buffer.pts) / Gst.MSECOND))
            with self.capture_times_lock:
                self.capture_times[pts_key] = time.monotonic()
                if len(self.capture_times) > 2048:
                    oldest_keys = sorted(self.capture_times)[:-1536]
                    for key in oldest_keys:
                        self.capture_times.pop(key, None)
            self.counters['capture'] += 1
            if self.stats['input_resolution'] == 'unknown':
                caps = identity.get_static_pad('src').get_current_caps().get_structure(0)
                self.stats['input_resolution'] = f"{caps.get_value('width')}x{caps.get_value('height')}"
                ok, numerator, denominator = caps.get_fraction('framerate')
                if ok and denominator:
                    self.stats['source_fps'] = numerator / denominator
        return Gst.PadProbeReturn.OK

    def _on_inference_sample(self, sink):
        sample = sink.emit('pull-sample')
        if self.stopped.is_set():
            return Gst.FlowReturn.FLUSHING
        if not self.stream_enabled:
            self.ready.set()
        # Keep the video path alive for WebRTC, but do not copy frames into the
        # detector queue while nobody is watching. This avoids GPU inference
        # and the associated CPU/memory traffic when the dashboard is idle.
        with self.lock:
            if not self.peers and not self.calibrator.active and not self.inference_always:
                return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        info = GstVideo.VideoInfo()
        info.from_caps(sample.get_caps())
        success, mapped = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        copy_started = time.perf_counter()
        try:
            frame = np.ndarray((info.height, info.width, 3), dtype=np.uint8,
                               buffer=mapped.data, strides=(info.stride[0], 3, 1)).copy()
        finally:
            buffer.unmap(mapped)
        copy_ms = (time.perf_counter() - copy_started) * 1000
        with self.lock:
            pts = self.epoch + buffer.pts
        item = (pts, frame)
        self.slot.put(item)
        if not self.stereo_source:
            self.flow_slot.put(item)
        with self.lock:
            self.stats['frame_copy_ms'] = round(copy_ms, 2)
        return Gst.FlowReturn.OK

    def _on_flow_left_sample(self, sink):
        sample = sink.emit('pull-sample')
        if self.stopped.is_set():
            return Gst.FlowReturn.FLUSHING
        with self.lock:
            if not self.peers and not self.calibrator.active and not self.inference_always:
                return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        if buffer.pts == Gst.CLOCK_TIME_NONE:
            return Gst.FlowReturn.OK
        info = GstVideo.VideoInfo()
        info.from_caps(sample.get_caps())
        success, mapped = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        started = time.perf_counter()
        try:
            if info.finfo.name == 'GRAY8':
                frame = np.ndarray((info.height, info.width), dtype=np.uint8,
                                   buffer=mapped.data,
                                   strides=(info.stride[0], 1)).copy()
            else:
                frame = np.ndarray((info.height, info.width, 3), dtype=np.uint8,
                                   buffer=mapped.data,
                                   strides=(info.stride[0], 3, 1)).copy()
        finally:
            buffer.unmap(mapped)
        with self.lock:
            pts = self.epoch + buffer.pts
        self.flow_slot.put((pts, frame))
        with self.lock:
            self.stats['flow_copy_ms'] = round((time.perf_counter() - started) * 1000, 2)
        return Gst.FlowReturn.OK

    def _on_flow_right_sample(self, sink):
        sample = sink.emit('pull-sample')
        if self.stopped.is_set():
            return Gst.FlowReturn.FLUSHING
        with self.lock:
            if not self.peers and not self.calibrator.active and not self.inference_always:
                return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        if buffer.pts == Gst.CLOCK_TIME_NONE:
            return Gst.FlowReturn.OK
        info = GstVideo.VideoInfo()
        info.from_caps(sample.get_caps())
        success, mapped = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        try:
            gray = np.ndarray((info.height, info.width), dtype=np.uint8,
                              buffer=mapped.data,
                              strides=(info.stride[0], 1)).copy()
        finally:
            buffer.unmap(mapped)
        with self.lock:
            pts = self.epoch + buffer.pts
        with self.right_frame_condition:
            self.flow_right_frames[pts] = gray
            while len(self.flow_right_frames) > self.flow_right_cache_size:
                del self.flow_right_frames[min(self.flow_right_frames)]
            self.right_frame_condition.notify_all()
        return Gst.FlowReturn.OK

    def _on_stereo_flow_sample(self, sink):
        """Split both flow eyes from one sample so queue drops stay paired."""
        sample = sink.emit('pull-sample')
        if self.stopped.is_set():
            return Gst.FlowReturn.FLUSHING
        with self.lock:
            if not self.peers and not self.calibrator.active and not self.inference_always:
                return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        if buffer.pts == Gst.CLOCK_TIME_NONE:
            return Gst.FlowReturn.OK
        info = GstVideo.VideoInfo()
        info.from_caps(sample.get_caps())
        if info.finfo.name != 'GRAY8' or info.width != self.flow_view_width * 2:
            LOG.error('Unexpected atomic stereo flow caps: %s', sample.get_caps().to_string())
            return Gst.FlowReturn.ERROR
        success, mapped = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        started = time.perf_counter()
        try:
            side_by_side = np.ndarray((info.height, info.width), dtype=np.uint8,
                                      buffer=mapped.data,
                                      strides=(info.stride[0], 1)).copy()
        finally:
            buffer.unmap(mapped)
        left = side_by_side[:, :self.flow_view_width]
        right = side_by_side[:, self.flow_view_width:]
        with self.lock:
            pts = self.epoch + buffer.pts
        with self.right_frame_condition:
            self.flow_right_frames[pts] = right
            while len(self.flow_right_frames) > self.flow_right_cache_size:
                del self.flow_right_frames[min(self.flow_right_frames)]
            self.right_frame_condition.notify_all()
        self.flow_slot.put((pts, left))
        with self.lock:
            self.stats['flow_copy_ms'] = round((time.perf_counter() - started) * 1000, 2)
        return Gst.FlowReturn.OK

    def _on_stereo_right_sample(self, sink):
        sample = sink.emit('pull-sample')
        if self.stopped.is_set():
            return Gst.FlowReturn.FLUSHING
        with self.lock:
            if not self.peers and not self.calibrator.active and not self.inference_always:
                return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        if buffer.pts == Gst.CLOCK_TIME_NONE:
            return Gst.FlowReturn.OK
        info = GstVideo.VideoInfo()
        info.from_caps(sample.get_caps())
        success, mapped = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        started = time.perf_counter()
        try:
            gray = np.ndarray((info.height, info.width), dtype=np.uint8,
                              buffer=mapped.data,
                              strides=(info.stride[0], 1)).copy()
        finally:
            buffer.unmap(mapped)
        with self.lock:
            pts = self.epoch + buffer.pts
        with self.right_frame_condition:
            self.right_frames[pts] = gray
            while len(self.right_frames) > 4:
                del self.right_frames[min(self.right_frames)]
            self.right_frame_condition.notify_all()
        with self.lock:
            self.stats['stereo_copy_ms'] = round((time.perf_counter() - started) * 1000, 2)
        return Gst.FlowReturn.OK

    def _take_right_frame(self, pts, timeout=0):
        with self.right_frame_condition:
            frame = self.right_frames.pop(pts, None)
            if frame is None and timeout > 0:
                self.right_frame_condition.wait_for(
                    lambda: pts in self.right_frames or self.stopped.is_set(), timeout=timeout)
                frame = self.right_frames.pop(pts, None)
            for stale in [key for key in self.right_frames if key < pts]:
                del self.right_frames[stale]
            return frame

    def _clear_right_frames(self):
        with self.right_frame_condition:
            self.right_frames.clear()
            self.flow_right_frames.clear()
            self.last_depth_boxes = []
            self.right_frame_condition.notify_all()

    def _take_flow_right_frame(self, pts, timeout=0):
        with self.right_frame_condition:
            frame = self.flow_right_frames.pop(pts, None)
            if frame is None and timeout > 0:
                self.right_frame_condition.wait_for(
                    lambda: pts in self.flow_right_frames or self.stopped.is_set(),
                    timeout=timeout)
                frame = self.flow_right_frames.pop(pts, None)
            for key in [key for key in self.flow_right_frames if key < pts]:
                del self.flow_right_frames[key]
            return frame

    def _inference_enabled(self):
        """Return whether inference is active for a viewer or explicit headless operation."""
        with self.lock:
            return (bool(self.peers) or self.inference_always or
                    getattr(getattr(self, 'calibrator', None), 'active', False)) and not self.paused.is_set() and not self.switching.is_set()

    def _infer(self):
        while not self.stopped.is_set():
            item = self.slot.get()
            if item is None:
                return
            pts, frame = item
            inference_started = time.monotonic()
            if (self.inference_max_fps and self.last_inference_started and
                    inference_started - self.last_inference_started <
                    1.0 / self.inference_max_fps):
                with self.lock:
                    self.counters['inference_skipped'] += 1
                continue
            with self.inference_gate:
                if not self._inference_enabled():
                    continue
                self.inference_busy = True
            self.last_inference_started = inference_started
            started = time.monotonic()
            try:
                left = frame
                if not self.peers and not self.inference_always:
                    continue
                low_light = False
                detector_image = left
                if self.low_light_enabled:
                    mean_luma = float(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).mean())
                    low_light = mean_luma < self.low_light_luma_threshold
                    if low_light:
                        detector_image = cv2.LUT(left, self.low_light_lut)
                    self.low_light_last_mean_luma = mean_luma
                    with self.lock:
                        self.stats['low_light_mean_luma'] = round(mean_luma, 2)
                        if low_light:
                            self.low_light_frame_count += 1
                            self.stats['low_light_frames'] = self.low_light_frame_count
                result = self.detector.infer(detector_image)
                detections = result[0] if isinstance(result, tuple) else result
                if self.low_light_enabled:
                    candidate_count = sum(box.confidence >= self.normal_confidence
                                          for box in detections)
                    adaptive = (self.adaptive_confidence_trigger and
                                candidate_count >= self.adaptive_confidence_trigger)
                    cutoff = (self.adaptive_confidence_high if adaptive else
                              self.low_light_confidence if low_light else
                              self.normal_confidence)
                    before_filter = len(detections)
                    detections = [box for box in detections if box.confidence >= cutoff]
                    with self.lock:
                        if adaptive:
                            self.stats['adaptive_confidence_frames'] += 1
                        self.stats['adaptive_confidence_removed_boxes'] += before_filter - len(detections)
                elif (self.adaptive_confidence_trigger and
                      len(detections) >= self.adaptive_confidence_trigger):
                    before_filter = len(detections)
                    detections = [box for box in detections
                                  if box.confidence >= self.adaptive_confidence_high]
                    with self.lock:
                        self.stats['adaptive_confidence_frames'] += 1
                        self.stats['adaptive_confidence_removed_boxes'] += before_filter - len(detections)
                # Stereo is computed once per captured flow frame in
                # _track_flow. Recomputing it on detector anchors duplicates
                # the dense disparity pass and competes with the 60 Hz depth
                # worker.
                if self.stereo.size_gate_max_relative_error and not self.stereo_source:
                    detections = self.stereo.filter_fuel_size_inconsistent(detections)
                    with self.lock:
                        self.stats.update(
                            stereo_size_gate_rejected=self.stereo.last_size_gate_rejected,
                            stereo_size_gate_duplicate_rejected=self.stereo.last_size_gate_duplicate_rejected)
                if self.stereo_source:
                    scale_x = self.flow_view_width / max(1.0, float(left.shape[1]))
                    scale_y = self.flow_view_height / max(1.0, float(left.shape[0]))
                    for detection in detections:
                        detection.x1 = int(round(detection.x1 * scale_x))
                        detection.x2 = int(round(detection.x2 * scale_x))
                        detection.y1 = int(round(detection.y1 * scale_y))
                        detection.y2 = int(round(detection.y2 * scale_y))
                completed_at = time.monotonic()
                self.fuel_tracker.update(detections, completed_at, pts,
                                         return_snapshot=False)
                with self.lock:
                    self.counters['inference'] += 1
                    self.stats['inference_latency_ms'] = round((time.monotonic()-started)*1000, 2)
                    if getattr(self.detector, 'last_profile', None):
                        self.stats.update(self.detector.last_profile)
            except Exception as exc:
                LOG.exception('Inference failed')
                with self.lock:
                    self.stats.update(status='inference_error', error=str(exc))
                return
            finally:
                with self.inference_gate:
                    self.inference_busy = False
                    self.inference_gate.notify_all()

    @staticmethod
    def _depth_boxes_from_tracks(tracks, width, height):
        boxes = []
        for track in tracks:
            cx, cy, box_width, box_height = track['box']
            detection = Detection(
                max(0, int(round(cx - box_width * .5))),
                max(0, int(round(cy - box_height * .5))),
                min(width, int(round(cx + box_width * .5))),
                min(height, int(round(cy + box_height * .5))),
                int(track['detection'].class_id), float(track['confidence']))
            detection.distance_m = track['distance_m']
            detection.disparity_px = getattr(track['detection'], 'disparity_px', None)
            detection.track_id = track['track_id']
            boxes.append(detection)
        return boxes

    def _track_flow(self):
        while not self.stopped.is_set():
            item = self.flow_slot.get()
            if item is None:
                return
            if self.switching.is_set():
                continue
            with self.flow_gate:
                self.flow_busy = True
            pts, frame = item
            completed_at = time.monotonic()
            try:
                height, width = frame.shape[:2]
                stereo_started = time.perf_counter()
                right = (self._take_flow_right_frame(pts, timeout=self.stereo_pair_wait_s)
                         if self.stereo_source else None)
                if right is not None and self.calibrator.active:
                    try:
                        self.calibrator.observe(
                            cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR),
                            cv2.cvtColor(right, cv2.COLOR_GRAY2BGR))
                    except Exception:
                        LOG.exception('ChArUco preview failed')
                depth_future = None
                depth_boxes = []
                if (right is not None and self.stereo.dense and
                        self.last_depth_boxes):
                    # Dense map computation is independent of this frame's
                    # optical-flow positions. Sample the map at the prior
                    # frame's boxes while LK/Kalman advances current tracks.
                    depth_boxes = self.last_depth_boxes
                    depth_future = self.stereo_executor.submit(
                        self.stereo.estimate, frame, right, depth_boxes)
                tracker_started = time.perf_counter()
                tracks = self.fuel_tracker.advance(frame, completed_at, pts)
                tracker_elapsed_ms = (time.perf_counter() - tracker_started) * 1000
                box_build_started = time.perf_counter()
                current_depth_boxes = self._depth_boxes_from_tracks(
                    tracks, width, height)
                box_build_ms = (time.perf_counter() - box_build_started) * 1000
                if right is not None:
                    if depth_future is not None:
                        depth_future.result()
                    elif current_depth_boxes:
                        depth_boxes = current_depth_boxes
                        self.stereo.estimate(frame, right, depth_boxes)
                    if depth_boxes:
                        if self.stereo.size_gate_max_relative_error:
                            depth_boxes = self.stereo.filter_fuel_size_inconsistent(
                                depth_boxes, frame_width=width)
                            with self.lock:
                                self.stats.update(
                                    stereo_size_gate_rejected=self.stereo.last_size_gate_rejected,
                                    stereo_size_gate_duplicate_rejected=
                                    self.stereo.last_size_gate_duplicate_rejected)
                        measurements = [
                            (box.track_id, box.distance_m, box.disparity_px)
                            for box in depth_boxes if box.distance_m is not None]
                        range_update_time = time.monotonic()
                        self.fuel_tracker.update_ranges(
                            measurements, now=range_update_time, return_snapshot=False)
                        fresh_ranges = {track_id: (distance, disparity)
                                        for track_id, distance, disparity in measurements}
                        for track in tracks:
                            measured_range = fresh_ranges.get(track['track_id'])
                            if measured_range is not None:
                                track['distance_m'] = measured_range[0]
                                track['range_age_ms'] = max(
                                    0.0, (range_update_time - completed_at) * 1000.0)
                                track['detection'].disparity_px = measured_range[1]
                        with self.lock:
                            self.stats['stereo_flow_frames'] += 1
                            self.stats['stereo_flow_fresh_tracks'] = len(measurements)
                            self.stats['stereo_flow_estimate_ms'] = round(
                                (time.perf_counter() - stereo_started) * 1000, 2)
                            self.stats['stereo_flow_tracker_ms'] = round(
                                tracker_elapsed_ms, 2)
                            self.stats['stereo_flow_box_build_ms'] = round(
                                box_build_ms, 2)
                            self.stats['stereo_map_ms'] = round(
                                self.stereo.last_profile.get('total_ms', 0.0), 2)
                            self.stats['stereo_rectify_ms'] = round(
                                self.stereo.last_profile.get('rectify_and_roi_ms', 0.0), 2)
                            self.stats['stereo_match_ms'] = round(
                                self.stereo.last_profile.get('matcher_and_transfer_ms', 0.0), 2)
                            self.stats['stereo_extract_ms'] = round(
                                self.stereo.last_profile.get('range_extract_ms', 0.0), 2)
                            self.stats['stereo_active_num_disparities'] = (
                                self.stereo.last_num_disparities)
                            self.stats['stereo_min_disparity'] = (
                                self.stereo.last_min_disparity)
                elif self.stereo_source:
                    with self.lock:
                        self.stats['stereo_flow_missing_pairs'] += 1
                self.last_depth_boxes = current_depth_boxes
                boxes = []
                for track in tracks:
                    detection = track['detection']
                    cx, cy, box_width, box_height = track['box']
                    distance = track['distance_m']
                    measured = track['state'] == 'measured'
                    boxes.append(dict(
                        x=(cx - box_width * 0.5) / width,
                        y=(cy - box_height * 0.5) / height,
                        w=box_width / width,
                        h=box_height / height,
                        class_id=detection.class_id,
                        label=self.detector.class_names[detection.class_id],
                        confidence=round(track['confidence'], 3),
                        distance_m=round(distance, 2) if distance is not None else None,
                        disparity_px=(round(detection.disparity_px, 2)
                                      if measured and getattr(detection, 'disparity_px', None)
                                      else None),
                        track_id=track['track_id'],
                        track_state=track['state'],
                        track_age_ms=track['age_ms'],
                        visual_age_ms=track['visual_age_ms'],
                        range_age_ms=track['range_age_ms']))
                payload = dict(type='detections', rtp_timestamp=rtp_timestamp(pts),
                               boxes=[box for box in boxes
                                      if box['track_state'] == 'measured'],
                               tracks=boxes, completed_at=completed_at)
                with self.lock:
                    self.counters['flow'] += 1
                    self.stats.update(tracked_instances=len(boxes),
                                      predicted_instances=sum(box['track_state'] == 'predicted'
                                                              for box in boxes),
                                      flow_frame_replaced=self.flow_slot.replaced)
                    self.detections = payload
                    pipeline_error = self.stats.get('error')
                if self.nt_publisher is not None:
                    try:
                        nt_stats = self.nt_publisher.publish(
                            tracks, float(pts) / float(Gst.SECOND), width, height,
                            self.stereo.project_pixel_to_camera,
                            self.stereo.calibrated, self.stereo.approximate,
                            pipeline_error=pipeline_error)
                        with self.lock:
                            self.stats.update(nt_connected=nt_stats['connected'],
                                              nt_status=nt_stats['status'],
                                              nt_frame=nt_stats['frame'])
                    except Exception:
                        LOG.exception('FRC NetworkTables publish failed')
                        with self.lock:
                            self.stats.update(nt_status='error')
            except Exception:
                LOG.exception('FUEL optical-flow update failed')
            finally:
                with self.flow_gate:
                    self.flow_busy = False
                    self.flow_gate.notify_all()

    def _on_rtp_sample(self, sink):
        sample = sink.emit('pull-sample')
        if self.stopped.is_set():
            return Gst.FlowReturn.FLUSHING
        buffer = sample.get_buffer()
        if buffer.pts == Gst.CLOCK_TIME_NONE:
            return Gst.FlowReturn.OK
        packet = bytearray(buffer.extract_dup(0, buffer.get_size()))
        if len(packet) < 12:
            return Gst.FlowReturn.OK
        with self.lock:
            pts = self.epoch + buffer.pts
            packet[2:4] = (self.sequence & 0xffff).to_bytes(2, 'big')
            packet[4:8] = rtp_timestamp(pts).to_bytes(4, 'big')
            self.sequence += 1
            self.rtp_caps = sample.get_caps().copy()
            if packet[1] & 0x80:
                self.counters['encoded'] += 1
                self.stats['encode_latency_ms'] = round(max(0, self.clock.get_time()-self.origin-pts)/1e6, 2)
            peers = tuple(self.peers)
        output = Gst.Buffer.new_allocate(None, len(packet), None)
        output.fill(0, packet)
        output.pts = output.dts = pts
        output.duration = buffer.duration
        for peer in peers:
            peer.push(output)
        self.ready.set()
        return Gst.FlowReturn.OK

    def _on_message(self, bus, message, owner=None):
        if owner is not None and owner is not self.pipeline:
            return
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            LOG.error('Video pipeline: %s (%s)', error, debug)
            with self.lock:
                self.stats.update(status='video_error', error=str(error))
            self.ready.set()
        elif message.type == Gst.MessageType.EOS and self.loop_file and not self.stopped.is_set():
            with self.source_lock:
                if self.stopped.is_set():
                    return
                if owner is not None and owner is not self.pipeline:
                    return
                with self.lock:
                    self.stats['loops'] += 1
                    self.new_epoch = True
                # Recreate the pipeline so qtdemux re-emits and relinks its dynamic
                # video pad. Reusing the same demux after EOS leaves it unlinked.
                old = self.pipeline
                self._discard_pipeline(old)
                self.pipeline = None
                self.pipeline = self._new_pipeline()
                if not self._play_source(self.pipeline):
                    self.stats.update(status='video_error', error='Unable to loop source video')

    def _new_pipeline(self):
        pipeline = Gst.parse_launch(self.description)
        pipeline.use_clock(self.clock)
        pipeline.get_by_name('inference').connect('new-sample', self._on_inference_sample)
        flow_left_sink = pipeline.get_by_name('flow-left')
        if flow_left_sink is not None:
            flow_left_sink.connect('new-sample', self._on_flow_left_sample)
        flow_right_sink = pipeline.get_by_name('flow-right')
        if flow_right_sink is not None:
            flow_right_sink.connect('new-sample', self._on_flow_right_sample)
        stereo_flow_sink = pipeline.get_by_name('flow-stereo')
        if stereo_flow_sink is not None:
            stereo_flow_sink.connect('new-sample', self._on_stereo_flow_sample)
        rtp_sink = pipeline.get_by_name('rtp')
        if rtp_sink is not None:
            rtp_sink.connect('new-sample', self._on_rtp_sample)
        pipeline.get_by_name('pace').connect('handoff', self._on_capture, pipeline)
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self._on_message, pipeline)
        return pipeline

    def _play_source(self, pipeline):
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            return False
        if self.loop_file and self.effective_playback_rate != 1.0:
            self.playback_seek_attempts = 0
            GLib.timeout_add(250, self._apply_playback_rate, pipeline)
        return True

    def _apply_playback_rate(self, pipeline):
        if self.stopped.is_set() or pipeline is not self.pipeline:
            return GLib.SOURCE_REMOVE
        ok = pipeline.seek(
            self.effective_playback_rate, Gst.Format.TIME,
            Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE,
            Gst.SeekType.SET, 0, Gst.SeekType.NONE, -1)
        if ok:
            LOG.info('Set source video playback rate to %.4fx', self.effective_playback_rate)
            return GLib.SOURCE_REMOVE
        self.playback_seek_attempts += 1
        if self.playback_seek_attempts >= 20:
            LOG.error('Unable to set file playback rate to %.4fx', self.effective_playback_rate)
            with self.lock:
                self.stats.update(status='video_error', error='Unable to set video playback speed')
            self.ready.set()
            return GLib.SOURCE_REMOVE
        return GLib.SOURCE_CONTINUE

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.rate_started
            if elapsed >= 1:
                for name, count in self.counters.items():
                    self.stats[name + '_fps'] = round((count-self.previous[name])/elapsed, 2)
                self.previous = self.counters.copy()
                self.rate_started = now
            return dict(self.stats, capture_frames=self.counters['capture'],
                        inference_frames=self.counters['inference'], encoded_frames=self.counters['encoded'],
                        inference_replaced=self.slot.replaced, viewers=len(self.peers),
                        inference_active=self._inference_enabled(),
                        inference_queue_depth=int(self.slot.value is not None),
                        viewer_dropped_packets=sum(p.dropped for p in self.peers))

    def set_confidence(self, confidence):
        confidence = float(confidence)
        if not 0.1 <= confidence <= 0.95:
            raise ValueError('Confidence must be between 0.10 and 0.95')
        if self.mode != 'tensorrt' or self.detector is None:
            raise RuntimeError('Confidence tuning requires the TensorRT detector')
        with self.detector.lock:
            self.normal_confidence = confidence
            self.detector.confidence = (min(confidence, self.low_light_confidence)
                                        if self.low_light_enabled else confidence)
        with self.lock:
            self.stats['confidence'] = confidence
        return self.snapshot()

    def set_engine(self, engine_path):
        if self.mode != 'tensorrt' or self.detector is None:
            raise RuntimeError('Engine selection requires the TensorRT detector')
        engine_path = str(engine_path)
        if Path(engine_path).resolve() == Path(self.engine_path).resolve():
            return self.snapshot()
        with self.inference_gate:
            self.switching.set()
            try:
                if not self.inference_gate.wait_for(lambda: not self.inference_busy, timeout=30):
                    raise RuntimeError('Timed out waiting for the current inference to finish')
                old_path, old = self.engine_path, self.detector
                settings = dict(confidence=old.confidence, max_candidates=old.max_candidates,
                                iou=getattr(old, 'iou', .45),
                                reuse_preprocess=old.reuse_preprocess,
                                class_confidences=old.class_confidences)
                old.close()
                self.detector = None
                del old
                gc.collect()
                try:
                    replacement = create_detector(self.mode, engine_path, **settings)
                except Exception:
                    self.detector = create_detector(self.mode, old_path, **settings)
                    raise
                self.detector = replacement
                self.engine_path = engine_path
                with self.lock:
                    self.stats.update(detector=replacement.name,
                                      precision=replacement.precision,
                                      engine_input_resolution=replacement.input_resolution,
                                      confidence=replacement.confidence,
                                      nms_iou=getattr(replacement, 'iou', None),
                                      class_confidences=replacement.class_confidences,
                                      max_candidates=replacement.max_candidates or None,
                                      reuse_preprocess=replacement.reuse_preprocess,
                                      engine_path=engine_path)
            finally:
                self.switching.clear()
        return self.snapshot()

    def set_paused(self, paused):
        with self.pause_lock:
            return self._set_paused(paused)

    def _set_paused(self, paused):
        paused = bool(paused)
        if paused == self.paused.is_set():
            return self.snapshot()
        if paused:
            self.paused.set()
            result = self.pipeline.set_state(Gst.State.PAUSED)
            if result == Gst.StateChangeReturn.FAILURE:
                self.paused.clear()
                raise RuntimeError('Unable to pause video pipeline')
            self.slot.clear()
            with self.inference_gate:
                if not self.inference_gate.wait_for(lambda: not self.inference_busy, timeout=5):
                    raise RuntimeError('Inference did not stop within five seconds')
            with self.lock:
                self.stats['paused'] = True
                self.stats['status'] = 'paused'
                self.previous = self.counters.copy()
                self.rate_started = time.monotonic()
        else:
            result = self.pipeline.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError('Unable to resume video pipeline')
            self.slot.clear()
            self.paused.clear()
            with self.lock:
                self.stats['paused'] = False
                self.stats['status'] = 'running'
                self.previous = self.counters.copy()
                self.rate_started = time.monotonic()
        return self.snapshot()

    def stop(self):
        self.stopped.set()
        self.slot.close()
        self.flow_slot.close()
        self._clear_right_frames()
        with self.lock:
            peers = tuple(self.peers)
            self.peers.clear()
        for peer in peers:
            peer.close()
        with self.source_lock:
            pipeline, self.pipeline = self.pipeline, None
            if pipeline:
                pipeline.set_state(Gst.State.NULL)
                pipeline.get_bus().remove_signal_watch()
        if self.worker:
            self.worker.join(timeout=5)
        if self.flow_worker:
            self.flow_worker.join(timeout=5)
        self.stereo_executor.shutdown(wait=True)
        if self.nt_publisher:
            self.nt_publisher.close()
        if self.mainloop:
            self.mainloop.quit()
        if self.mainloop_thread:
            self.mainloop_thread.join(timeout=2)
