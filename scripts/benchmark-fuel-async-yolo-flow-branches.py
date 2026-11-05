#!/usr/bin/env python3
"""Replay native stereo video with GPU-scaled tracker and full-res async YOLO branches."""
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '2.0'
from scout.gst_runtime import Gst, GstVideo
import cv2
import numpy as np
from scout.detectors import Detection, TensorRTDetector
from scout.tracking import FuelTracker

VIDEO = ROOT / 'media/frc-fuel-live-heavy-occlusion-20261103.mp4'
ENGINE = Path(os.environ.get('FUEL_BENCHMARK_ENGINE',
                            'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'))
if not ENGINE.is_absolute():
    ENGINE = ROOT / ENGINE
OUTPUT = ROOT / os.environ.get('FUEL_BENCHMARK_OUTPUT',
                              'media/frc-fuel-async-yolo-flow-branches-20261104.json')
FPS = 59.966685

condition = threading.Condition()
latest = None
finished = False
current_frame_id = -1
detector_rows = []
detector_profiles = []
detector_errors = []
tracker = FuelTracker()


def yolo_worker(detector):
    global finished
    while True:
        sample = detector_sink.emit('try-pull-sample', 200 * Gst.MSECOND)
        if sample is None:
            if finished:
                return
            continue
        buffer = sample.get_buffer()
        pts_ns = buffer.pts
        frame_id = round(pts_ns * FPS / Gst.SECOND) if pts_ns != Gst.CLOCK_TIME_NONE else -1
        info = GstVideo.VideoInfo()
        if not info.from_caps(sample.get_caps()):
            detector_errors.append('Unable to parse native detector frame format')
            continue
        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            detector_errors.append('Unable to map native detector frame')
            continue
        try:
            source = np.ndarray((info.height, info.width, 3), dtype=np.uint8,
                                buffer=mapped.data, strides=(info.stride[0], 3, 1))
            image = source.copy()
        finally:
            buffer.unmap(mapped)
        sample = None
        started = time.perf_counter()
        try:
            boxes, total_ms = detector.infer(image)
            selected = []
            for box in boxes:
                scaled = Detection(round(box.x1 * .2), round(box.y1 * .2),
                                   round(box.x2 * .2), round(box.y2 * .2),
                                   box.class_id, box.confidence)
                cx, cy = (scaled.x1 + scaled.x2) * .5, (scaled.y1 + scaled.y2) * .5
                if box.confidence >= .05 and 130 <= cx <= 210 and 95 <= cy <= 145:
                    selected.append(scaled)
            selected = sorted(selected, key=lambda b: b.confidence, reverse=True)[:2]
            update_started = time.perf_counter()
            tracker.update(selected, pts_ns / Gst.SECOND, frame_id)
            update_ms = (time.perf_counter() - update_started) * 1000
            with condition:
                lag = max(0, current_frame_id - frame_id)
            detector_rows.append((frame_id, time.perf_counter() - started, total_ms,
                                  len(selected), update_ms, lag))
            profile = dict(detector.last_profile)
            profile['frame_id'] = frame_id
            detector_profiles.append(profile)
        except Exception as exc:
            detector_errors.append(repr(exc))


def summarize(values):
    if not values:
        return None
    return {'median': round(float(statistics.median(values)), 3),
            'p95': round(float(np.percentile(values, 95)), 3),
            'mean': round(float(statistics.mean(values)), 3)}


description = (
    f'filesrc location="{VIDEO}" ! qtdemux name=demux '
    'demux.video_0 ! queue ! h264parse ! nvv4l2decoder ! '
    'identity sync=true single-segment=true ! tee name=views '
    'views. ! queue max-size-buffers=4 max-size-bytes=0 max-size-time=0 ! '
    'nvvidconv ! video/x-raw,format=BGRx,width=640,height=240 ! videocrop right=320 ! '
    'videoconvert ! video/x-raw,format=BGR ! '
    'appsink name=tracks sync=false max-buffers=4 drop=false enable-last-sample=false '
    'views. ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! '
    'nvvidconv ! video/x-raw,format=BGRx,width=3200,height=1200 ! videocrop right=1600 ! '
    'videoconvert ! video/x-raw,format=BGR ! '
    'appsink name=detections sync=false max-buffers=1 drop=true enable-last-sample=false'
)
pipeline = Gst.parse_launch(description)
track_sink = pipeline.get_by_name('tracks')
detector_sink = pipeline.get_by_name('detections')
detector = TensorRTDetector(str(ENGINE), confidence=.05, iou=.15)
pipeline.set_state(Gst.State.PLAYING)
sample = detector_sink.emit('try-pull-sample', 5 * Gst.SECOND)
if sample is None:
    raise RuntimeError(f'GStreamer could not decode {VIDEO}')
info = GstVideo.VideoInfo()
if not info.from_caps(sample.get_caps()):
    raise RuntimeError('Unable to parse GStreamer frame format')
ok, mapped = sample.get_buffer().map(Gst.MapFlags.READ)
if not ok:
    raise RuntimeError('Unable to map warm-up frame')
try:
    warm = np.ndarray((info.height, info.width, 3), dtype=np.uint8,
                      buffer=mapped.data, strides=(info.stride[0], 3, 1)).copy()
finally:
    sample.get_buffer().unmap(mapped)
detector.infer(warm)
sample = None
pipeline.set_state(Gst.State.NULL)
pipeline = Gst.parse_launch(description)
track_sink = pipeline.get_by_name('tracks')
detector_sink = pipeline.get_by_name('detections')
pipeline.set_state(Gst.State.PLAYING)
worker = threading.Thread(target=yolo_worker, args=(detector,), name='async-yolo')
worker.start()
flow_ms, gst_pull_ms, frame_copy_ms, output_counts = [], [], [], []
frame_id = 0
started = time.perf_counter()
try:
    while True:
        stage_started = time.perf_counter()
        sample = track_sink.emit('try-pull-sample', 5 * Gst.SECOND)
        gst_pull_ms.append((time.perf_counter() - stage_started) * 1000)
        if sample is None:
            message = pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
            if message:
                err, debug = message.parse_error()
                raise RuntimeError(f'GStreamer replay error: {err}; {debug}')
            break
        info = GstVideo.VideoInfo()
        if not info.from_caps(sample.get_caps()):
            raise RuntimeError('Unable to parse GStreamer frame format')
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError('Unable to map GStreamer frame')
        try:
            low = np.ndarray((info.height, info.width, 3), dtype=np.uint8,
                             buffer=mapped.data, strides=(info.stride[0], 3, 1))
            stage_started = time.perf_counter()
            low = low.copy()
            frame_copy_ms.append((time.perf_counter() - stage_started) * 1000)
        finally:
            buffer.unmap(mapped)
        sample = None
        pts = frame_id
        tic = time.perf_counter()
        tracks = tracker.advance(low, pts / FPS, pts)
        flow_ms.append((time.perf_counter() - tic) * 1000)
        output_counts.append(len(tracks))
        with condition:
            current_frame_id = frame_id
        frame_id += 1
finally:
    with condition:
        finished = True
        condition.notify_all()
    worker.join()
    detector.close()
    pipeline.set_state(Gst.State.NULL)
wall = time.perf_counter() - started

frames = frame_id
tracker_hz = frames / wall
detector_hz = len(detector_rows) / wall
profiles = detector_profiles
profile_fields = ('preprocess_ms', 'h2d_ms', 'tensorrt_ms', 'd2h_ms', 'postprocess_ms', 'detector_total_ms')
profile_summary = {field: summarize([float(p[field]) for p in profiles if field in p])
                   for field in profile_fields}
profile_summary['async_detector_call_ms'] = summarize([r[1] * 1000 for r in detector_rows])
lag_frames = []
for position, (detected_frame, _, _, _, _, _) in enumerate(detector_rows):
    # Frames consumed during the asynchronous inference call.
    if position + 1 < len(detector_rows):
        lag_frames.append(max(0, detector_rows[position + 1][0] - detected_frame))
out = {
    'video': str(VIDEO.relative_to(ROOT)), 'frames': frames, 'source_fps': FPS,
    'engine': str(ENGINE.relative_to(ROOT)), 'mode': 'latest-frame asynchronous YOLO worker concurrent with per-frame LK/Kalman',
    'wall_seconds': round(wall, 3), 'replay_speed_x': round(wall / (frames / FPS), 3),
    'tracker': {'frames_advanced': frames, 'reported_position_hz_wall': round(tracker_hz, 2),
                'flow_advance_ms': summarize(flow_ms),
                'gstreamer_pull_ms': summarize(gst_pull_ms[:-1]),
                'gstreamer_to_numpy_copy_ms': summarize(frame_copy_ms),
                'frames_with_tracks': sum(n > 0 for n in output_counts),
                'frames_with_two_tracks': sum(n >= 2 for n in output_counts)},
    'yolo': {'completed_inferences': len(detector_rows), 'inference_hz_wall': round(detector_hz, 2),
             'source_anchor_hz': round(len(detector_rows) / (frames / FPS), 2),
             'anchor_detection_count_histogram': {str(n): sum(row[3] == n for row in detector_rows) for n in (0, 1, 2)},
             'source_frame_lag': summarize([row[5] for row in detector_rows]),
             'kalman_anchor_tracker_update_ms': summarize([r[4] for r in detector_rows]),
             'stage_latency_ms': profile_summary},
    'errors': detector_errors,
    'notes': ['GStreamer tees the native decoded 3200x1200 stereo frame. NVVIDCONV scales the tracker branch to 640x240 before cropping its 320x240 left view; the YOLO branch retains the native 1600x1200 left eye.',
              'YOLO pulls the newest frame from a one-frame drop-old detector appsink; flow/Kalman consumes every frame on its separate branch.',
              'Replay is paced to the source clip rate; Hz fields report measured wall-clock rates.',
              'Stereo, NetworkTables, GStreamer capture, and streaming encoder are excluded.']
}
OUTPUT.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
print('saved', OUTPUT)
