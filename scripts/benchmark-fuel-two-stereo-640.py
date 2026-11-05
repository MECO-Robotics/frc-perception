#!/usr/bin/env python3
"""Concurrent replay of two full-resolution stereo captures on one 640 TRT engine."""
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

CAPTURES = {
    'earlier': ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr.mp4',
    'corrected': ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4',
}
ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_640_fp16.engine'
OUTPUT = ROOT / 'media/frc-fuel-two-stereo-640-simultaneous-20261104.json'
FPS = 59.90016638935108

description_template = (
    'filesrc location="{video}" ! qtdemux name=demux '
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


def stats(values):
    if not values:
        return None
    return {'median': round(float(statistics.median(values)), 3),
            'p95': round(float(np.percentile(values, 95)), 3),
            'mean': round(float(statistics.mean(values)), 3)}


def make_pipeline(video):
    pipeline = Gst.parse_launch(description_template.format(video=video))
    return pipeline, pipeline.get_by_name('tracks'), pipeline.get_by_name('detections')


detector = TensorRTDetector(str(ENGINE), confidence=.10, iou=.15)
# Warm the engine before timing both streams.
warm_capture = cv2.VideoCapture(str(next(iter(CAPTURES.values()))))
ok, warm_sbs = warm_capture.read()
warm_capture.release()
if not ok:
    raise RuntimeError('Could not read a warm-up frame')
detector.infer(warm_sbs[:, :1600])

states = {}
for name, video in CAPTURES.items():
    pipeline, track_sink, detector_sink = make_pipeline(video)
    states[name] = {
        'video': video, 'pipeline': pipeline, 'track_sink': track_sink,
        'detector_sink': detector_sink, 'tracker': FuelTracker(),
        'frame_count': 0, 'current_frame': -1, 'finished': False,
        'flow_ms': [], 'pull_ms': [], 'copy_ms': [], 'track_counts': [],
        'detector_rows': [], 'profiles': [], 'errors': [],
    }

for state in states.values():
    state['pipeline'].set_state(Gst.State.PLAYING)

engine_lock = threading.Lock()
timing_start = time.perf_counter()


def track_worker(name, state):
    while True:
        t0 = time.perf_counter()
        sample = state['track_sink'].emit('try-pull-sample', 5 * Gst.SECOND)
        state['pull_ms'].append((time.perf_counter() - t0) * 1000)
        if sample is None:
            bus_error = state['pipeline'].get_bus().pop_filtered(Gst.MessageType.ERROR)
            if bus_error:
                err, debug = bus_error.parse_error()
                state['errors'].append(f'tracker branch: {err}; {debug}')
            break
        buffer = sample.get_buffer()
        pts_ns = buffer.pts
        frame = round(pts_ns * FPS / Gst.SECOND) if pts_ns != Gst.CLOCK_TIME_NONE else state['frame_count']
        info = GstVideo.VideoInfo()
        if not info.from_caps(sample.get_caps()):
            state['errors'].append('Unable to parse tracker frame caps')
            break
        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            state['errors'].append('Could not map tracker frame')
            break
        try:
            view = np.ndarray((info.height, info.width, 3), dtype=np.uint8,
                              buffer=mapped.data, strides=(info.stride[0], 3, 1))
            t0 = time.perf_counter()
            low = view.copy()
            state['copy_ms'].append((time.perf_counter() - t0) * 1000)
        finally:
            buffer.unmap(mapped)
        sample = None
        tic = time.perf_counter()
        tracks = state['tracker'].advance(low, pts_ns / Gst.SECOND, frame)
        state['flow_ms'].append((time.perf_counter() - tic) * 1000)
        state['track_counts'].append(len(tracks))
        state['frame_count'] += 1
        state['current_frame'] = frame
    state['finished'] = True


def detector_worker(name, state):
    while True:
        sample = state['detector_sink'].emit('try-pull-sample', 200 * Gst.MSECOND)
        if sample is None:
            if state['finished']:
                return
            continue
        buffer = sample.get_buffer()
        pts_ns = buffer.pts
        frame = round(pts_ns * FPS / Gst.SECOND) if pts_ns != Gst.CLOCK_TIME_NONE else -1
        info = GstVideo.VideoInfo()
        if not info.from_caps(sample.get_caps()):
            state['errors'].append('Unable to parse detector frame caps')
            continue
        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            state['errors'].append('Could not map detector frame')
            continue
        try:
            view = np.ndarray((info.height, info.width, 3), dtype=np.uint8,
                              buffer=mapped.data, strides=(info.stride[0], 3, 1))
            image = view.copy()
        finally:
            buffer.unmap(mapped)
        sample = None
        started = time.perf_counter()
        try:
            with engine_lock:
                boxes, infer_ms = detector.infer(image)
                profile = dict(detector.last_profile)
            selected = []
            for box in boxes:
                scaled = Detection(round(box.x1 * .2), round(box.y1 * .2),
                                   round(box.x2 * .2), round(box.y2 * .2),
                                   box.class_id, box.confidence)
                cx, cy = (scaled.x1 + scaled.x2) * .5, (scaled.y1 + scaled.y2) * .5
                if box.confidence >= .10 and 130 <= cx <= 210 and 95 <= cy <= 145:
                    selected.append(scaled)
            selected = sorted(selected, key=lambda b: b.confidence, reverse=True)[:2]
            update_started = time.perf_counter()
            state['tracker'].update(selected, pts_ns / Gst.SECOND, frame)
            update_ms = (time.perf_counter() - update_started) * 1000
            lag = max(0, state['current_frame'] - frame)
            state['detector_rows'].append({
                'frame': frame, 'call_ms': (time.perf_counter() - started) * 1000,
                'infer_ms': infer_ms, 'count': len(selected), 'update_ms': update_ms,
                'lag_frames': lag,
            })
            state['profiles'].append(profile)
        except Exception as exc:
            state['errors'].append(repr(exc))


threads = []
for name, state in states.items():
    for target in (track_worker, detector_worker):
        thread = threading.Thread(target=target, args=(name, state),
                                  name=f'{target.__name__}-{name}')
        thread.start()
        threads.append(thread)
for thread in threads:
    thread.join()
wall = time.perf_counter() - timing_start
for state in states.values():
    state['pipeline'].set_state(Gst.State.NULL)
detector.close()

per_stream = {}
total_inferences = 0
for name, state in states.items():
    rows = state['detector_rows']
    total_inferences += len(rows)
    field_names = ('preprocess_ms', 'h2d_ms', 'tensorrt_ms', 'd2h_ms',
                   'postprocess_ms', 'detector_total_ms')
    profile = {field: stats([float(p[field]) for p in state['profiles'] if field in p])
               for field in field_names}
    per_stream[name] = {
        'video': str(state['video'].relative_to(ROOT)),
        'tracker_frames': state['frame_count'],
        'tracker_hz': round(state['frame_count'] / wall, 2),
        'flow_advance_ms': stats(state['flow_ms']),
        'tracker_gstreamer_pull_ms': stats(state['pull_ms'][:-1]),
        'tracker_frame_copy_ms': stats(state['copy_ms']),
        'detector_inferences': len(rows),
        'detector_hz': round(len(rows) / wall, 2),
        'anchor_detection_count_histogram': {str(k): sum(r['count'] == k for r in rows) for k in (0, 1, 2)},
        'detector_source_frame_lag': stats([r['lag_frames'] for r in rows]),
        'detector_call_ms': stats([r['call_ms'] for r in rows]),
        'kalman_anchor_update_ms': stats([r['update_ms'] for r in rows]),
        'detector_stages_ms': profile,
        'errors': state['errors'],
    }

out = {
    'method': 'Two independent 3200x1200@~60fps side-by-side stereo captures replayed simultaneously through two hardware-decoded GStreamer pipelines. Each has an NVVIDCONV 320x240 tracker branch and native 1600x1200 left-eye detector branch. Both detector workers share one 640x640 FP16 TensorRT engine/context, protected by a lock. Each stream has an independent LK/Kalman tracker.',
    'engine': str(ENGINE.relative_to(ROOT)), 'confidence': .10,
    'wall_seconds': round(wall, 3), 'combined_detector_inferences': total_inferences,
    'combined_detector_hz': round(total_inferences / wall, 2),
    'combined_tracker_frames': sum(s['frame_count'] for s in states.values()),
    'combined_tracker_hz': round(sum(s['frame_count'] for s in states.values()) / wall, 2),
    'per_stream': per_stream,
    'limitations': ['Two saved static captures, not two physical USB cameras. GStreamer decodes and paces both files concurrently.',
                    'No stereo depth/SGBM, NetworkTables, stream encoder, or live USB capture contention.',
                    'Detector recall is not estimated here; this run measures concurrent throughput.'],
}
OUTPUT.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
print('saved', OUTPUT)
