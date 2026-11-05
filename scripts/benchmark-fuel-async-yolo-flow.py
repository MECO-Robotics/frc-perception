#!/usr/bin/env python3
"""Replay FUEL video with latest-frame YOLO worker concurrent with LK/Kalman."""
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '2.0'
from scout.detectors import TensorRTDetector
from scout.tracking import FuelTracker

VIDEO = ROOT / 'media/frc-fuel-live-heavy-occlusion-20261103.mp4'
ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
OUTPUT = ROOT / 'media/frc-fuel-async-yolo-flow-20261104.json'
FPS = 59.966685

condition = threading.Condition()
latest = None
finished = False
detector_rows = []
detector_profiles = []
detector_errors = []
tracker = FuelTracker()


def yolo_worker(detector):
    global latest, finished
    seen = -1
    while True:
        with condition:
            condition.wait_for(lambda: finished or (latest is not None and latest[0] != seen))
            if latest is None or (finished and latest[0] == seen):
                return
            frame_id, pts, image = latest
            seen = frame_id
        started = time.perf_counter()
        try:
            boxes, total_ms = detector.infer(image)
            selected = []
            for box in boxes:
                cx, cy = (box.x1 + box.x2) * .5, (box.y1 + box.y2) * .5
                if box.confidence >= .05 and 130 <= cx <= 210 and 95 <= cy <= 145:
                    selected.append(box)
            selected = sorted(selected, key=lambda b: b.confidence, reverse=True)[:2]
            update_started = time.perf_counter()
            tracker.update(selected, pts / FPS, frame_id)
            update_ms = (time.perf_counter() - update_started) * 1000
            detector_rows.append((frame_id, time.perf_counter() - started, total_ms,
                                  len(selected), update_ms))
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


cap = cv2.VideoCapture(str(VIDEO))
detector = TensorRTDetector(str(ENGINE), confidence=.05, iou=.15)
ok, first = cap.read()
if not ok:
    raise RuntimeError(f'Unable to read {VIDEO}')
warm = cv2.resize(first[:, :1600], (320, 240), interpolation=cv2.INTER_AREA)
detector.infer(warm)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
worker = threading.Thread(target=yolo_worker, args=(detector,), name='async-yolo')
worker.start()
flow_ms, decode_ms, resize_ms, slot_copy_ms, output_counts = [], [], [], [], []
frame_id = 0
started = time.perf_counter()
try:
    while True:
        stage_started = time.perf_counter()
        ok, side_by_side = cap.read()
        decode_ms.append((time.perf_counter() - stage_started) * 1000)
        if not ok:
            break
        left = side_by_side[:, :1600]
        stage_started = time.perf_counter()
        low = cv2.resize(left, (320, 240), interpolation=cv2.INTER_AREA)
        resize_ms.append((time.perf_counter() - stage_started) * 1000)
        pts = frame_id
        tic = time.perf_counter()
        tracks = tracker.advance(low, pts / FPS, pts)
        flow_ms.append((time.perf_counter() - tic) * 1000)
        output_counts.append(len(tracks))
        stage_started = time.perf_counter()
        with condition:
            latest = (frame_id, pts, low.copy())
            condition.notify()
        slot_copy_ms.append((time.perf_counter() - stage_started) * 1000)
        frame_id += 1
finally:
    with condition:
        finished = True
        condition.notify_all()
    worker.join()
    detector.close()
    cap.release()
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
for position, (detected_frame, _, _, _, _) in enumerate(detector_rows):
    # Frames consumed during the asynchronous inference call.
    if position + 1 < len(detector_rows):
        lag_frames.append(max(0, detector_rows[position + 1][0] - detected_frame))
out = {
    'video': str(VIDEO.relative_to(ROOT)), 'frames': frames, 'source_fps': FPS,
    'engine': str(ENGINE.relative_to(ROOT)), 'mode': 'latest-frame asynchronous YOLO worker concurrent with per-frame LK/Kalman',
    'wall_seconds': round(wall, 3), 'replay_speed_x': round(wall / (frames / FPS), 3),
    'tracker': {'frames_advanced': frames, 'reported_position_hz_wall': round(tracker_hz, 2),
                'flow_advance_ms': summarize(flow_ms),
                'decode_ms': summarize(decode_ms[:-1]),
                'left_resize_ms': summarize(resize_ms),
                'latest_frame_copy_publish_ms': summarize(slot_copy_ms),
                'frames_with_tracks': sum(n > 0 for n in output_counts),
                'frames_with_two_tracks': sum(n >= 2 for n in output_counts)},
    'yolo': {'completed_inferences': len(detector_rows), 'inference_hz_wall': round(detector_hz, 2),
             'source_anchor_hz': round(len(detector_rows) / (frames / FPS), 2),
             'anchor_detection_count_histogram': {str(n): sum(row[3] == n for row in detector_rows) for n in (0, 1, 2)},
             'source_frame_lag': summarize(lag_frames),
             'kalman_anchor_tracker_update_ms': summarize([r[4] for r in detector_rows]),
             'stage_latency_ms': profile_summary},
    'errors': detector_errors,
    'notes': ['YOLO consumes the newest available 320x240 left frame; pending frames are overwritten like the live bounded queue.',
              'Replay is not paced at 60 fps; Hz fields report wall-clock processing rates.',
              'Stereo, NetworkTables, GStreamer capture, and streaming encoder are excluded.']
}
OUTPUT.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
print('saved', OUTPUT)
