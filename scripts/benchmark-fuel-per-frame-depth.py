#!/usr/bin/env python3
"""Replay FUEL captures and measure fresh stereo depth on every tracked frame."""
import json
import math
import os
import statistics
import sys
import time
import argparse
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('SCOUT_STEREO_ENABLED', '1')
os.environ.setdefault('SCOUT_STEREO_RECTIFY', '1')
os.environ.setdefault('SCOUT_STEREO_CALIBRATION',
                       'media/calibration/stereo_calibration_live_order_candidate.yml')
os.environ.setdefault('SCOUT_STEREO_BASELINE_M', '0.0635')
os.environ.setdefault('SCOUT_STEREO_BLOCK_SIZE', '3')
os.environ.setdefault('SCOUT_STEREO_MATCH_SCALE', '0.5')
os.environ.setdefault('SCOUT_STEREO_VERTICAL_ROI_PAD', '24')
os.environ.setdefault('SCOUT_STEREO_HORIZONTAL_ROI', '1')
os.environ.setdefault('SCOUT_STEREO_ADAPTIVE_DISPARITY_WIDTH', '1')

import cv2
import numpy as np

sys.path.insert(0, str(ROOT))
from scout.detectors import Detection
from scout.stereo import StereoDepth
from scout.tracking import FuelTracker

cv2.setNumThreads(1)
DATA = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
CLIPS = {
    'earlier': ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr.mp4',
    'corrected-settings': ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4',
}
OUT = ROOT / 'media/frc-fuel-per-frame-depth-benchmark.json'
ANCHOR_STRIDE = 3
BACKENDS = tuple(value.strip() for value in
                 os.getenv('SCOUT_STEREO_BENCHMARK_BACKENDS', 'opencv').split(',')
                 if value.strip())


def summary(values):
    if not values:
        return None
    return {'median_ms': round(float(statistics.median(values)), 3),
            'p95_ms': round(float(np.percentile(values, 95)), 3),
            'mean_ms': round(float(statistics.mean(values)), 3),
            'max_ms': round(float(max(values)), 3)}


def run(scene, backend, max_frames=None):
    os.environ['SCOUT_STEREO_BACKEND'] = backend
    stereo = StereoDepth()
    capture = cv2.VideoCapture(str(CLIPS[scene]))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open {CLIPS[scene]}')
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
    frame_rows = next(row['per_frame'] for row in DATA['results']
                      if row['scene'] == scene)
    tracker = FuelTracker()
    latencies, valid_by_frame, depths, stereo_profiles = [], [], [], []
    depths_by_track = {}
    frame_count = frames_with_tracks = frames_all_ranges = 0
    wall_start = time.perf_counter()
    while True:
        if max_frames is not None and frame_count >= max_frames:
            break
        ok, sbs = capture.read()
        if not ok:
            break
        started = time.perf_counter()
        left, right = sbs[:, :1600], sbs[:, 1600:]
        pts = frame_count / 60.0
        tracks = tracker.advance(left, pts, frame_count)
        if frame_count % ANCHOR_STRIDE == 0 and frame_count < len(frame_rows):
            detections = [Detection(int(r['x1']), int(r['y1']), int(r['x2']),
                                    int(r['y2']), 0, float(r['confidence']))
                          for r in frame_rows[frame_count]['detections']
                          if float(r['confidence']) >= 0.05]
            tracks = tracker.update(detections, pts, frame_count)
        boxes = []
        for track in tracks:
            x, y, w, h = track['box']
            boxes.append(SimpleNamespace(
                x1=max(0, int(round(x - w / 2))),
                y1=max(0, int(round(y - h / 2))),
                x2=min(1600, int(round(x + w / 2))),
                y2=min(1200, int(round(y + h / 2))),
                class_id=0, confidence=float(track['confidence']),
                distance_m=track['distance_m'],
                disparity_px=getattr(track['detection'], 'disparity_px', None),
                track_id=track['track_id']))
        if boxes:
            frames_with_tracks += 1
            stereo.estimate(left, right, boxes)
            stereo_profiles.append(dict(stereo.last_profile))
            tracker.update_ranges(
                [(box.track_id, box.distance_m, box.disparity_px)
                 for box in boxes if box.distance_m is not None], now=pts)
            valid = [box for box in boxes if box.disparity_px is not None and
                     box.distance_m is not None and math.isfinite(box.distance_m)]
            valid_by_frame.append(len(valid))
            if len(valid) == len(boxes):
                frames_all_ranges += 1
            depths.extend(float(box.distance_m) for box in valid)
            for box in valid:
                depths_by_track.setdefault(str(box.track_id), []).append(
                    float(box.distance_m))
        else:
            valid_by_frame.append(0)
        latencies.append((time.perf_counter() - started) * 1000)
        frame_count += 1
    wall = time.perf_counter() - wall_start
    capture.release()
    return {
        'backend': backend, 'scene': scene, 'frames': frame_count,
        'video_fps': round(float(source_fps), 3),
        'anchor_stride_frames': ANCHOR_STRIDE,
        'depth_updates_attempted_hz': round(frames_with_tracks / wall, 2),
        'processing_hz_including_software_decode': round(frame_count / wall, 2),
        'frames_with_tracks': frames_with_tracks,
        'frames_with_fresh_depth_for_every_track': frames_all_ranges,
        'fresh_all_track_depth_fraction': round(frames_all_ranges / frame_count, 5)
            if frame_count else None,
        'valid_depths_per_frame_histogram': {
            str(k): valid_by_frame.count(k)
            for k in sorted(set(valid_by_frame))},
        'per_frame_processing_ms': summary(latencies),
        'stereo_stage_ms': {
            key: summary([row[key] for row in stereo_profiles if key in row])
            for key in sorted({key for row in stereo_profiles for key in row})},
        'stereo_configuration': {
            'match_scale': stereo.match_scale,
            'rectified': stereo.rectify,
            'horizontal_roi': stereo.horizontal_roi_enabled,
            'vertical_roi_pad': stereo.vertical_roi_pad,
            'num_disparities': stereo.num_disparities,
            'last_min_disparity': stereo.last_min_disparity,
            'last_num_disparities': stereo.last_num_disparities,
            'block_size': stereo.block_size,
        },
        'range_median_m': round(float(statistics.median(depths)), 4) if depths else None,
        'range_by_track_m': {
            track_id: {'n': len(values),
                       'median': round(float(statistics.median(values)), 4),
                       'mad': round(float(statistics.median(
                           [abs(value - statistics.median(values)) for value in values])), 4)}
            for track_id, values in sorted(depths_by_track.items())},
        'limitations': [
            'Depth values are fresh stereo measurements for flow-tracked 2D ROIs, not independent detector outputs.',
            'These saved scenes are stationary; no ground-truth 3D trajectories are available.',
            'OpenCV VideoCapture uses software decode here; stage latency separates only by overall frame timing.',
        ],
    }


parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=OUT)
parser.add_argument('--scene', choices=tuple(CLIPS), action='append')
parser.add_argument('--max-frames', type=int)
args = parser.parse_args()
scenes = args.scene or list(CLIPS)
results = [run(scene, backend, args.max_frames)
           for backend in BACKENDS for scene in scenes]
args.output.write_text(json.dumps({'method': 'Every decoded stereo frame gets a fresh disparity/range estimate for each current FuelTracker ROI. Detector anchors are replayed every third frame from cached outputs; intermediate image centers come from the production LK/Kalman tracker. Fresh depth is fed back into track state to test a narrowed disparity search on following frames.', 'results': results}, indent=2) + '\n')
print(args.output.read_text())
print('saved', args.output)
