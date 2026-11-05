#!/usr/bin/env python3
"""Replay saved FUEL clips through FuelTracker with sparse detector anchors."""
import json
import itertools
import math
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scout.tracking import FuelTracker

DATA = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
CLIPS = {
    'earlier': ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr.mp4',
    'corrected-settings': ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4',
}
TARGETS = ((808.0, 589.0), (902.0, 591.0))
ANCHOR_STRIDES = (3, 7)
CONFIDENCE = 0.05
cv2.setNumThreads(1)


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]


def run_scene(scene, stride):
    frames = next(row['per_frame'] for row in DATA['results']
                  if row['scene'] == scene)
    capture = cv2.VideoCapture(str(CLIPS[scene]))
    tracker = FuelTracker()
    counts, errors, latencies = [], [[], []], []
    states, ages, confidence = {}, {}, []
    frame_index = 0
    started = time.perf_counter()
    while True:
        ok, side_by_side = capture.read()
        if not ok:
            break
        frame_started = time.perf_counter()
        left = side_by_side[:, :1600]
        pts = frame_index * 1500  # 60 Hz RTP-clock units, monotonic per frame
        tracks = tracker.advance(left, pts / 90000.0, pts)
        if frame_index % stride == 0:
            detections = []
            for row in frames[frame_index]['detections']:
                if row['confidence'] < CONFIDENCE:
                    continue
                detections.append(SimpleNamespace(
                    x1=row['x1'], y1=row['y1'], x2=row['x2'], y2=row['y2'],
                    class_id=0, confidence=row['confidence'], distance_m=None,
                    disparity_px=None))
            tracks = tracker.update(detections, pts / 90000.0, pts)
        count = len(tracks)
        counts.append(count)
        for track in tracks:
            states[track['state']] = states.get(track['state'], 0) + 1
            ages.setdefault(track['state'], []).append(track['age_ms'])
            confidence.append(track['confidence'])
        # Match the two returned tracks to fixed manually labeled centers.
        if len(tracks) >= 2:
            # Choose the two tracks that jointly best explain the fixed targets;
            # this prevents an extra spurious track from stealing a target slot.
            best = min(itertools.permutations(tracks, 2), key=lambda pair:
                       math.dist(pair[0]['box'][:2], TARGETS[0]) +
                       math.dist(pair[1]['box'][:2], TARGETS[1]))
            for target, track in enumerate(best):
                errors[target].append(math.dist(track['box'][:2], TARGETS[target]))
        latencies.append((time.perf_counter() - frame_started) * 1000)
        frame_index += 1
    capture.release()
    elapsed = time.perf_counter() - started
    return {
        'scene': scene,
        'anchor_stride_frames': stride,
        'nominal_detector_anchor_hz': 60 / stride,
        'frames': frame_index,
        'processing_elapsed_s': round(elapsed, 3),
        'processing_rate_hz': round(frame_index / elapsed, 2),
        'track_count_histogram': {str(count): counts.count(count)
                                  for count in sorted(set(counts))},
        'frames_with_two_tracks': counts.count(2),
        'two_track_frame_coverage': round(counts.count(2) / frame_index, 5),
        'frames_with_any_track': sum(count > 0 for count in counts),
        'track_state_instances': states,
        'age_ms_by_state': {state: {
            'median': round(statistics.median(values), 1),
            'p95': round(percentile(values, .95), 1)}
            for state, values in ages.items()},
        'confidence': {'median': round(statistics.median(confidence), 4)
                       if confidence else None,
                       'minimum': round(min(confidence), 4) if confidence else None},
        'center_error_px_by_target': [{
            'n': len(values),
            'median': round(statistics.median(values), 2) if values else None,
            'p95': round(percentile(values, .95), 2) if values else None,
        } for values in errors],
        'per_frame_tracker_latency_ms': {
            'median': round(statistics.median(latencies), 3),
            'p95': round(percentile(latencies, .95), 3),
            'p99': round(percentile(latencies, .99), 3),
        },
    }


output = {
    'method': ('Replay each saved stereo video frame through the production '
               'FuelTracker implementation on the left view; use cached 1280 '
               'TensorRT detections at fixed frame strides as detector anchors. '
               'This isolates tracking and LK cost; it does not execute TensorRT '
               'or stereo matching.'),
    'anchor_confidence': CONFIDENCE,
    'target_centers_px': TARGETS,
    'max_track_age_ms': 500,
    'flow_grace_ms': 120,
    'confidence_decay_ms': 250,
    'results': [run_scene(scene, stride)
                for scene in ('earlier', 'corrected-settings')
                for stride in ANCHOR_STRIDES],
    'limitations': ('The clips contain stationary targets and cannot validate '
                    'moving-ball accuracy, occlusion, crossings, identity '
                    'switches, stereo range refresh, or end-to-end 60 Hz. '
                    'Detector anchors are cached from offline per-frame '
                    'inference, sampled at regular intervals; live anchor timing '
                    'is irregular. Tracker latency excludes video decode, '
                    'TensorRT, stereo, capture, serialization, and NetworkTables.'),
}
out_path = ROOT / 'media/frc-fuel-hybrid-tracker-replay.json'
out_path.write_text(json.dumps(output, indent=2) + '\n')
for result in output['results']:
    print(result['scene'], 'stride', result['anchor_stride_frames'],
          'two-track', result['two_track_frame_coverage'],
          'processing Hz', result['processing_rate_hz'],
          'center p95', [item['p95'] for item in result['center_error_px_by_target']])
print('saved', out_path)
