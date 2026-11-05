#!/usr/bin/env python3
"""Compare box-anchor+LK with Hough-refined anchor+LK on saved FUEL clips.

Uses cached detections; no detector inference, stereo, or live service. FPS includes
video decoding, while stage latency is measured after each frame has been decoded.
"""
import itertools
import json
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
TARGETS = ((808.0, 589.0), (902.0, 591.0))
STRIDE = 3
CONFIDENCE = 0.05
cv2.setNumThreads(1)


def percentile(values, q):
    if not values:
        return None
    return float(sorted(values)[min(len(values) - 1, math.ceil(q * len(values)) - 1)])


def summarize(values):
    return {'n': len(values), 'median': round(statistics.median(values), 3) if values else None,
            'p95': round(percentile(values, .95), 3) if values else None}


def det_object(row, center=None):
    x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
    if center is not None:
        w, h = x2 - x1, y2 - y1
        x1, y1, x2, y2 = center[0] - w / 2, center[1] - h / 2, center[0] + w / 2, center[1] + h / 2
    return SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2, class_id=0,
                           confidence=row['confidence'], distance_m=None, disparity_px=None)


def hough_center(gray, row):
    bx, by = (row['x1'] + row['x2']) / 2, (row['y1'] + row['y2']) / 2
    short = min(row['x2'] - row['x1'], row['y2'] - row['y1'])
    x0, y0 = max(0, int(round(bx - short * .8))), max(0, int(round(by - short * .8)))
    x1, y1 = min(gray.shape[1], int(round(bx + short * .8))), min(gray.shape[0], int(round(by + short * .8)))
    crop = cv2.GaussianBlur(gray[y0:y1, x0:x1], (9, 9), 2)
    rmin, rmax = max(8, int(short * .35)), max(9, int(short * .65))
    circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(10, int(short * .5)),
                               param1=90, param2=25, minRadius=rmin, maxRadius=rmax) if crop.size else None
    candidates = []
    if circles is not None:
        for x, y, _r in circles[0]:
            center = (float(x + x0), float(y + y0))
            offset = math.dist(center, (bx, by))
            if offset <= short * .30:
                candidates.append((offset, center))
    return min(candidates, key=lambda c: c[0])[1] if candidates else None


def nearest_targets(tracks):
    # The two tracks jointly best matching the fixed labels; extra tracks stay extras.
    if len(tracks) < 2:
        return [None, None]
    best = min(itertools.permutations(tracks, 2), key=lambda pair:
               math.dist(pair[0]['box'][:2], TARGETS[0]) + math.dist(pair[1]['box'][:2], TARGETS[1]))
    return list(best)


def run_scene(clip):
    scene, video = clip['scene'], ROOT / clip['video']
    per_frame = clip['per_frame']
    cap = cv2.VideoCapture(str(video))
    trackers = {'box_anchor_lk': FuelTracker(), 'hough_anchor_lk': FuelTracker()}
    rec = {name: {'counts': [], 'errors': [[], []], 'target_ids': [[], []], 'states': {},
                  'stage_ms': [], 'anchor_hough_valid': 0, 'anchor_hough_total': 0,
                  'anchor_detections': 0, 'stage_cpu_ms': []} for name in trackers}
    frame_idx = 0
    wall_start = time.perf_counter()
    decode_ms, perframe_wall_ms = [], []
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            break
        decode_ms.append((t1 - t0) * 1000)
        left = frame[:, :1600]
        gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        anchor = frame_idx % STRIDE == 0
        rows = [r for r in per_frame[frame_idx]['detections'] if r['confidence'] >= CONFIDENCE] if anchor else []
        pts = frame_idx * 1500
        for name, tracker in trackers.items():
            stage_start = time.perf_counter()
            cpu_start = time.process_time()
            tracks = tracker.advance(left, pts / 90000.0, pts)
            if anchor:
                detections = []
                for row in rows:
                    center = None
                    if name == 'hough_anchor_lk':
                        rec[name]['anchor_hough_total'] += 1
                        center = hough_center(gray, row)
                        rec[name]['anchor_hough_valid'] += int(center is not None)
                    detections.append(det_object(row, center))
                rec[name]['anchor_detections'] += len(detections)
                tracks = tracker.update(detections, pts / 90000.0, pts)
            stage_ms = (time.perf_counter() - stage_start) * 1000
            stage_cpu_ms = (time.process_time() - cpu_start) * 1000
            r = rec[name]
            r['stage_ms'].append(stage_ms)
            r['stage_cpu_ms'].append(stage_cpu_ms)
            r['counts'].append(len(tracks))
            targets = nearest_targets(tracks)
            for i, track in enumerate(targets):
                if track is None:
                    r['target_ids'][i].append(None)
                    continue
                r['errors'][i].append(math.dist(track['box'][:2], TARGETS[i]))
                r['target_ids'][i].append(track['track_id'])
                state = track['state']
                r['states'][state] = r['states'].get(state, 0) + 1
        perframe_wall_ms.append((time.perf_counter() - t0) * 1000)
        frame_idx += 1
    cap.release()
    elapsed = time.perf_counter() - wall_start
    results = []
    for name, r in rec.items():
        id_switches = []
        for seq in r['target_ids']:
            valid = [v for v in seq if v is not None]
            id_switches.append(sum(a != b for a, b in zip(valid, valid[1:])))
        counts = r['counts']
        results.append({
            'architecture': name, 'reported_frames': frame_idx,
            'two_track_coverage': round(sum(c >= 2 for c in counts) / max(1, frame_idx), 5),
            'exactly_two_track_coverage': round(counts.count(2) / max(1, frame_idx), 5),
            'one_or_more_track_coverage': round(sum(c >= 1 for c in counts) / max(1, frame_idx), 5),
            'track_count_histogram': {str(c): counts.count(c) for c in sorted(set(counts))},
            'center_error_px_by_fixed_target': [summarize(v) for v in r['errors']],
            'id_switches_by_target': id_switches,
            'track_state_instances': r['states'],
            'anchor_frames': math.ceil(frame_idx / STRIDE), 'cached_anchor_detections': r['anchor_detections'],
            'hough_valid_fraction_at_anchors': (round(r['anchor_hough_valid'] / r['anchor_hough_total'], 5)
                                                 if r['anchor_hough_total'] else None),
            'processing_ms_per_frame_excluding_decode': summarize(r['stage_ms']),
            'process_cpu_ms_per_frame_excluding_decode': summarize(r['stage_cpu_ms']),
            'estimated_one_core_cpu_fraction': round(statistics.mean(r['stage_cpu_ms']) /
                                                    max(0.001, statistics.mean(r['stage_ms'])), 3),
            'whole_pipeline_ms_per_frame_including_decode': summarize(perframe_wall_ms),
            'whole_pipeline_throughput_including_decode_fps': round(frame_idx / elapsed, 2),
            'track_stage_only_rate_fps': round(1000 / statistics.mean(r['stage_ms']), 2) if r['stage_ms'] else None,
        })
    return {'scene': scene, 'frames': frame_idx, 'decoded_fps': round(frame_idx / elapsed, 2),
            'decode_ms_per_frame': summarize(decode_ms), 'methods': results}


output = {
    'method': 'Replay cached detector outputs every 3 frames; compare production FuelTracker with box anchors against the same tracker using Hough-refined centers at anchors. Both use LK flow on every frame between anchors. Hough param2=25 and fallback to box center when no acceptable circle is found. No TensorRT, stereo, NT, or live camera.',
    'target_centers_px': TARGETS, 'anchor_stride_frames': STRIDE, 'nominal_anchor_rate_hz': 20,
    'anchor_confidence_threshold': CONFIDENCE,
    'timing_definition': 'Whole pipeline FPS includes OpenCV video decode and both compared tracker passes, but excludes cached detector inference, stereo matching, camera capture, serialization, and NetworkTables. Per-frame stage latency includes grayscale conversion, LK tracker calls, anchor Hough when used, and cached-anchor updates, but excludes decode.',
    'limitations': 'Both clips show two nearly stationary balls. Fixed manual image centers are the position reference; no motion, crossing, meaningful occlusion, or physical range truth is present. ID switch counts therefore only describe label association in this static replay. Sequential dual-method timing is useful for relative CPU load but is not an isolated production FPS estimate.',
    'results': [run_scene(c) for c in DATA['results']],
}
path = ROOT / 'media/frc-fuel-hough-anchor3-lk-comparison.json'
path.write_text(json.dumps(output, indent=2) + '\n')
for scene in output['results']:
    print(scene['scene'], 'decode fps', scene['decoded_fps'])
    for method in scene['methods']:
        print(method['architecture'], 'coverage', method['two_track_coverage'],
              'center p95', [v['p95'] for v in method['center_error_px_by_fixed_target']],
              'stage p95 ms', method['processing_ms_per_frame_excluding_decode']['p95'],
              'FPS incl decode', method['whole_pipeline_throughput_including_decode_fps'],
              'Hough use', method['hough_valid_fraction_at_anchors'])
print('saved', path)
