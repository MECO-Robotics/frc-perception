#!/usr/bin/env python3
"""Run fresh TensorRT inference and compare fixed/adaptive confidence cutoffs."""
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '2.0'
from scout.detectors import TensorRTDetector

VIDEOS = (
    ('earlier', ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr.mp4'),
    ('corrected-settings', ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'),
)
ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
TARGETS = ((808.0, 589.0), (902.0, 591.0))
MATCH_RADIUS = 60
LOW, HIGH, LUMA_CUTOFF = 0.05, 0.075, 18.0


def score(detections, threshold):
    chosen = [None, None]
    extras = 0
    for box in sorted((b for b in detections if b.confidence >= threshold),
                      key=lambda b: b.confidence, reverse=True):
        center = ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)
        distance, index = min((math.dist(center, target), i)
                              for i, target in enumerate(TARGETS))
        if distance <= MATCH_RADIUS and chosen[index] is None:
            chosen[index] = box
        else:
            extras += 1
    return sum(box is not None for box in chosen), extras


def summarize(rows):
    tp, fp = sum(x['tp'] for x in rows), sum(x['fp'] for x in rows)
    fn = 2 * len(rows) - tp
    return {'frames': len(rows), 'tp': tp, 'fp': fp, 'fn': fn,
            'exact_two_frames': sum(x['tp'] == 2 and x['fp'] == 0 for x in rows),
            'precision': round(tp / (tp + fp), 6) if tp + fp else None,
            'recall': round(tp / (tp + fn), 6) if tp + fn else None}


def timing(values):
    if not values:
        return None
    ordered = sorted(values)
    return {'median_ms': round(statistics.median(ordered), 2),
            'p95_ms': round(ordered[math.ceil(.95 * len(ordered)) - 1], 2),
            'mean_ms': round(statistics.mean(ordered), 2)}


detector = TensorRTDetector(str(ENGINE), confidence=LOW)
warmup_capture = cv2.VideoCapture(str(VIDEOS[0][1]))
warmup_ok, warmup_frame = warmup_capture.read()
warmup_capture.release()
if warmup_ok:
    detector.infer(warmup_frame[:, :1600])
results = []
try:
    for scene, video in VIDEOS:
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS)
        rows, inference_ms, luma_ms = [], [], []
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            left = frame[:, :1600]
            start = time.perf_counter()
            detections, measured_ms = detector.infer(left)
            inference_ms.append(measured_ms)
            gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            luma = float(gray.mean())
            luma_ms.append((time.perf_counter() - start) * 1000 - measured_ms)
            fixed_low = score(detections, LOW)
            fixed_high = score(detections, HIGH)
            selected = fixed_low if luma <= LUMA_CUTOFF else fixed_high
            rows.append({'frame': index, 'time_s': index / fps, 'luma': luma,
                         'low': {'tp': fixed_low[0], 'fp': fixed_low[1]},
                         'high': {'tp': fixed_high[0], 'fp': fixed_high[1]},
                         'adaptive': {'tp': selected[0], 'fp': selected[1]}})
            index += 1
        capture.release()
        summary = {'scene': scene, 'video': str(video.relative_to(ROOT)),
                   'frames': len(rows),
                   'fixed_0.05': summarize([row['low'] for row in rows]),
                   'fixed_0.075': summarize([row['high'] for row in rows]),
                   'adaptive_luma_18': summarize([row['adaptive'] for row in rows]),
                   'inference_latency_ms': timing(inference_ms),
                   'extra_luma_and_scoring_ms': timing(luma_ms),
                   'first_0.5s': {
                       name: summarize([row[key] for row in rows if row['time_s'] < .5])
                       for name, key in (('fixed_0.05', 'low'), ('fixed_0.075', 'high'),
                                         ('adaptive_luma_18', 'adaptive'))},
                   'frames_detail': rows}
        results.append(summary)
        print(scene, {key: summary[key] for key in (
            'frames', 'fixed_0.05', 'fixed_0.075', 'adaptive_luma_18',
            'inference_latency_ms', 'extra_luma_and_scoring_ms', 'first_0.5s')})
finally:
    detector.close()

output = {
    'method': ('Fresh TensorRT inference on each saved frame with the FRC 1280x960 '
               'engine, configured at 0.05, then rescore the same detections at '
               '0.05, 0.075, and the luma-18 adaptive rule. This verifies the '
               'offline confidence adaptation against a new detector pass.'),
    'settings': {'engine': str(ENGINE.relative_to(ROOT)), 'base_confidence': LOW,
                 'high_confidence': HIGH, 'luma_cutoff': LUMA_CUTOFF,
                 'match_radius_px': MATCH_RADIUS},
    'limitations': ('Same fixed scenes/labels and no motion, external lighting '
                    'variation, stereo processing, or measured range truth. '
                    'Luma is computed after inference for timing; the detector '
                    'threshold itself is post-filtered from the 0.05 output.'),
    'results': results,
}
path = ROOT / 'media/frc-fuel-luma-adaptive-live-inference-evaluation.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print('saved', path)
