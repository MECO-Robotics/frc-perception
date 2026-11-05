#!/usr/bin/env python3
"""Sweep confidence on actual 640 TensorRT detections across both saved clips."""
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
CLIPS = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())['results']
THRESHOLDS = (.01, .02, .03, .04, .05, .06, .075, .10, .15, .20)
TARGETS = ((808.0, 589.0), (902.0, 591.0))
MATCH_RADIUS = 60.0
ENGINE = 'models/rebuilt_fuel_yolo26n_640_fp16.engine'

os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '2.0'
from scout.detectors import TensorRTDetector


def summarize(values):
    if not values:
        return {'n': 0}
    ordered = sorted(values)
    center = statistics.median(ordered)
    return {'n': len(ordered), 'mean_ms': round(statistics.mean(ordered), 2),
            'median_ms': round(center, 2),
            'p95_ms': round(ordered[min(len(ordered) - 1,
                                        math.ceil(.95 * len(ordered)) - 1)], 2),
            'mad_ms': round(statistics.median(abs(x - center) for x in ordered), 2)}


def score(detections, threshold):
    matched = [False, False]
    extras = 0
    for box in detections:
        if box.confidence < threshold:
            continue
        cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        distances = [math.hypot(cx - x, cy - y) for x, y in TARGETS]
        target = min(range(2), key=distances.__getitem__)
        if distances[target] <= MATCH_RADIUS and not matched[target]:
            matched[target] = True
        else:
            extras += 1
    return sum(matched), extras


os.environ['SCOUT_MAX_CANDIDATES'] = '0'
detector = TensorRTDetector(str(ROOT / ENGINE), confidence=.01, iou=.10,
                            reuse_preprocess=True)
output = {
    'method': ('Run the actual FRC9550 640x640 TensorRT engine once on every '
               'frame of both saved stereo videos at candidate confidence 0.01 '
               'and NMS IoU 0.10. Rescore retained post-NMS boxes at higher '
               'confidence thresholds using fixed manually labeled target centers.'),
    'engine': ENGINE,
    'candidate_confidence': .01,
    'nms_iou': .10,
    'thresholds': list(THRESHOLDS),
    'limitations': ('Two stationary balls at one camera pose; fixed-center '
                    'matching does not measure arbitrary locations, physical '
                    'occlusion, stereo range, or unseen scene false positives.'),
    'results': [],
}

try:
    for clip in CLIPS:
        capture = cv2.VideoCapture(str(ROOT / clip['video']))
        ok, warm = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read {clip['video']}")
        detector.infer(warm[:, :1600])
        capture.release()

        capture = cv2.VideoCapture(str(ROOT / clip['video']))
        inference_ms = []
        totals = {threshold: {'hist': {0: 0, 1: 0, 2: 0},
                              'exact': 0, 'paired': 0, 'tp': 0,
                              'fp': 0, 'fn': 0}
                  for threshold in THRESHOLDS}
        frames = 0
        started = time.perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            detections, elapsed_ms = detector.infer(frame[:, :1600])
            inference_ms.append(elapsed_ms)
            for threshold in THRESHOLDS:
                hits, extras = score(detections, threshold)
                stats = totals[threshold]
                stats['hist'][hits] += 1
                stats['exact'] += hits == 2 and extras == 0
                stats['paired'] += hits == 2
                stats['tp'] += hits
                stats['fp'] += extras
                stats['fn'] += 2 - hits
        capture.release()
        summaries = []
        for threshold in THRESHOLDS:
            stats = totals[threshold]
            precision = stats['tp'] / (stats['tp'] + stats['fp']) if stats['tp'] + stats['fp'] else 0.0
            recall = stats['tp'] / (stats['tp'] + stats['fn']) if stats['tp'] + stats['fn'] else 0.0
            summaries.append({
                'confidence': threshold,
                'count_histogram': {str(k): v for k, v in stats['hist'].items()},
                'exact_two_frames': stats['exact'], 'paired_frames': stats['paired'],
                'tp': stats['tp'], 'fp': stats['fp'], 'fn': stats['fn'],
                'precision': round(precision, 5), 'recall': round(recall, 5),
                'f1': round(2 * precision * recall / (precision + recall), 5)
                if precision + recall else 0.0,
            })
        result = {
            'scene': clip['scene'], 'video': clip['video'], 'frames': frames,
            'decode_and_inference_seconds': round(time.perf_counter() - started, 2),
            'detector_latency_ms': summarize(inference_ms),
            'threshold_sweep': summaries,
        }
        output['results'].append(result)
        print(clip['scene'], 'frames', frames, 'latency', result['detector_latency_ms'])
        for row in summaries:
            print(row['confidence'], 'exact', row['exact_two_frames'],
                  'paired', row['paired_frames'], 'fp/fn', row['fp'], row['fn'],
                  'recall', row['recall'])
finally:
    detector.close()

destination = ROOT / 'media/frc-fuel-640-confidence-evaluation.json'
destination.write_text(json.dumps(output, indent=2) + '\n')
print('saved', destination)
