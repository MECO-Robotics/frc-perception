#!/usr/bin/env python3
"""Sweep NMS IoU for the 640 FRC9550 engine on both saved videos."""
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
IOUS = (.05, .08, .10, .15, .20)
CONFIDENCE = .05
CASCADE_FALLBACK_MIN = .06
TARGETS = ((808.0, 589.0), (902.0, 591.0))
MATCH_RADIUS = 60.0
ENGINE = 'models/rebuilt_fuel_yolo26n_640_fp16.engine'

os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '2.0'
os.environ['SCOUT_MAX_CANDIDATES'] = '0'
from scout.detectors import TensorRTDetector


def med(values):
    return round(statistics.median(values), 2) if values else None


def score(boxes):
    matched = [False, False]
    extras = 0
    for box in sorted(boxes, key=lambda item: item.confidence, reverse=True):
        cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        distances = [math.hypot(cx - x, cy - y) for x, y in TARGETS]
        target = min(range(2), key=distances.__getitem__)
        if distances[target] <= MATCH_RADIUS and not matched[target]:
            matched[target] = True
        else:
            extras += 1
    return sum(matched), extras


detector = TensorRTDetector(str(ROOT / ENGINE), confidence=CONFIDENCE,
                            iou=.10, reuse_preprocess=True)
output = {
    'method': ('Run the same 640 TensorRT engine once at each NMS IoU per frame, '
               'holding confidence at 0.05. Evaluate exact two-target outputs and '
               'the confidence-cascade fallback trigger (count != 2 or minimum '
               'confidence < 0.06).'),
    'engine': ENGINE,
    'confidence': CONFIDENCE,
    'nms_ious': list(IOUS),
    'fallback_min_confidence': CASCADE_FALLBACK_MIN,
    'limitations': ('Fixed two-ball target windows and stationary saved clips only. '
                    'Fallback frequency is a trigger estimate; 1280 outputs are '
                    'not rerun for each IoU, so these metrics characterize the '
                    'fast stage rather than full cascade quality.'),
    'results': [],
}

try:
    for clip in CLIPS:
        captures = {iou: cv2.VideoCapture(str(ROOT / clip['video'])) for iou in IOUS}
        for iou, capture in captures.items():
            ok, warm = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read {clip['video']}")
            detector.iou = iou
            detector.infer(warm[:, :1600])
            capture.release()
            captures[iou] = cv2.VideoCapture(str(ROOT / clip['video']))
        records = {iou: {'hist': {0: 0, 1: 0, 2: 0}, 'exact': 0,
                         'tp': 0, 'fp': 0, 'fn': 0, 'fallback': 0,
                         'fallback_count': 0, 'fallback_weak': 0,
                         'latencies': []} for iou in IOUS}
        frame_count = 0
        while True:
            frames = {}
            for iou, capture in captures.items():
                ok, frame = capture.read()
                frames[iou] = frame if ok else None
            if all(frame is None for frame in frames.values()):
                break
            if any(frame is None for frame in frames.values()):
                raise RuntimeError('Clip decoders became misaligned')
            frame_count += 1
            for iou in IOUS:
                detector.iou = iou
                boxes, elapsed_ms = detector.infer(frames[iou][:, :1600])
                record = records[iou]
                record['latencies'].append(elapsed_ms)
                hits, extras = score(boxes)
                record['hist'][hits] += 1
                record['exact'] += hits == 2 and extras == 0
                record['tp'] += hits
                record['fp'] += extras
                record['fn'] += 2 - hits
                weak = min((box.confidence for box in boxes), default=0) < CASCADE_FALLBACK_MIN
                if len(boxes) != 2 or weak:
                    record['fallback'] += 1
                    record['fallback_count'] += len(boxes) != 2
                    record['fallback_weak'] += len(boxes) == 2 and weak
        for capture in captures.values():
            capture.release()
        summaries = []
        for iou in IOUS:
            record = records[iou]
            precision = record['tp'] / (record['tp'] + record['fp']) if record['tp'] + record['fp'] else 0
            recall = record['tp'] / (record['tp'] + record['fn']) if record['tp'] + record['fn'] else 0
            summaries.append({
                'iou': iou, 'count_histogram': {str(k): v for k, v in record['hist'].items()},
                'exact_two_frames': record['exact'], 'tp': record['tp'],
                'fp': record['fp'], 'fn': record['fn'],
                'precision': round(precision, 5), 'recall': round(recall, 5),
                'f1': round(2 * precision * recall / (precision + recall), 5)
                if precision + recall else 0,
                'estimated_fallback_frames': record['fallback'],
                'fallback_due_count': record['fallback_count'],
                'fallback_due_weak_confidence': record['fallback_weak'],
                'detector_latency_ms_median': med(record['latencies']),
                'detector_latency_ms_mean': round(statistics.mean(record['latencies']), 2),
            })
        output['results'].append({'scene': clip['scene'], 'video': clip['video'],
                                  'frames': frame_count, 'iou_sweep': summaries})
        print(clip['scene'], 'frames', frame_count)
        for row in summaries:
            print(row)
finally:
    detector.close()

destination = ROOT / 'media/frc-fuel-640-nms-evaluation.json'
destination.write_text(json.dumps(output, indent=2) + '\n')
print('saved', destination)
