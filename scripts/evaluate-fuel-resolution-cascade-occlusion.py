#!/usr/bin/env python3
"""Compare 640-first and 1280 fallback detection under synthetic occlusion."""
import json
import math
import os
import statistics
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
TARGETS = ((808.0, 589.0), (902.0, 591.0))
TIMES = tuple(1.0 + .4 * index for index in range(15))
OCCLUSIONS = (0.0, .25, .50, .75)
FALLBACK_THRESHOLDS = (.06, .10)
CONFIDENCE = .05
MATCH_RADIUS_PX = 60
FAST_ENGINE = 'models/rebuilt_fuel_yolo26n_640_fp16.engine'
FULL_ENGINE = 'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'

os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '2.0'
from scout.detectors import Detection, TensorRTDetector


def score(detections):
    hits = [None, None]
    extras = 0
    for box in sorted(detections, key=lambda item: item.confidence, reverse=True):
        center = ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)
        distance, index = min((math.dist(center, target), i)
                              for i, target in enumerate(TARGETS))
        if distance <= MATCH_RADIUS_PX and hits[index] is None:
            hits[index] = box
        else:
            extras += 1
    return sum(item is not None for item in hits), extras


def summarize(values):
    if not values:
        return {'n': 0}
    ordered = sorted(values)
    median = statistics.median(ordered)
    return {'n': len(ordered), 'mean_ms': round(statistics.mean(ordered), 2),
            'median_ms': round(median, 2),
            'p95_ms': round(ordered[min(len(ordered) - 1,
                                        math.ceil(.95 * len(ordered)) - 1)], 2)}


def json_box(box):
    return {'x1': box.x1, 'y1': box.y1, 'x2': box.x2, 'y2': box.y2,
            'confidence': round(box.confidence, 5)}


def detector_views(items):
    return [Detection(int(item['x1']), int(item['y1']), int(item['x2']),
                      int(item['y2']), 0, float(item['confidence']))
            for item in items]


fast_detector = TensorRTDetector(str(ROOT / FAST_ENGINE), confidence=CONFIDENCE)
full_detector = TensorRTDetector(str(ROOT / FULL_ENGINE), confidence=CONFIDENCE)
warmup = cv2.VideoCapture(str(ROOT / DATA['results'][0]['video']))
warmup.set(cv2.CAP_PROP_POS_MSEC, 1000)
warmup_ok, warmup_frame = warmup.read()
warmup.release()
if warmup_ok:
    fast_detector.infer(warmup_frame[:, :1600])
    full_detector.infer(warmup_frame[:, :1600])
output = {
    'method': ('At 15 timestamps per clip, black-mask the lower 0/25/50/75% '
               'of each clean manually matched ball box in the left image. Run '
               'the actual 640 and 1280x960 TensorRT engines once on each masked '
               'frame. Score cascade policies that use 640 unless output count '
               'is not two or minimum confidence is below 0.06/0.10; use the '
               '1280x960 output on fallback frames.'),
    'settings': {'confidence': CONFIDENCE, 'fallback_min_confidence': FALLBACK_THRESHOLDS,
                 'occlusion_fractions': OCCLUSIONS, 'sample_times_s': TIMES,
                 'mask_orientation': 'lower portion of each target box',
                 'match_radius_px': MATCH_RADIUS_PX,
                 'fast_engine': FAST_ENGINE, 'fallback_engine': FULL_ENGINE},
    'limitations': ('Synthetic black lower masks on 15 fixed-scene timestamps '
                    'per clip, not real foreground occlusion or parallax. Fixed '
                    'target windows score detections only. Timings exclude '
                    'stereo and live pipeline queues.'),
    'results': [],
}
try:
    for clip in DATA['results']:
        capture = cv2.VideoCapture(str(ROOT / clip['video']))
        fps = capture.get(cv2.CAP_PROP_FPS)
        fractions = []
        for fraction in OCCLUSIONS:
            rows = []
            for time_s in TIMES:
                frame_index = round(time_s * fps)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue
                left = frame[:, :1600].copy()
                target_boxes = [None, None]
                for box in sorted((item for item in clip['per_frame'][frame_index]['detections']
                                   if item['confidence'] >= .10),
                                  key=lambda item: item['confidence'], reverse=True):
                    center = ((box['x1'] + box['x2']) / 2,
                              (box['y1'] + box['y2']) / 2)
                    distance, index = min((math.dist(center, target), i)
                                          for i, target in enumerate(TARGETS))
                    if distance <= MATCH_RADIUS_PX and target_boxes[index] is None:
                        target_boxes[index] = box
                for box in target_boxes:
                    if box is None or not fraction:
                        continue
                    x1, y1, x2, y2 = map(int, (box['x1'], box['y1'], box['x2'], box['y2']))
                    mask_top = max(0, y2 - max(1, round((y2 - y1) * fraction)))
                    left[mask_top:y2, x1:x2] = 0
                fast_boxes, fast_ms = fast_detector.infer(left)
                full_boxes, full_ms = full_detector.infer(left)
                rows.append({
                    'time_s': time_s, 'frame': frame_index,
                    'fast_boxes': [json_box(box) for box in fast_boxes],
                    'full_boxes': [json_box(box) for box in full_boxes],
                    'fast_ms': fast_ms, 'full_ms': full_ms,
                })
            policy_rows = []
            for threshold in FALLBACK_THRESHOLDS:
                tp = fp = fn = exact_two = fallback_count = 0
                selected_times = []
                for row in rows:
                    fast = row['fast_boxes']
                    should_fallback = (len(fast) != 2 or
                                       min((item['confidence'] for item in fast), default=0)
                                       < threshold)
                    selected = row['full_boxes'] if should_fallback else fast
                    hits, extras = score(detector_views(selected))
                    tp += hits
                    fn += 2 - hits
                    fp += extras
                    exact_two += int(hits == 2 and extras == 0)
                    fallback_count += int(should_fallback)
                    selected_times.append(row['fast_ms'] + (row['full_ms'] if should_fallback else 0))
                precision = tp / (tp + fp) if tp + fp else 0
                recall = tp / (tp + fn) if tp + fn else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
                policy_rows.append({
                    'min_confidence_fallback': threshold,
                    'frames': len(rows), 'fallback_frames': fallback_count,
                    'fallback_fraction': round(fallback_count / len(rows), 5) if rows else None,
                    'exact_two_frames': exact_two, 'tp': tp, 'fp': fp, 'fn': fn,
                    'precision': round(precision, 5), 'recall': round(recall, 5),
                    'f1': round(f1, 5),
                    'cascade_detector_call_ms': summarize(selected_times),
                })
            direct_full_rows = [row['full_boxes'] for row in rows]
            full_tp = full_fp = full_fn = full_exact = 0
            for detections in direct_full_rows:
                hits, extras = score(detector_views(detections))
                full_tp += hits
                full_fn += 2 - hits
                full_fp += extras
                full_exact += int(hits == 2 and extras == 0)
            fractions.append({
                'occlusion_fraction': fraction, 'frames': len(rows),
                'full_1280x960_baseline': {'exact_two_frames': full_exact,
                                           'tp': full_tp, 'fp': full_fp,
                                           'fn': full_fn},
                'cascade_policies': policy_rows,
                'frames_detail': rows,
            })
        capture.release()
        output['results'].append({'scene': clip['scene'], 'video': clip['video'],
                                  'fractions': fractions})
        print(clip['scene'])
        for fraction in fractions:
            print('mask', fraction['occlusion_fraction'],
                  'full exact', fraction['full_1280x960_baseline']['exact_two_frames'],
                  'cascade', [(item['min_confidence_fallback'], item['exact_two_frames'],
                               item['fallback_fraction'], item['cascade_detector_call_ms']['mean_ms'])
                              for item in fraction['cascade_policies']])
finally:
    fast_detector.close()
    full_detector.close()

path = ROOT / 'media/frc-fuel-resolution-cascade-occlusion-evaluation.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print('saved', path)
