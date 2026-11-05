#!/usr/bin/env python3
"""Run the 640-first conditional 1280x960 fallback cascade on both FUEL videos."""
import argparse
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
DATA = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fast-engine', default='models/rebuilt_fuel_yolo26n_640_fp16.engine')
parser.add_argument('--fallback-engine', default='models/rebuilt_fuel_yolo26n_1280x960_fp16.engine')
parser.add_argument('--confidence', type=float, default=.05)
parser.add_argument('--min-confidence-fallback', type=float, default=.06)
parser.add_argument('--output', default='media/frc-fuel-resolution-cascade-runtime.json')
args = parser.parse_args()

os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '2.0'
from scout.detectors import TensorRTDetector

TARGETS = ((808, 589), (902, 591))
MATCH_RADIUS_PX = 60


def summarize(values):
    if not values:
        return {'n': 0}
    values = sorted(values)
    median = statistics.median(values)
    return {'n': len(values), 'mean_ms': round(statistics.mean(values), 2),
            'median_ms': round(median, 2),
            'p95_ms': round(values[min(len(values) - 1, math.ceil(.95 * len(values)) - 1)], 2),
            'mad_ms': round(statistics.median(abs(value - median) for value in values), 2)}


def match(detections):
    targets = [None, None]
    extras = 0
    for box in sorted(detections, key=lambda item: item.confidence, reverse=True):
        cx = (box.x1 + box.x2) / 2
        cy = (box.y1 + box.y2) / 2
        distance, target_index = min(
            (math.hypot(cx - x, cy - y), index)
            for index, (x, y) in enumerate(TARGETS))
        if distance <= MATCH_RADIUS_PX and targets[target_index] is None:
            targets[target_index] = box
        else:
            extras += 1
    return targets, extras


def box_json(box):
    return {'x1': box.x1, 'y1': box.y1, 'x2': box.x2, 'y2': box.y2,
            'confidence': round(box.confidence, 5)}


fast = TensorRTDetector(str(ROOT / args.fast_engine), confidence=args.confidence)
fallback = TensorRTDetector(str(ROOT / args.fallback_engine), confidence=args.confidence)
results = []
try:
    for clip in DATA['results']:
        capture = cv2.VideoCapture(str(ROOT / clip['video']))
        # Warm up each engine once with the opening left frame; do not include it.
        ok, warm_frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read {clip['video']}")
        fast.infer(warm_frame[:, :1600])
        fallback.infer(warm_frame[:, :1600])
        capture.release()

        capture = cv2.VideoCapture(str(ROOT / clip['video']))
        frame_rows = []
        fast_times, fallback_times, cascade_times = [], [], []
        tp = fp = fn = pair_frames = exact_two = fallback_count = 0
        count_hist = {}
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            left = frame[:, :1600]
            detections_fast, fast_ms = fast.infer(left)
            fast_times.append(fast_ms)
            should_fallback = (
                len(detections_fast) != 2 or
                min((d.confidence for d in detections_fast), default=0.0)
                < args.min_confidence_fallback)
            fallback_ms = 0.0
            if should_fallback:
                selected, fallback_ms = fallback.infer(left)
                fallback_times.append(fallback_ms)
                fallback_count += 1
                selected_source = '1280x960'
            else:
                selected = detections_fast
                selected_source = '640'
            elapsed = fast_ms + fallback_ms
            cascade_times.append(elapsed)
            targets, extras = match(selected)
            hits = sum(item is not None for item in targets)
            tp += hits
            fn += 2 - hits
            fp += extras
            pair_frames += int(hits == 2)
            exact_two += int(hits == 2 and extras == 0)
            count = len(selected)
            count_hist[str(count)] = count_hist.get(str(count), 0) + 1
            frame_rows.append({
                'frame': frame_index - 1,
                'fallback': should_fallback,
                'fallback_reason': ('count' if len(detections_fast) != 2 else
                                    'low_min_confidence' if should_fallback else None),
                'selected_source': selected_source,
                'fast_ms': round(fast_ms, 3),
                'fallback_ms': round(fallback_ms, 3),
                'selected_count': count,
                'target_hits': hits,
                'off_target_boxes': extras,
                'fast_boxes': [box_json(box) for box in detections_fast],
                'selected_boxes': [box_json(box) for box in selected],
            })
        capture.release()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results.append({
            'scene': clip['scene'], 'video': clip['video'],
            'frames': frame_index, 'confidence': args.confidence,
            'min_confidence_fallback': args.min_confidence_fallback,
            'fallback_frames': fallback_count,
            'fallback_fraction': round(fallback_count / frame_index, 5) if frame_index else None,
            'count_histogram': count_hist,
            'target_pairs': pair_frames, 'exact_two_outputs': exact_two,
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(precision, 5), 'recall': round(recall, 5),
            'f1': round(f1, 5),
            'fast_detector_ms': summarize(fast_times),
            'fallback_detector_ms': summarize(fallback_times),
            'cascade_detector_ms': summarize(cascade_times),
            'frames_detail': frame_rows,
        })
finally:
    fast.close()
    fallback.close()

output = {
    'method': ('Run the actual TensorRT 640 engine on every decoded frame. If '
               'it returns a candidate count other than two or the minimum '
               'confidence is below the configured fallback threshold, run the '
               'actual 1280x960 engine on that same frame and replace the 640 '
               'result. Both videos are decoded in full; engine warm-up calls '
               'are excluded from timing.'),
    'configuration': {'fast_engine': args.fast_engine,
                      'fallback_engine': args.fallback_engine,
                      'confidence': args.confidence,
                      'min_confidence_fallback': args.min_confidence_fallback,
                      'target_centers_px': TARGETS,
                      'match_radius_px': MATCH_RADIUS_PX},
    'limitations': ('Static two-ball views at one pose and range. Target windows '
                    'are evaluation-only labels, although the runtime fallback '
                    'does not use them. The script measures detector inference '
                    'only; it excludes the live capture, preprocessing between '
                    'pipeline stages, stereo, network, and fallback integration '
                    'overhead. It does not validate metric distance.'),
    'results': results,
}
path = ROOT / args.output
path.write_text(json.dumps(output, indent=2) + '\n')
for row in results:
    print(row['scene'], 'pairs/exact', row['target_pairs'], row['exact_two_outputs'],
          'fallback', row['fallback_fraction'], 'fast/fallback/cascade ms',
          row['fast_detector_ms'], row['fallback_detector_ms'], row['cascade_detector_ms'],
          'P/R/F1', row['precision'], row['recall'], row['f1'])
print('saved', path)
