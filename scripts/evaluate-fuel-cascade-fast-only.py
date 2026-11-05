#!/usr/bin/env python3
"""Score saved 640-only cascade candidates against the selected cascade outputs."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'media/frc-fuel-resolution-cascade-runtime.json'
TARGETS = ((808.0, 589.0), (902.0, 591.0))
CONFIDENCE = .05
MATCH_RADIUS = 60.0


def score(rows, field):
    hist = {0: 0, 1: 0, 2: 0}
    exact = tp = fp = fn = 0
    for row in rows:
        matched = [False, False]
        extras = 0
        for box in row[field]:
            if box['confidence'] < CONFIDENCE:
                continue
            cx = (box['x1'] + box['x2']) / 2
            cy = (box['y1'] + box['y2']) / 2
            distances = [math.hypot(cx - x, cy - y) for x, y in TARGETS]
            target = min(range(2), key=distances.__getitem__)
            if distances[target] <= MATCH_RADIUS and not matched[target]:
                matched[target] = True
            else:
                extras += 1
        count = sum(matched)
        hist[count] += 1
        exact += count == 2 and extras == 0
        tp += count
        fn += 2 - count
        fp += extras
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        'count_histogram': {str(k): v for k, v in hist.items()},
        'exact_two_frames': exact, 'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(precision, 5), 'recall': round(recall, 5),
        'f1': round(2 * precision * recall / (precision + recall), 5)
        if precision + recall else 0.0,
    }


data = json.loads(SOURCE.read_text())
output = {
    'method': 'Score saved 640 fast-engine boxes alone versus cascade-selected boxes on every frame in each recorded video, using confidence >=0.05 and fixed manually labeled target centers with a 60 px match radius.',
    'limitations': 'Fixed two-ball, stationary scenes only; does not measure arbitrary cluster layouts, stereo range, motion, or unseen false positives outside the recorded scenes.',
    'results': [],
}
for clip in data['results']:
    rows = clip['frames_detail']
    result = {
        'scene': clip['scene'], 'frames': len(rows),
        'fast_detector_latency_ms': clip['fast_detector_ms'],
        'cascade_detector_latency_ms': clip['cascade_detector_ms'],
        'fast_only': score(rows, 'fast_boxes'),
        'cascade_selected': score(rows, 'selected_boxes'),
    }
    output['results'].append(result)
    print(clip['scene'], json.dumps({k: result[k] for k in (
        'fast_only', 'cascade_selected', 'fast_detector_latency_ms',
        'cascade_detector_latency_ms')}))

destination = ROOT / 'media/frc-fuel-cascade-fast-only-evaluation.json'
destination.write_text(json.dumps(output, indent=2) + '\n')
print('saved', destination)
