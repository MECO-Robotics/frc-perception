#!/usr/bin/env python3
"""Sweep stricter detector box-aspect limits using dense saved-frame outputs."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'media/frc-fuel-dense-count-evaluation.json'
LIMITS = (1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0)
CONFIDENCE = 0.10
TARGETS = ((808, 589), (902, 591))


def aspect(box):
    width, height = box['x2'] - box['x1'], box['y2'] - box['y1']
    return max(width / height, height / width) if min(width, height) > 0 else float('inf')


def evaluate(frames, limit):
    tp = fp = fn = exact = paired = 0
    target_aspects = [[], []]
    shape_rejected_targets = 0
    for frame in frames:
        boxes = [box for box in frame['detections']
                 if box['confidence'] >= CONFIDENCE and aspect(box) <= limit]
        matched = [None, None]
        extras = 0
        for box in sorted(boxes, key=lambda item: item['confidence'], reverse=True):
            cx = (box['x1'] + box['x2']) / 2
            cy = (box['y1'] + box['y2']) / 2
            distance, index = min((math.hypot(cx - x, cy - y), i)
                                  for i, (x, y) in enumerate(TARGETS))
            if distance <= 60 and matched[index] is None:
                matched[index] = box
                target_aspects[index].append(aspect(box))
            else:
                extras += 1
        for index, (x, y) in enumerate(TARGETS):
            if matched[index] is not None:
                continue
            unfiltered = [box for box in frame['detections']
                          if box['confidence'] >= CONFIDENCE and
                          math.hypot((box['x1']+box['x2'])/2-x,
                                     (box['y1']+box['y2'])/2-y) <= 60]
            if unfiltered and all(aspect(box) > limit for box in unfiltered):
                shape_rejected_targets += 1
        count = sum(box is not None for box in matched)
        tp += count
        fn += 2 - count
        fp += extras
        paired += count == 2
        exact += count == 2 and extras == 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {
        'max_aspect_ratio': limit,
        'tp': tp, 'fp': fp, 'fn': fn,
        'paired_frames': paired, 'exact_two_frames': exact,
        'precision': round(precision, 5), 'recall': round(recall, 5),
        'f1': round(2 * precision * recall / (precision + recall), 5)
        if precision + recall else 0,
        'shape_rejected_target_events': shape_rejected_targets,
        'target_aspect_observed_max': [round(max(values), 4) if values else None
                                       for values in target_aspects],
    }


source = json.loads(SOURCE.read_text())
output = {
    'method': 'Re-filter the active FRC9550 1280x960 per-frame candidates at confidence 0.10 with stricter symmetric width/height aspect limits. The stored candidates were originally decoded with the active maximum limit of 2.0.',
    'limitations': 'Only stationary saved clips. Stricter limits can be evaluated from these candidates, but wider limits above 2.0 cannot because the decoder already rejected those boxes.',
    'source': str(SOURCE.relative_to(ROOT)),
    'confidence': CONFIDENCE,
    'results': [],
}
for clip in source['results']:
    output['results'].append({
        'scene': clip['scene'],
        'video': clip['video'],
        'frames': len(clip['per_frame']),
        'sweep': [evaluate(clip['per_frame'], limit) for limit in LIMITS],
    })
path = ROOT / 'media/frc-fuel-dense-aspect-sweep.json'
path.write_text(json.dumps(output, indent=2) + '\n')
for clip in output['results']:
    print(clip['scene'])
    for row in clip['sweep']:
        print(row['max_aspect_ratio'], 'exact', row['exact_two_frames'],
              'fp/fn', row['fp'], row['fn'], 'F1', row['f1'],
              'shape-rejected targets', row['shape_rejected_target_events'],
              'target aspect maxima', row['target_aspect_observed_max'])
print('saved', path)
