#!/usr/bin/env python3
"""Cross-check public-image adaptive confidence candidates on both stereo clips."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ((808.0, 589.0), (902.0, 591.0))
MATCH_RADIUS = 60.0
NMS_INPUTS = {
    0.10: ROOT / 'media/frc-fuel-dense-count-nms010.json',
    0.15: ROOT / 'media/frc-fuel-dense-count-nms015.json',
}
POLICIES = (
    {'name': 'fixed_005', 'trigger': 0, 'high': 0.05},
    {'name': 'trigger3_high020', 'trigger': 3, 'high': 0.20},
    {'name': 'trigger2_high030', 'trigger': 2, 'high': 0.30},
    {'name': 'trigger3_high030', 'trigger': 3, 'high': 0.30},
)


def score(frames, trigger, high):
    counts = {}
    tp = fp = fn = adaptive_frames = exact_two = 0
    for frame in frames:
        candidates = [item for item in frame['detections']
                      if item['confidence'] >= 0.05]
        if trigger and len(candidates) >= trigger:
            cutoff = high
            adaptive_frames += 1
        else:
            cutoff = 0.05
        predictions = [item for item in candidates if item['confidence'] >= cutoff]
        hits = 0
        used = set()
        for tx, ty in TARGETS:
            options = []
            for index, box in enumerate(predictions):
                if index in used:
                    continue
                cx = (box['x1'] + box['x2']) / 2
                cy = (box['y1'] + box['y2']) / 2
                distance = math.hypot(cx - tx, cy - ty)
                if distance <= MATCH_RADIUS:
                    options.append((distance, index))
            if options:
                _, selected = min(options)
                used.add(selected)
                hits += 1
        tp += hits
        fp += len(predictions) - hits
        fn += 2 - hits
        if len(predictions) == 2 and hits == 2:
            exact_two += 1
        counts[len(predictions)] = counts.get(len(predictions), 0) + 1
    return {
        'frames': len(frames),
        'count_histogram': {str(key): value for key, value in sorted(counts.items())},
        'exact_two_target_frames': exact_two,
        'tp': tp, 'fp': fp, 'fn': fn,
        'adaptive_frames': adaptive_frames,
    }


results = []
for nms, path in NMS_INPUTS.items():
    data = json.loads(path.read_text())
    for scene in data['results']:
        row = {'scene': scene['scene'], 'nms_iou': nms,
               'policies': {}}
        for policy in POLICIES:
            metrics = score(scene['per_frame'], policy['trigger'], policy['high'])
            row['policies'][policy['name']] = metrics
        results.append(row)

output = {
    'method': ('Use cached per-frame YOLO26n confidence candidates at NMS 0.10 and 0.15 '
               'on both complete saved stereo videos. Simulate confidence adaptation '
               'from a 0.05 candidate floor, then match target centers within 60 px.'),
    'policies': POLICIES,
    'limitations': ('The two clips contain only two stationary visible targets; this '
                    'tests the dim opening and repeatability, not varied cluster sizes, '
                    'motion, real occlusion, or absolute range. The score uses 2D target '
                    'windows; stereo gate/position accuracy are not recalculated here.'),
    'results': results,
}
out = ROOT / 'media/frc-fuel-adaptive-confidence-crosscheck.json'
out.write_text(json.dumps(output, indent=2) + '\n')
for item in results:
    print(item['scene'], item['nms_iou'], json.dumps(item['policies']))
print('saved', out)
