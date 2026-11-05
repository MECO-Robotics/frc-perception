#!/usr/bin/env python3
"""Compare two- and three-candidate adaptive confidence triggers offline."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'media/frc-fuel-dense-count-nms015.json'
OUTPUT = ROOT / 'media/frc-fuel-adaptive-trigger-two-analysis.json'
TARGETS = ((808, 589), (902, 591))
BASE_CONFIDENCE = 0.05
HIGH_THRESHOLDS = (0.10, 0.20, 0.30)
TRIGGERS = (2, 3)


def score(rows, trigger, high_confidence):
    totals = dict(exact_two_frames=0, paired_frames=0, tp=0, fp=0, fn=0,
                  triggered_frames=0, frames=len(rows))
    for row in rows:
        base = [d for d in row['detections']
                if d['confidence'] >= BASE_CONFIDENCE]
        triggered = len(base) >= trigger
        cutoff = high_confidence if triggered else BASE_CONFIDENCE
        totals['triggered_frames'] += int(triggered)
        hits = [False, False]
        extras = 0
        selected = (d for d in row['detections'] if d['confidence'] >= cutoff)
        for detection in sorted(selected, key=lambda d: -d['confidence']):
            cx = (detection['x1'] + detection['x2']) / 2
            cy = (detection['y1'] + detection['y2']) / 2
            distance, target = min(
                (math.hypot(cx - x, cy - y), i)
                for i, (x, y) in enumerate(TARGETS))
            if distance <= 60 and not hits[target]:
                hits[target] = True
            else:
                extras += 1
        count = sum(hits)
        totals['tp'] += count
        totals['fn'] += 2 - count
        totals['fp'] += extras
        totals['paired_frames'] += int(count == 2)
        totals['exact_two_frames'] += int(count == 2 and extras == 0)
    return totals


data = json.loads(SOURCE.read_text())
out = {
    'method': ('Rescore saved NMS-IoU-0.10 per-frame FRC YOLO26n detections. '
               'At each trigger, raise confidence from 0.05 to the listed '
               'high threshold when that frame has at least K base detections.'),
    'source': str(SOURCE.relative_to(ROOT)),
    'limitations': ('Two fixed stationary balls only. This compares count '
                    'behavior and does not evaluate moving scenes, occlusion, '
                    'or absolute range accuracy. Retained box coordinates '
                    'are unchanged by confidence filtering.'),
    'base_confidence': BASE_CONFIDENCE,
    'nms_iou': data['nms_iou'],
    'comparisons': [],
}
for trigger in TRIGGERS:
    for high in HIGH_THRESHOLDS:
        item = {'trigger_candidates': trigger,
                'high_confidence': high,
                'clips': {}}
        for clip in data['results']:
            item['clips'][clip['scene']] = score(
                clip['per_frame'], trigger, high)
        out['comparisons'].append(item)

OUTPUT.write_text(json.dumps(out, indent=2) + '\n')
for item in out['comparisons']:
    print('K', item['trigger_candidates'], 'high', item['high_confidence'],
          {name: (score['exact_two_frames'], score['fp'], score['fn'],
                  score['triggered_frames'])
           for name, score in item['clips'].items()})
print('saved', OUTPUT)
