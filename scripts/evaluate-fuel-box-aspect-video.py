#!/usr/bin/env python3
"""Rescore saved stereo-video detections with tighter FUEL box aspect limits."""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'media/frc-fuel-dense-count-nms010.json'
TARGETS = ((808, 589), (902, 591))
CONFIDENCE = 0.05
ASPECT_LIMITS = (1.2, 1.4, 1.6, 1.8, 2.0)
source = json.loads(SOURCE.read_text())
output = {
    'method': ('Rescore the two saved 600-frame stereo clips at confidence 0.05 after '
               'filtering by maximum box aspect ratio. Match remaining detections to '
               'the two fixed labeled target centers within 60 px.'),
    'source': str(SOURCE.relative_to(ROOT)),
    'confidence': CONFIDENCE,
    'target_centers_px': TARGETS,
    'aspect_limits': ASPECT_LIMITS,
    'results': [],
}
for clip in source['results']:
    result = {'scene': clip['scene'], 'frames': len(clip['per_frame']), 'limits': []}
    for limit in ASPECT_LIMITS:
        pairs = exact = tp = fp = fn = suppressed_target_instances = 0
        target_aspects = []
        for row in clip['per_frame']:
            kept = []
            for box in sorted((x for x in row['detections'] if x['confidence'] >= CONFIDENCE),
                              key=lambda x: -x['confidence']):
                w, h = box['x2']-box['x1'], box['y2']-box['y1']
                aspect = max(w/h, h/w)
                cx, cy = (box['x1']+box['x2'])/2, (box['y1']+box['y2'])/2
                distance, target_i = min((math.hypot(cx-x, cy-y), i)
                                         for i, (x, y) in enumerate(TARGETS))
                if distance <= 60:
                    target_aspects.append(aspect)
                if aspect <= limit:
                    kept.append(box)
            unfiltered_targets = [None, None]
            for box in sorted((x for x in row['detections'] if x['confidence'] >= CONFIDENCE),
                              key=lambda x: -x['confidence']):
                cx, cy = (box['x1']+box['x2'])/2, (box['y1']+box['y2'])/2
                distance, target_i = min((math.hypot(cx-x, cy-y), i)
                                         for i, (x, y) in enumerate(TARGETS))
                if distance <= 60 and unfiltered_targets[target_i] is None:
                    unfiltered_targets[target_i] = box
            matched = [None, None]
            extras = 0
            for box in kept:
                cx, cy = (box['x1']+box['x2'])/2, (box['y1']+box['y2'])/2
                distance, target_i = min((math.hypot(cx-x, cy-y), i)
                                         for i, (x, y) in enumerate(TARGETS))
                if distance <= 60 and matched[target_i] is None:
                    matched[target_i] = box
                else:
                    extras += 1
            count = sum(box is not None for box in matched)
            tp += count; fn += 2-count; fp += extras
            pairs += count == 2
            exact += count == 2 and extras == 0
            suppressed_target_instances += sum(box is not None for box in unfiltered_targets) - count
        result['limits'].append({
            'max_aspect_ratio': limit, 'paired_frames': pairs,
            'exact_two_frames': exact, 'tp': tp, 'fp': fp, 'fn': fn,
            'suppressed_target_instances': suppressed_target_instances,
            'max_target_aspect_observed': round(max(target_aspects), 4) if target_aspects else None,
        })
    output['results'].append(result)
path = ROOT / 'media/frc-fuel-box-aspect-video-screen.json'
path.write_text(json.dumps(output, indent=2)+'\n')
for result in output['results']:
    print(result['scene'], result['limits'])
print('saved', path)
