#!/usr/bin/env python3
"""Sweep cascade fallback thresholds using saved per-frame detector outputs."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'media/frc-fuel-resolution-cascade-runtime.json'
OUTPUT = ROOT / 'media/frc-fuel-cascade-trigger-breakpoints.json'
TARGETS = ((808, 589), (902, 591))
THRESHOLDS = (0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056,
              0.057, 0.0574, 0.05741, 0.0575, 0.0576, 0.0579,
              0.058, 0.059, 0.06)


def score(boxes):
    hits = [False, False]
    extras = 0
    for box in sorted(boxes, key=lambda item: -item['confidence']):
        cx = (box['x1'] + box['x2']) / 2
        cy = (box['y1'] + box['y2']) / 2
        distances = [math.hypot(cx - x, cy - y) for x, y in TARGETS]
        target = min(range(2), key=distances.__getitem__)
        if distances[target] <= 60 and not hits[target]:
            hits[target] = True
        else:
            extras += 1
    return sum(hits), extras


source = json.loads(SOURCE.read_text())
rows = []
for threshold in THRESHOLDS:
    result = {'threshold': threshold, 'frames': 0, 'exact_two': 0,
              'true_positives': 0, 'false_positives': 0,
              'false_negatives': 0, 'fallback_frames': 0, 'by_scene': {}}
    for clip in source['results']:
        scene = {'frames': 0, 'exact_two': 0, 'true_positives': 0,
                 'false_positives': 0, 'false_negatives': 0,
                 'fallback_frames': 0}
        for frame in clip['frames_detail']:
            fast = frame['fast_boxes']
            minimum = min((box['confidence'] for box in fast), default=0.0)
            fallback = len(fast) != 2 or minimum < threshold
            selected = frame['selected_boxes'] if fallback else fast
            hits, extras = score(selected)
            exact = hits == 2 and extras == 0
            for key, value in (
                ('frames', 1), ('exact_two', int(exact)),
                ('true_positives', hits), ('false_positives', extras),
                ('false_negatives', 2 - hits),
                ('fallback_frames', int(fallback)),
            ):
                result[key] += value
                scene[key] += value
        result['by_scene'][clip['scene']] = scene
    rows.append(result)

report = {
    'method': ('Rescore the saved 640 and 1280 outputs from the actual sequential '
               'cascade run at threshold 0.06. For thresholds <=0.06, reuse the '
               'saved 1280 result on frames that would trigger and use the saved '
               '640 detections otherwise.'),
    'target_centers_px': TARGETS,
    'match_radius_px': 60,
    'source': 'media/frc-fuel-resolution-cascade-runtime.json',
    'limitations': ('Offline threshold replay only, two static clips, and fixed '
                    'target-center labels. It measures detector counts and fallback '
                    'frequency, not stereo accuracy, moving scenes, or live runtime.'),
    'results': rows,
}
OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
for row in rows:
    print(f"{row['threshold']:.5f} exact={row['exact_two']}/{row['frames']} "
          f"tp={row['true_positives']} fp={row['false_positives']} "
          f"fn={row['false_negatives']} fallback={row['fallback_frames']}")
print('saved', OUTPUT)
