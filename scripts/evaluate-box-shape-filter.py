#!/usr/bin/env python3
"""Offline sweep of confidence and box aspect filters on the two FUEL replays."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ((808 / 1600, 589 / 1200), (902 / 1600, 591 / 1200))
INPUTS = (
    ('earlier', 'media/frc-2026-live-two-fuel-60fps-cfr-replay-detections.jsonl'),
    ('corrected-settings', 'media/frc-2026-fuel-camera-settings-fixed-60fps-replay-detections.jsonl'),
)
CONFIDENCES = (0.10, 0.12, 0.125, 0.15, 0.20, 0.25)
MAX_ASPECT_RATIOS = (None, 1.5, 2.0, 2.5, 3.0)


def evaluate(events, confidence, max_aspect):
    tp = fp = fn = paired = exact_two = 0
    for event in events:
        candidates = []
        for box in event.get('boxes', []):
            if box.get('confidence', 0) < confidence:
                continue
            width, height = box['w'] * 1600, box['h'] * 1200
            ratio = max(width / height, height / width)
            if max_aspect is not None and ratio > max_aspect:
                continue
            candidates.append((box, width, height))
        candidates.sort(key=lambda item: item[0]['confidence'], reverse=True)
        matched = [False, False]
        extras = 0
        for box, _, _ in candidates:
            cx = box['x'] + box['w'] / 2
            cy = box['y'] + box['h'] / 2
            distances = [math.hypot((cx - x) * 1600, (cy - y) * 1200)
                         for x, y in TARGETS]
            distance, target = min((d, i) for i, d in enumerate(distances))
            if distance <= 60 and not matched[target]:
                matched[target] = True
                tp += 1
            else:
                extras += 1
        fp += extras
        fn += 2 - sum(matched)
        paired += all(matched)
        exact_two += all(matched) and extras == 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {'confidence': confidence, 'max_aspect_ratio': max_aspect,
            'frames': len(events), 'paired_frames': paired, 'exact_two_frames': exact_two,
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}


results = []
all_events = []
for label, relative in INPUTS:
    events = [json.loads(line) for line in (ROOT / relative).read_text().splitlines() if line.strip()]
    all_events.extend(events)
    results.append({'video': relative, 'scene': label, 'outputs': len(events),
                    'sweep': [evaluate(events, confidence, aspect)
                              for confidence in CONFIDENCES for aspect in MAX_ASPECT_RATIOS]})
results.append({'video': 'combined saved-video replays', 'scene': 'combined', 'outputs': len(all_events),
                'sweep': [evaluate(all_events, confidence, aspect)
                          for confidence in CONFIDENCES for aspect in MAX_ASPECT_RATIOS]})
live_path = ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps-aspect2-replay-detections.jsonl'
live_events = [json.loads(line) for line in live_path.read_text().splitlines() if line.strip()]
output = {
    'method': 'Fixed target centers (808,589) and (902,591) in each 1600x1200 left view; match within 60 px. Count unmatched boxes as false positives. The logs contain candidates produced at base confidence 0.10; confidence and max width/height aspect ratio are replay filters only.',
    'limits': 'Only two clear, stationary balls in a fixed room scene. Target windows are scene-specific. This does not establish performance on moving field scenes, real occlusion, or distant balls.',
    'results': results,
    'same_live_replay_confidence_sweep': {
        'source': str(live_path.relative_to(ROOT)),
        'outputs': len(live_events),
        'note': 'Re-filters one live replay already decoded at confidence 0.10 and aspect ratio 2.0, holding its output sequence fixed while raising confidence.',
        'results': [evaluate(live_events, confidence, 2.0)
                    for confidence in (0.10, 0.12, 0.125, 0.15, 0.20, 0.25)],
    },
}
(ROOT / 'media/frc-box-aspect-filter-sweep.json').write_text(json.dumps(output, indent=2) + '\n')
for group in results:
    print(group['scene'], group['outputs'])
    for row in group['sweep']:
        if row['max_aspect_ratio'] in (None, 2.0) and row['confidence'] in (0.10, 0.15, 0.20, 0.25):
            print(row)
print('same-live-replay-threshold-sweep')
for row in output['same_live_replay_confidence_sweep']['results']:
    print(row)
