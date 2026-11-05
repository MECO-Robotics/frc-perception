#!/usr/bin/env python3
"""Evaluate per-frame confidence cutoff selected by left-image luminance."""
import json
import math
import statistics
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SOURCE = json.loads((ROOT / 'media/frc-fuel-low-confidence-evaluation.json').read_text())
TARGETS = ((808.0, 589.0), (902.0, 591.0))
MATCH_RADIUS = 60
LUMA_CUTS = tuple(range(10, 61, 2))
LOW_CONFIDENCE = 0.05
HIGH_CONFIDENCE = 0.075


def score(detections, cutoff):
    hits = [None, None]
    extras = 0
    for box in sorted((item for item in detections if item['confidence'] >= cutoff),
                      key=lambda item: item['confidence'], reverse=True):
        center = ((box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2)
        distance, index = min((math.dist(center, target), i)
                              for i, target in enumerate(TARGETS))
        if distance <= MATCH_RADIUS and hits[index] is None:
            hits[index] = box
        else:
            extras += 1
    return sum(box is not None for box in hits), extras


def summarize(rows):
    tp = sum(row['tp'] for row in rows)
    fp = sum(row['fp'] for row in rows)
    fn = 2 * len(rows) - tp
    exact = sum(row['tp'] == 2 and row['fp'] == 0 for row in rows)
    return {'frames': len(rows), 'tp': tp, 'fp': fp, 'fn': fn,
            'exact_two_frames': exact,
            'precision': round(tp / (tp + fp), 6) if tp + fp else None,
            'recall': round(tp / (tp + fn), 6) if tp + fn else None}


output = {
    'method': ('Rescore every saved low-cutoff detector output at fixed 0.05 and '
               '0.075, then test a simple per-frame rule: use 0.05 when mean '
               'left-image grayscale luma is at/below a cutoff, otherwise 0.075.'),
    'limitations': ('Two static videos and manually fixed target centers. Global '
                    'mean luma may not represent local ball visibility; detector '
                    'outputs were generated offline and no stereo or motion '
                    'latency is included. Luma cutoffs are exploratory and must '
                    'be selected on one clip and checked on the other.'),
    'rule': {'low_luma_confidence': LOW_CONFIDENCE,
             'high_luma_confidence': HIGH_CONFIDENCE,
             'thresholds': LUMA_CUTS},
    'results': [],
}

for clip in SOURCE['results']:
    capture = cv2.VideoCapture(str(ROOT / clip['video']))
    fps = capture.get(cv2.CAP_PROP_FPS)
    per_frame = []
    for index, detections in enumerate(clip['per_frame']):
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame[:, :1600], cv2.COLOR_BGR2GRAY)
        luma = float(gray.mean())
        counts = {}
        for confidence in (LOW_CONFIDENCE, HIGH_CONFIDENCE):
            tp, fp = score(detections, confidence)
            counts[str(confidence)] = {'tp': tp, 'fp': fp}
        per_frame.append({'frame': index, 'time_s': index / fps,
                          'mean_luma': luma, 'fixed': counts})
    capture.release()
    fixed = {}
    for confidence in (LOW_CONFIDENCE, HIGH_CONFIDENCE):
        fixed[str(confidence)] = summarize([
            {'tp': row['fixed'][str(confidence)]['tp'],
             'fp': row['fixed'][str(confidence)]['fp']} for row in per_frame])
    adaptive = []
    for threshold in LUMA_CUTS:
        rows = []
        for row in per_frame:
            confidence = LOW_CONFIDENCE if row['mean_luma'] <= threshold else HIGH_CONFIDENCE
            rows.append({'tp': row['fixed'][str(confidence)]['tp'],
                         'fp': row['fixed'][str(confidence)]['fp']})
        adaptive.append({'luma_cutoff': threshold, **summarize(rows)})
    # Explicit first-half-second startup results, where the policy should retain
    # low-light recall while testing higher cutoffs on normal frames.
    startup_count = max(1, round(.5 * fps))
    startup = {}
    for confidence in (LOW_CONFIDENCE, HIGH_CONFIDENCE):
        startup[str(confidence)] = summarize([
            {'tp': row['fixed'][str(confidence)]['tp'],
             'fp': row['fixed'][str(confidence)]['fp']}
            for row in per_frame[:startup_count]])
    result = {'scene': clip['scene'], 'video': clip['video'],
              'frames': len(per_frame), 'fixed': fixed,
              'first_0_5s': startup, 'adaptive': adaptive,
              'per_frame_luma': [row['mean_luma'] for row in per_frame]}
    output['results'].append(result)
    best = max(adaptive, key=lambda item: (item['exact_two_frames'], -item['fp']))
    print(clip['scene'], 'fixed', fixed, 'startup', startup, 'best adaptive', best)

path = ROOT / 'media/frc-fuel-luma-adaptive-confidence-evaluation.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print('saved', path)
