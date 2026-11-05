#!/usr/bin/env python3
"""Score a 640-first, conditional 1280x960 fallback detector on both clips."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAST = json.loads((ROOT / 'media/frc-fuel-dense-count-640.json').read_text())
ACCURATE = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
TARGETS = ((808, 589), (902, 591))
MATCH_RADIUS_PX = 60
BASE_CONFIDENCE = 0.05
MIN_CONFIDENCE_FALLBACK = (0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                           0.12, 0.20, 0.30, 0.40, 0.50, 0.60)


def match(detections, threshold=BASE_CONFIDENCE):
    targets = [None, None]
    extras = 0
    for box in sorted((item for item in detections
                       if item['confidence'] >= threshold),
                      key=lambda item: item['confidence'], reverse=True):
        cx = (box['x1'] + box['x2']) / 2
        cy = (box['y1'] + box['y2']) / 2
        distance, target_index = min(
            (math.hypot(cx - x, cy - y), index)
            for index, (x, y) in enumerate(TARGETS))
        if distance <= MATCH_RADIUS_PX and targets[target_index] is None:
            targets[target_index] = box
        else:
            extras += 1
    return targets, extras


def metrics(rows):
    tp = fp = fn = paired = exact_two = 0
    for row in rows:
        detections, _ = row
        targets, extras = match(detections)
        hits = sum(item is not None for item in targets)
        tp += hits
        fn += 2 - hits
        fp += extras
        paired += int(hits == 2)
        exact_two += int(hits == 2 and extras == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {'target_pairs': paired, 'exact_two_outputs': exact_two,
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(precision, 5), 'recall': round(recall, 5),
            'f1': round(f1, 5)}


output = {
    'method': ('Re-score every saved per-frame candidate from the FRC YOLO26n '
               '640 engine and the active 1280x960 engine at confidence 0.05. '
               'For a threshold q, use 640 unless its candidate count is not '
               'two or the lowest candidate confidence is below q; then replace '
               'that frame with the 1280x960 output. Estimate mean detector time '
               'as measured 640 median plus fallback fraction times measured '
               '1280x960 median.'),
    'settings': {'confidence': BASE_CONFIDENCE,
                 'min_confidence_fallback_thresholds': MIN_CONFIDENCE_FALLBACK,
                 'match_radius_px': MATCH_RADIUS_PX,
                 'measured_median_single_view_ms_by_scene': {
                     fast_clip['scene']: {
                         '640': fast_clip['single_view_inference_ms_median'],
                         '1280x960': accurate_clip['single_view_inference_ms_median']}
                     for fast_clip, accurate_clip in zip(FAST['results'], ACCURATE['results'])}},
    'limitations': ('The two clips contain two static targets at one camera pose. '
                    'Fixed target windows are evaluation labels, not available '
                    'to the runtime policy. Expected time is an estimate from '
                    'separate detector medians; it excludes preprocessing, '
                    'stereo, queues, and fallback scheduling. It assumes the '
                    'saved model outputs represent the same confidence-0.05 '
                    'runtime candidates.'),
    'results': [],
}

for fast_clip, accurate_clip in zip(FAST['results'], ACCURATE['results']):
    fast_ms = fast_clip['single_view_inference_ms_median']
    fallback_ms = accurate_clip['single_view_inference_ms_median']
    fast_rows = fast_clip['per_frame']
    accurate_rows = accurate_clip['per_frame']
    assert len(fast_rows) == len(accurate_rows)
    direct_fast = [(row['detections'], False) for row in fast_rows]
    direct_accurate = [(row['detections'], False) for row in accurate_rows]
    policies = []
    direct_fast_metrics = metrics(direct_fast)
    direct_accurate_metrics = metrics(direct_accurate)
    for min_conf in MIN_CONFIDENCE_FALLBACK:
        selected = []
        fallback_count = 0
        for fast_row, accurate_row in zip(fast_rows, accurate_rows):
            candidates = [box for box in fast_row['detections']
                          if box['confidence'] >= BASE_CONFIDENCE]
            should_fallback = (len(candidates) != 2 or
                               min((box['confidence'] for box in candidates),
                                   default=0) < min_conf)
            if should_fallback:
                selected.append((accurate_row['detections'], True))
                fallback_count += 1
            else:
                selected.append((fast_row['detections'], False))
        scored = metrics(selected)
        fallback_fraction = fallback_count / len(selected) if selected else 0
        scored.update({
            'min_confidence_fallback_threshold': min_conf,
            'fallback_frames': fallback_count,
            'fallback_fraction': round(fallback_fraction, 5),
            'estimated_mean_detector_ms': round(fast_ms + fallback_fraction * fallback_ms, 2),
            'estimated_worst_case_detector_ms': round(fast_ms + fallback_ms, 2)
            if fallback_count else round(fast_ms, 2),
        })
        policies.append(scored)
    output['results'].append({
        'scene': fast_clip['scene'], 'video': fast_clip['video'],
        'frames': len(fast_rows), 'fast_640_baseline': direct_fast_metrics,
        'full_1280x960_baseline': direct_accurate_metrics,
        'cascade_policies': policies,
    })

path = ROOT / 'media/frc-fuel-resolution-cascade-evaluation.json'
path.write_text(json.dumps(output, indent=2) + '\n')
for clip in output['results']:
    print(clip['scene'], '640', clip['fast_640_baseline'],
          '1280x960', clip['full_1280x960_baseline'])
    for policy in clip['cascade_policies']:
        print('cascade', policy['min_confidence_fallback_threshold'],
              'pairs/exact', policy['target_pairs'], policy['exact_two_outputs'],
              'fallback', policy['fallback_fraction'],
              'estimated ms', policy['estimated_mean_detector_ms'],
              'f1', policy['f1'])
print('saved', path)
