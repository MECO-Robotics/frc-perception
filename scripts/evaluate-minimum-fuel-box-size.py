#!/usr/bin/env python3
"""Sweep minimum detector box short-side filters on the two saved FUEL clips."""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ((808 / 1600, 589 / 1200), (902 / 1600, 591 / 1200))
INPUTS = (
    ('earlier', 'media/frc-2026-live-two-fuel-60fps-cfr-aspect2-replay-detections.jsonl'),
    ('corrected-settings', 'media/frc-2026-fuel-camera-settings-fixed-60fps-aspect2-replay-detections.jsonl'),
)
MIN_SHORT_SIDES_PX = (0, 55, 60, 63, 65, 70, 75, 80, 90)
MATCH_RADIUS_PX = 60


def med(values):
    return statistics.median(values) if values else None


def mad(values):
    center = med(values)
    return med([abs(value - center) for value in values]) if values else None


def evaluate(events, minimum):
    tp = fp = fn = paired = exact_two = 0
    target_ranges = [[], []]
    target_short_sides = [[], []]
    all_target_short_sides = [[], []]
    frame_counts = []
    for event in events:
        detections = []
        for box in event.get('boxes', []):
            width_px, height_px = box['w'] * 1600, box['h'] * 1200
            cx, cy = box['x'] + box['w'] / 2, box['y'] + box['h'] / 2
            errors = [math.hypot((cx - tx) * 1600, (cy - ty) * 1200) for tx, ty in TARGETS]
            error, target_i = min((value, i) for i, value in enumerate(errors))
            if error <= MATCH_RADIUS_PX:
                all_target_short_sides[target_i].append(min(width_px, height_px))
            if min(width_px, height_px) < minimum:
                continue
            detections.append((box, target_i if error <= MATCH_RADIUS_PX else None,
                               min(width_px, height_px)))
        detections.sort(key=lambda row: row[0].get('confidence', 0), reverse=True)
        matched = [False, False]
        extras = 0
        for box, target_i, short_side in detections:
            if target_i is not None and not matched[target_i]:
                matched[target_i] = True
                if box.get('distance_m') is not None:
                    target_ranges[target_i].append(box['distance_m'])
                target_short_sides[target_i].append(short_side)
                tp += 1
            else:
                extras += 1
        fp += extras
        fn += 2 - sum(matched)
        paired += all(matched)
        exact_two += all(matched) and extras == 0
        frame_counts.append(sum(matched) + extras)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {
        'minimum_short_side_px': minimum,
        'tp': tp, 'fp': fp, 'fn': fn,
        'paired_frames': paired, 'exact_two_frames': exact_two,
        'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4),
        'output_count_histogram': {str(n): frame_counts.count(n) for n in sorted(set(frame_counts))},
        'targets': [
            {'target': i + 1, 'matched_samples': len(vals),
             'median_short_side_px': round(med(target_short_sides[i]), 1)
             if target_short_sides[i] else None,
             'minimum_short_side_px': round(min(target_short_sides[i]), 1)
             if target_short_sides[i] else None,
             'minimum_unfiltered_target_short_side_px': round(min(all_target_short_sides[i]), 1)
             if all_target_short_sides[i] else None,
             'range_median_m': round(med(vals), 4) if vals else None,
             'range_mad_m': round(mad(vals), 4) if vals else None}
            for i, vals in enumerate(target_ranges)
        ],
    }


groups = []
all_events = []
for scene, filename in INPUTS:
    events = [json.loads(line) for line in (ROOT / filename).read_text().splitlines() if line.strip()]
    all_events.extend(events)
    groups.append({'scene': scene, 'source': filename, 'outputs': len(events),
                   'sweep': [evaluate(events, threshold) for threshold in MIN_SHORT_SIDES_PX]})
groups.append({'scene': 'combined', 'source': 'both saved replay logs', 'outputs': len(all_events),
               'sweep': [evaluate(all_events, threshold) for threshold in MIN_SHORT_SIDES_PX]})
output = {
    'method': 'Replay each 0.10-confidence, max-aspect-2.0 detection output while dropping boxes whose shorter side is below the threshold. Match remaining detections to the two fixed target centers within 60 px; unmatched boxes count as false positives.',
    'limitations': 'The only positive examples are two stationary balls at about 1.4-1.6 m in fixed room views. A minimum-size filter may discard farther or partially occluded balls. These scene-specific precision/recall values are not field performance.',
    'inputs': groups,
}
out_path = ROOT / 'media/frc-minimum-fuel-box-size-sweep.json'
out_path.write_text(json.dumps(output, indent=2) + '\n')
for group in groups:
    print(group['scene'], group['outputs'])
    for row in group['sweep']:
        print(row['minimum_short_side_px'], 'P/R/F1=', row['precision'], row['recall'], row['f1'],
              'pair/exact=', row['paired_frames'], row['exact_two_frames'],
              'fp/fn=', row['fp'], row['fn'],
              'ranges=', [(x['range_median_m'], x['range_mad_m']) for x in row['targets']])
