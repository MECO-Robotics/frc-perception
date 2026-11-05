#!/usr/bin/env python3
"""Test causal track confirmation and range smoothing at Scout's update rate."""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / 'media/frc-fuel-dense-count-evaluation.json'
STEREO = ROOT / 'media/frc-fuel-dense-stereo-evaluation.json'
STRIDE = 7
PHASES = range(STRIDE)
WINDOWS = (3, 5)


def med(values):
    return statistics.median(values) if values else None


def mad(values):
    center = med(values)
    return med([abs(value - center) for value in values]) if values else None


def p95_deviation(values, center):
    if not values or center is None:
        return None
    deviations = sorted(abs(value - center) for value in values)
    return deviations[math.ceil(.95 * len(deviations)) - 1]


def targets_for_frame(frame):
    targets = ((808, 589), (902, 591))
    chosen = [None, None]
    for box in sorted((item for item in frame['detections'] if item['confidence'] >= .10),
                      key=lambda item: item['confidence'], reverse=True):
        cx, cy = (box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2
        distance, index = min((math.hypot(cx-x, cy-y), i)
                              for i, (x, y) in enumerate(targets))
        if distance <= 60 and chosen[index] is None:
            chosen[index] = box
    return chosen


counts = json.loads(COUNTS.read_text())
stereo = json.loads(STEREO.read_text())
out = {
    'method': 'Subsample every seventh decoded frame (about 8.6 Hz) across all seven possible phases, then compare raw target presence/ranges with causal majority confirmation and rolling medians over the last 3 or 5 available outputs.',
    'limitations': 'Fixed two-target scene and manually labeled centers. Track persistence can retain stale counts/positions when a ball moves away; these clips do not test motion, crossings, or real occlusion. Range spread is repeatability, not absolute accuracy.',
    'sampling_stride': STRIDE,
    'windows': [],
    'clips': [],
}

for count_clip, stereo_clip in zip(counts['results'], stereo['results']):
    frames = count_clip['per_frame']
    depth_frames = stereo_clip['frames']
    assert len(frames) == len(depth_frames)
    phase_rows = []
    for phase in PHASES:
        indices = list(range(phase, len(frames), STRIDE))
        events = []
        for index in indices:
            chosen = targets_for_frame(frames[index])
            depth_row = depth_frames[index]
            present = [box is not None for box in chosen]
            ranges = depth_row['ranges_m']
            spacing = depth_row['pair_spacing_m']
            events.append({'frame': index, 'time_s': frames[index]['time_s'],
                           'present': present, 'ranges': ranges,
                           'spacing': spacing})
        raw_pairs = sum(all(event['present']) for event in events)
        raw_target_ranges = [[event['ranges'][target] for event in events
                              if event['present'][target] and event['ranges'][target] is not None]
                             for target in range(2)]
        raw_spacing = [event['spacing'] for event in events if event['spacing'] is not None]
        filtered = {}
        for window in WINDOWS:
            confirmed_pairs = 0
            smoothed_ranges = [[], []]
            smoothed_spacing = []
            for i, event in enumerate(events):
                history = events[max(0, i-window+1):i+1]
                confirmed = [
                    sum(sample['present'][target] for sample in history) >= len(history)//2+1
                    for target in range(2)
                ]
                confirmed_pairs += all(confirmed)
                if not all(confirmed):
                    continue
                for target in range(2):
                    values = [sample['ranges'][target] for sample in history
                              if sample['present'][target] and sample['ranges'][target] is not None]
                    if values:
                        smoothed_ranges[target].append(med(values))
                spacing_values = [sample['spacing'] for sample in history
                                  if sample['spacing'] is not None and
                                  all(sample['present'])]
                if spacing_values:
                    smoothed_spacing.append(med(spacing_values))
            filtered[str(window)] = {
                'confirmed_pair_outputs': confirmed_pairs,
                'raw_pair_outputs': raw_pairs,
                'sampled_outputs': len(events),
                'confirmation_delay_s_upper_bound': round((window-1)*STRIDE/59.900166, 3),
                'targets': [{
                    'raw_valid_ranges': len(raw_target_ranges[target]),
                    'raw_range_median_m': round(med(raw_target_ranges[target]), 4)
                    if raw_target_ranges[target] else None,
                    'raw_range_mad_m': round(mad(raw_target_ranges[target]), 4)
                    if raw_target_ranges[target] else None,
                    'smoothed_valid_ranges': len(smoothed_ranges[target]),
                    'smoothed_range_median_m': round(med(smoothed_ranges[target]), 4)
                    if smoothed_ranges[target] else None,
                    'smoothed_range_mad_m': round(mad(smoothed_ranges[target]), 4)
                    if smoothed_ranges[target] else None,
                    'raw_p95_abs_deviation_m': round(p95_deviation(
                        raw_target_ranges[target], med(raw_target_ranges[target])), 4)
                        if raw_target_ranges[target] else None,
                    'smoothed_p95_abs_deviation_m': round(p95_deviation(
                        smoothed_ranges[target], med(raw_target_ranges[target])), 4)
                        if smoothed_ranges[target] else None,
                } for target in range(2)],
                'raw_pair_spacing_m': {
                    'valid': len(raw_spacing),
                    'median': round(med(raw_spacing), 4) if raw_spacing else None,
                    'mad': round(mad(raw_spacing), 4) if raw_spacing else None,
                    'p95_abs_deviation_m': round(p95_deviation(
                        raw_spacing, med(raw_spacing)), 4) if raw_spacing else None,
                },
                'smoothed_pair_spacing_m': {
                    'valid': len(smoothed_spacing),
                    'median': round(med(smoothed_spacing), 4) if smoothed_spacing else None,
                    'mad': round(mad(smoothed_spacing), 4) if smoothed_spacing else None,
                    'p95_abs_deviation_m': round(p95_deviation(
                        smoothed_spacing, med(raw_spacing)), 4)
                        if smoothed_spacing and raw_spacing else None,
                },
            }
        phase_rows.append({'phase': phase, 'sampled_outputs': len(events),
                           'raw_pair_outputs': raw_pairs, 'methods': filtered})
    # Average counts over the phase sweeps so no single stride alignment dominates.
    aggregate = {}
    for window in WINDOWS:
        values = [row['methods'][str(window)]['confirmed_pair_outputs']
                  for row in phase_rows]
        raw_values = [row['raw_pair_outputs'] for row in phase_rows]
        aggregate[str(window)] = {
            'mean_raw_pair_outputs': round(statistics.mean(raw_values), 2),
            'mean_confirmed_pair_outputs': round(statistics.mean(values), 2),
            'mean_additional_pair_outputs': round(statistics.mean(values) - statistics.mean(raw_values), 2),
            'mean_sampled_outputs': round(statistics.mean(row['sampled_outputs']
                                                          for row in phase_rows), 2),
        }
    out['clips'].append({'scene': count_clip['scene'], 'video': count_clip['video'],
                         'phase_sweeps': phase_rows, 'phase_average_counts': aggregate})

path = ROOT / 'media/frc-fuel-dense-temporal-sweep.json'
path.write_text(json.dumps(out, indent=2) + '\n')
for clip in out['clips']:
    print(clip['scene'], 'phase average', clip['phase_average_counts'])
    for window in WINDOWS:
        methods = [row['methods'][str(window)] for row in clip['phase_sweeps']]
        print(' window', window, 'range MADs per target', [
            (round(statistics.mean(row['targets'][i]['raw_range_mad_m'] for row in methods), 4),
             round(statistics.mean(row['targets'][i]['smoothed_range_mad_m'] for row in methods), 4))
            for i in range(2)], 'spacing MAD',
            round(statistics.mean(row['raw_pair_spacing_m']['mad'] for row in methods), 4),
            '->', round(statistics.mean(row['smoothed_pair_spacing_m']['mad'] for row in methods), 4))
print('saved', path)
