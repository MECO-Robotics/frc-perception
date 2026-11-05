#!/usr/bin/env python3
"""Compare raw detections with causal track voting and range medians on saved clips.

This is a fixed-scene replay diagnostic: target gates are centered on the two
known balls in these clips. It measures temporal stability, not generalization.
"""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    ('earlier', 'media/frc-2026-live-two-fuel-60fps-cfr-aspect2-replay-detections.jsonl'),
    ('corrected-settings', 'media/frc-2026-fuel-camera-settings-fixed-60fps-aspect2-replay-detections.jsonl'),
)
# Normalized target centers from the fixed saved scene; coordinates are full left-view fractions.
TARGETS = ((808 / 1600, 589 / 1200), (902 / 1600, 591 / 1200))
GATE_PX = 60
WINDOWS = (3, 5, 9)


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(value - center) for value in values]) if values else None


def summarize(path, scene):
    events = [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]
    tracks = [[None, None] for _ in events]
    extra_counts = []
    counts = []
    elapsed = []
    for frame_i, event in enumerate(events):
        elapsed.append(float(event.get('elapsed_s', frame_i / 7.0)))
        chosen = [None, None]
        extras = 0
        for box in event.get('boxes', []):
            cx, cy = box['x'] + box['w'] / 2, box['y'] + box['h'] / 2
            errors = [math.hypot((cx - tx) * 1600, (cy - ty) * 1200) for tx, ty in TARGETS]
            error, target_i = min((value, i) for i, value in enumerate(errors))
            if error <= GATE_PX:
                if chosen[target_i] is None or box['confidence'] > chosen[target_i]['confidence']:
                    if chosen[target_i] is not None:
                        extras += 1
                    chosen[target_i] = box
                else:
                    extras += 1
            else:
                extras += 1
        tracks[frame_i] = chosen
        counts.append(sum(item is not None for item in chosen))
        extra_counts.append(extras)

    result = {'scene': scene, 'events': len(events), 'target_gate_px': GATE_PX,
              'raw_target_count_histogram': {str(n): counts.count(n) for n in sorted(set(counts))},
              'raw_extra_detections': sum(extra_counts), 'methods': []}
    base_by_target = []
    for target_i in range(2):
        ranges = [frame[target_i]['distance_m'] for frame in tracks
                  if frame[target_i] is not None and frame[target_i].get('distance_m') is not None]
        base_by_target.append(ranges)
    for width in WINDOWS:
        half = width // 2
        filtered_tracks = [[False, False] for _ in events]
        filtered_ranges = [[], []]
        for i in range(len(events)):
            start = max(0, i - width + 1)  # causal; no future frames
            for target_i in range(2):
                history = tracks[start:i + 1]
                visible = sum(frame[target_i] is not None for frame in history)
                confirmed = visible >= (len(history) // 2 + 1)
                filtered_tracks[i][target_i] = confirmed
                if confirmed:
                    # Causal median of only currently observed ranges in the window.
                    vals = [frame[target_i]['distance_m'] for frame in history
                            if frame[target_i] is not None and frame[target_i].get('distance_m') is not None]
                    if vals:
                        filtered_ranges[target_i].append(statistics.median(vals))
        smoothed_counts = [sum(row) for row in filtered_tracks]
        targets = []
        for target_i in range(2):
            raw = base_by_target[target_i]
            smooth = filtered_ranges[target_i]
            targets.append({
                'target': target_i + 1,
                'raw_valid_ranges': len(raw),
                'raw_range_median_m': round(median(raw), 4) if raw else None,
                'raw_range_mad_m': round(mad(raw), 4) if raw else None,
                'smoothed_valid_ranges': len(smooth),
                'smoothed_range_median_m': round(median(smooth), 4) if smooth else None,
                'smoothed_range_mad_m': round(mad(smooth), 4) if smooth else None,
                'raw_p95_abs_deviation_from_clip_median_m': round(
                    sorted([abs(v - median(raw)) for v in raw])[min(len(raw)-1, math.ceil(.95*len(raw))-1)], 4)
                    if raw else None,
                'smoothed_p95_abs_deviation_from_clip_median_m': round(
                    sorted([abs(v - median(raw)) for v in smooth])[min(len(smooth)-1, math.ceil(.95*len(smooth))-1)], 4)
                    if smooth and raw else None,
            })
        result['methods'].append({
            'causal_window_events': width,
            'count_histogram_after_majority_confirmation': {
                str(n): smoothed_counts.count(n) for n in sorted(set(smoothed_counts))},
            'exact_pair_events': smoothed_counts.count(2),
            'count_errors_vs_known_two_piece_scene': sum(n != 2 for n in smoothed_counts),
            'max_confirmation_delay_s_at_observed_sampling': round(
                (width - 1) * median([b-a for a, b in zip(elapsed, elapsed[1:])]), 3),
            'targets': targets,
        })
    return result


output = {'method': 'Fixed target-center association (60 px gate), then causal majority confirmation of target presence and a causal median over available target ranges. Count truth is two stationary FUEL balls throughout. This tests temporal stability on these clips only; it is not a general tracker or field count benchmark.',
          'limitations': 'Fixed scene and camera; known target centers; no motion, crossings, real occlusion, or metric distance ground truth. Median deviation from a clip median measures repeatability, not absolute accuracy. Majority confirmation can delay newly appearing targets.',
          'profile': {'model': 'FRC 9550 YOLO26n TensorRT 1280', 'confidence': 0.10,
                      'max_box_aspect_ratio': 2.0, 'stereo': 'SGBM block 5 / uniqueness 8'},
          'results': [summarize(path, scene) for scene, path in INPUTS]}
out_path = ROOT / 'media/frc-temporal-fuel-track-sweep.json'
out_path.write_text(json.dumps(output, indent=2) + '\n')
for scene in output['results']:
    print(scene['scene'], 'raw=', scene['raw_target_count_histogram'],
          'extras=', scene['raw_extra_detections'])
    for method in scene['methods']:
        print(' ', method['causal_window_events'], 'count=', method['count_histogram_after_majority_confirmation'],
              'delay_s=', method['max_confirmation_delay_s_at_observed_sampling'],
              'ranges=', [(x['raw_range_mad_m'], x['smoothed_range_mad_m']) for x in method['targets']])
