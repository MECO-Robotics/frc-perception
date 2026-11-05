#!/usr/bin/env python3
"""Check size-vs-stereo gate thresholds against clean and parallax stress data."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS_PATH = ROOT / 'media/frc-fuel-dense-count-nms015.json'
STEREO_PATH = ROOT / 'media/frc-fuel-dense-stereo-nms015-conf005.json'
STRESS_PATH = ROOT / 'media/frc-fuel-occlusion-parallax-synthetic-evaluation.json'
OUTPUT = ROOT / 'media/frc-fuel-size-gate-parallax-thresholds.json'
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.55)
FRAME = 180  # 3 s, matching the texture used for the synthetic stress pairs
FOCAL_RAW_PX = 1046.1599129444185
FOCAL_RECTIFIED_PX = 1094.9511480835965
FUEL_DIAMETER_M = 0.15
BASELINE_M = 0.0635
BAD_RANGE_ERROR_M = 0.05
TARGETS = ((808, 589), (902, 591))

detection_data = json.loads(DETECTIONS_PATH.read_text())
stereo_data = json.loads(STEREO_PATH.read_text())
stress_data = json.loads(STRESS_PATH.read_text())

clean = []

size_range_by_target = {}
for clip in detection_data['results']:
    row = next(frame for frame in clip['per_frame'] if frame['frame'] == FRAME)
    detections = [item for item in row['detections'] if item['confidence'] >= 0.05]
    chosen = [None, None]
    for item in sorted(detections, key=lambda item: -item['confidence']):
        cx = (item['x1'] + item['x2']) / 2
        cy = (item['y1'] + item['y2']) / 2
        distance, target = min((math.hypot(cx - x, cy - y), i)
                               for i, (x, y) in enumerate(TARGETS))
        if distance <= 60 and chosen[target] is None:
            chosen[target] = item
    for target, item in enumerate(chosen, 1):
        if item is None:
            raise RuntimeError(f'Missing target {target} in {clip["scene"]}')
        width = item['x2'] - item['x1']
        height = item['y2'] - item['y1']
        size_range_by_target[(clip['scene'], target)] = (
            FOCAL_RAW_PX * FUEL_DIAMETER_M / math.sqrt(width * height))

for detection_clip, stereo_clip in zip(detection_data['results'],
                                        stereo_data['results']):
    for frame, stereo_frame in zip(detection_clip['per_frame'],
                                   stereo_clip['frames']):
        chosen = [None, None]
        detections = [item for item in frame['detections']
                      if item['confidence'] >= 0.05]
        for item in sorted(detections, key=lambda item: -item['confidence']):
            cx = (item['x1'] + item['x2']) / 2
            cy = (item['y1'] + item['y2']) / 2
            distance, target = min((math.hypot(cx - x, cy - y), i)
                                   for i, (x, y) in enumerate(TARGETS))
            if distance <= 60 and chosen[target] is None:
                chosen[target] = item
        for target, item in enumerate(chosen):
            stereo_range = stereo_frame['ranges_m'][target]
            if item is None or not stereo_range:
                continue
            width = item['x2'] - item['x1']
            height = item['y2'] - item['y1']
            size_range = FOCAL_RAW_PX * FUEL_DIAMETER_M / math.sqrt(width * height)
            clean.append(abs(size_range - stereo_range) / stereo_range)

synthetic = []
for row in stress_data['samples']:
    if (row['occlusion_fraction'] != 0.75 or
            row['occluder_extra_disparity_half_px'] < 4 or
            row['raw_roi_median_disparity_half_px'] is None):
        continue
    estimated = (BASELINE_M * FOCAL_RECTIFIED_PX /
                 (2 * row['raw_roi_median_disparity_half_px']))
    size_range = size_range_by_target[(row['scene'], row['target'])]
    relative_error = abs(size_range - estimated) / estimated
    range_error = abs(estimated - row['known_object_range_m'])
    synthetic.append({
        'scene': row['scene'],
        'target': row['target'],
        'config': row['config'],
        'occluder_extra_disparity_half_px':
            row['occluder_extra_disparity_half_px'],
        'stereo_range_m': estimated,
        'size_proxy_range_m': size_range,
        'relative_size_stereo_error': relative_error,
        'absolute_range_error_m': range_error,
        'bad_range_over_5cm': range_error > BAD_RANGE_ERROR_M,
    })

results = []
for threshold in THRESHOLDS:
    clean_rejects = sum(error > threshold for error in clean)
    bad_ranges = [row for row in synthetic if row['bad_range_over_5cm']]
    caught = sum(row['relative_size_stereo_error'] > threshold
                 for row in bad_ranges)
    results.append({
        'max_relative_error': threshold,
        'clean_rejected': clean_rejects,
        'clean_samples': len(clean),
        'bad_ranges_caught': caught,
        'bad_ranges': len(bad_ranges),
    })

out = {
    'method': ('Sweep the runtime FUEL apparent-size/stereo relative-error '
               'gate. Clean errors come from unmasked saved-video samples; '
               'stress errors compare controlled-parallax synthetic stereo '
               'ranges with the known synthetic disparity.'),
    'clean_sources': [str(DETECTIONS_PATH.relative_to(ROOT)),
                      str(STEREO_PATH.relative_to(ROOT))],
    'stress_source': str(STRESS_PATH.relative_to(ROOT)),
    'runtime_relative_error_definition': 'abs(size_range - stereo_range) / stereo_range',
    'known_range_error_threshold_m': BAD_RANGE_ERROR_M,
    'limitations': ('The clean data contain only two static targets; stress '
                    'samples use synthetic masks and a frame-180 size proxy '
                    'per target. This does not establish a field-safe gate '
                    'or protect against real size variation and occlusion.'),
    'results': results,
    'synthetic_bad_range_samples': synthetic,
}
OUTPUT.write_text(json.dumps(out, indent=2) + '\n')
for row in results:
    print(row)
print('saved', OUTPUT)
