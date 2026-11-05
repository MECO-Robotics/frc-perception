#!/usr/bin/env python3
"""Check whether one baseline multiplier aligns stereo with FUEL-size range."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'media/frc-fuel-known-diameter-range-evaluation.json'
OUTPUT = ROOT / 'media/frc-fuel-baseline-scale-consistency.json'
BASELINE_M = 0.0635
HOUGH_PARAM2 = 25

data = json.loads(SOURCE.read_text())
comparisons = []
for clip in data['results']:
    sweep = next(row for row in clip['param2_sweep']
                 if row['param2'] == HOUGH_PARAM2)
    for target in sweep['targets']:
        size_range = target['circle_range_m']['median_m']
        stereo_range = target['stereo_range_m']['median_m']
        scale = size_range / stereo_range
        comparisons.append({
            'scene': clip['scene'],
            'target': target['target'],
            'nominal_diameter_range_m': size_range,
            'stereo_range_m': stereo_range,
            'required_baseline_scale': scale,
            'required_baseline_m': BASELINE_M * scale,
            'size_proxy_mad_m': target['circle_range_m']['mad_m'],
        })

common_scale = statistics.median(
    item['required_baseline_scale'] for item in comparisons)
for item in comparisons:
    item['range_after_common_scale_m'] = item['stereo_range_m'] * common_scale
    item['residual_to_size_proxy_after_common_scale_m'] = (
        item['range_after_common_scale_m'] - item['nominal_diameter_range_m'])

out = {
    'method': ('Compare per-target median SGBM range with detector-centered '
               'Hough nominal-diameter range from the same saved videos. '
               'Compute the per-target baseline multiplier needed to align '
               'the medians, then apply one robust common multiplier.'),
    'source': str(SOURCE.relative_to(ROOT)),
    'hough_param2': HOUGH_PARAM2,
    'configured_baseline_m': BASELINE_M,
    'common_median_scale': common_scale,
    'common_scaled_baseline_m': BASELINE_M * common_scale,
    'comparisons': comparisons,
    'limitations': ('Nominal FUEL diameter is only a geometry proxy; neither '
                    'range source has tape-measured truth. This test can '
                    'reject a single scale as a consistent explanation, '
                    'but cannot calibrate absolute distance.'),
}
OUTPUT.write_text(json.dumps(out, indent=2) + '\n')
for item in comparisons:
    print(item['scene'], f"target{item['target']}",
          f"required_scale={item['required_baseline_scale']:.4f}",
          f"common_scale_residual_m={item['residual_to_size_proxy_after_common_scale_m']:+.3f}")
print(f'common_scale={common_scale:.4f}',
      f'baseline_m={BASELINE_M * common_scale:.5f}')
print('saved', OUTPUT)
