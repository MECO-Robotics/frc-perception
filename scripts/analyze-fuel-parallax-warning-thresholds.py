#!/usr/bin/env python3
"""Compare LR warning thresholds on clean video and synthetic bad ranges."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / 'media/frc-fuel-clean-stereo-quality-evaluation.json'
STRESS_PATH = ROOT / 'media/frc-fuel-occlusion-parallax-synthetic-evaluation.json'
OUTPUT = ROOT / 'media/frc-fuel-parallax-warning-thresholds.json'
LR_THRESHOLDS = (0.58, 0.80, 0.85, 0.90, 0.95, 0.98)
MAD_THRESHOLD = 1.25
BAD_RANGE_THRESHOLD_M = 0.05

clean_data = json.loads(CLEAN_PATH.read_text())
stress_data = json.loads(STRESS_PATH.read_text())
clean = [target for clip in clean_data['clips']
         for row in clip['per_sample'] for target in row['targets']]
stress = [row for row in stress_data['samples']
          if row['occlusion_fraction'] == 0.75 and
          row['occluder_extra_disparity_half_px'] >= 4]
bad = [row for row in stress
       if row['raw_roi_absolute_range_error_m'] is not None and
       row['raw_roi_absolute_range_error_m'] > BAD_RANGE_THRESHOLD_M]

results = []
for threshold in LR_THRESHOLDS:
    clean_rejected = sum(
        row['mad_px'] > MAD_THRESHOLD or
        row['lr_consistency_fraction'] < threshold
        for row in clean)
    bad_flagged = sum(
        row['raw_roi_disparity_mad_half_px'] is not None and
        (row['raw_roi_disparity_mad_half_px'] > MAD_THRESHOLD or
         row['lr_consistency_fraction_within_0p58_half_px'] is None or
         row['lr_consistency_fraction_within_0p58_half_px'] < threshold)
        for row in bad)
    results.append({
        'lr_consistency_threshold': threshold,
        'clean_rejected': clean_rejected,
        'clean_samples': len(clean),
        'bad_ranges_flagged': bad_flagged,
        'bad_ranges': len(bad),
    })

out = {
    'method': ('Hold MAD >1.25 px fixed and sweep the left-right consistency '
               'warning threshold. Compare real unmasked saved-video samples '
               'with synthetic 75%-occlusion samples whose range error exceeds 5 cm.'),
    'clean_source': str(CLEAN_PATH.relative_to(ROOT)),
    'stress_source': str(STRESS_PATH.relative_to(ROOT)),
    'mad_threshold_half_px': MAD_THRESHOLD,
    'bad_range_threshold_m': BAD_RANGE_THRESHOLD_M,
    'limitations': ('Synthetic stress and real clean samples use different '
                    'ROI construction and do not model physical foreground '
                    'occlusion. This threshold sweep is diagnostic only and '
                    'does not validate a production quality gate.'),
    'results': results,
}
OUTPUT.write_text(json.dumps(out, indent=2) + '\n')
for row in results:
    print(row)
print('saved', OUTPUT)
