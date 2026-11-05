#!/usr/bin/env python3
"""Check SGBM settings on exact-shift stereo pairs built from saved FUEL clips."""
import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'media/frc-fuel-known-disparity-synthetic-evaluation.json'
VIDEOS = (
    ('earlier', ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr.mp4'),
    ('corrected-settings', ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'),
)
TARGETS = ((404, 294), (451, 296))  # half-resolution left-image centers
TARGET_RADIUS_PX = 18
SHIFTS_HALF_PX = (18, 22, 26, 30, 18.25, 22.5, 26.75, 30.5)
BASELINE_M = 0.0635
FOCAL_PX = 1094.9511480835965
CONFIGS = (
    {'name': 'production_b5_u8_d128', 'block_size': 5,
     'uniqueness_ratio': 8, 'num_disparities': 128},
    {'name': 'block3_u8_d128', 'block_size': 3,
     'uniqueness_ratio': 8, 'num_disparities': 128},
    {'name': 'block7_u8_d128', 'block_size': 7,
     'uniqueness_ratio': 8, 'num_disparities': 128},
    {'name': 'block5_u5_d128', 'block_size': 5,
     'uniqueness_ratio': 5, 'num_disparities': 128},
    {'name': 'block5_u12_d128', 'block_size': 5,
     'uniqueness_ratio': 12, 'num_disparities': 128},
    {'name': 'block5_u8_d64', 'block_size': 5,
     'uniqueness_ratio': 8, 'num_disparities': 64},
    {'name': 'block5_u8_d96', 'block_size': 5,
     'uniqueness_ratio': 8, 'num_disparities': 96},
)


def load_frame(path):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_MSEC, 3000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f'Could not read 3 s frame from {path}')
    left = frame[:, :1600]
    return cv2.cvtColor(cv2.resize(left, (800, 600)), cv2.COLOR_BGR2GRAY)


def summarize(values):
    if not values:
        return None
    return {
        'n': len(values),
        'median': round(statistics.median(values), 5),
        'p95': round(float(np.percentile(values, 95)), 5),
        'max': round(max(values), 5),
    }


frames = {name: load_frame(path) for name, path in VIDEOS}
yy, xx = np.ogrid[:600, :800]
masks = [((xx - x) ** 2 + (yy - y) ** 2) <= TARGET_RADIUS_PX ** 2
         for x, y in TARGETS]
records = []
for config in CONFIGS:
    errors = []
    range_errors = []
    valid_fractions = []
    for scene, left in frames.items():
        for shift in SHIFTS_HALF_PX:
            right = cv2.warpAffine(
                left, np.float32(((1, 0, -shift), (0, 1, 0))),
                (left.shape[1], left.shape[0]), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT)
            matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=config['num_disparities'],
                blockSize=config['block_size'],
                P1=8 * config['block_size'] ** 2,
                P2=32 * config['block_size'] ** 2,
                uniquenessRatio=config['uniqueness_ratio'],
                speckleWindowSize=50,
                speckleRange=2,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
            disparity = matcher.compute(left, right).astype(np.float32) / 16.0
            expected_range = BASELINE_M * FOCAL_PX / (2 * shift)
            for target_i, mask in enumerate(masks):
                samples = disparity[mask]
                valid = samples[np.isfinite(samples) & (samples > 0)]
                valid_fractions.append(float(valid.size / samples.size))
                if valid.size < 8:
                    median = None
                    disparity_error = None
                    range_error = None
                else:
                    median = float(np.median(valid))
                    disparity_error = abs(median - shift)
                    estimated_range = BASELINE_M * FOCAL_PX / (2 * median)
                    range_error = abs(estimated_range - expected_range)
                    errors.append(disparity_error)
                    range_errors.append(range_error)
                records.append({
                    'config': config['name'],
                    'scene': scene,
                    'target': target_i + 1,
                    'known_disparity_half_px': shift,
                    'known_range_m': round(expected_range, 5),
                    'estimated_disparity_half_px': (
                        round(median, 5) if median is not None else None),
                    'absolute_disparity_error_half_px': (
                        round(disparity_error, 5)
                        if disparity_error is not None else None),
                    'absolute_range_error_m': (
                        round(range_error, 5) if range_error is not None else None),
                    'valid_fraction_in_target_disk': round(
                        float(valid.size / samples.size), 4),
                })
    config['summary'] = {
        'absolute_disparity_error_half_px': summarize(errors),
        'absolute_range_error_m': summarize(range_errors),
        'valid_fraction_in_target_disk_p10': round(
            float(np.percentile(valid_fractions, 10)), 4),
        'evaluated_samples': len(errors),
    }

out = {
    'method': ('For 3 s frames from both saved videos, construct a synthetic '
               'rectified right image by shifting the left grayscale image '
               'left by known integer and fractional disparities. Compare production SGBM '
               'with block-size, uniqueness, and search-width variants in '
               'central target disks.'),
    'limitations': ('This exact-shift planar-texture check validates matcher '
                    'disparity response and scale, not real stereo geometry, '
                    'sphere parallax, physical occlusion, or absolute range. '
                    'The two clips contain the same stationary setup.'),
    'baseline_m': BASELINE_M,
    'focal_px': FOCAL_PX,
    'known_disparities_half_px': list(SHIFTS_HALF_PX),
    'target_disk_radius_half_px': TARGET_RADIUS_PX,
    'configurations': CONFIGS,
    'samples': records,
}
OUTPUT.write_text(json.dumps(out, indent=2) + '\n')
for config in CONFIGS:
    summary = config['summary']
    print(config['name'],
          'disp_abs_err_p95_half_px=',
          summary['absolute_disparity_error_half_px']['p95'],
          'range_abs_err_p95_m=', summary['absolute_range_error_m']['p95'],
          'valid_p10=', summary['valid_fraction_in_target_disk_p10'])
print('saved', OUTPUT)
