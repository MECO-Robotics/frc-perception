#!/usr/bin/env python3
"""Stress stereo ROI disparity with synthetic foreground-mask parallax."""
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'media/frc-fuel-occlusion-parallax-synthetic-evaluation.json'
VIDEOS = (
    ('earlier', ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr.mp4'),
    ('corrected-settings', ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'),
)
TARGETS = ((404, 294, 22.5), (451, 296, 24.5))
RADIUS = 18
OCCLUSION_FRACTIONS = (0.0, 0.25, 0.50, 0.75)
OCCLUDER_DISPARITY_OFFSETS = (0.0, 4.0, 8.0)
BASELINE_M = 0.0635
FOCAL_PX = 1094.9511480835965
CONFIGS = (
    {'name': 'active_b5_u8_d64', 'block_size': 5,
     'uniqueness_ratio': 8, 'num_disparities': 64},
    {'name': 'b3_u8_d64', 'block_size': 3,
     'uniqueness_ratio': 8, 'num_disparities': 64},
    {'name': 'b7_u8_d64', 'block_size': 7,
     'uniqueness_ratio': 8, 'num_disparities': 64},
    {'name': 'b9_u8_d64', 'block_size': 9,
     'uniqueness_ratio': 8, 'num_disparities': 64},
    {'name': 'b11_u8_d64', 'block_size': 11,
     'uniqueness_ratio': 8, 'num_disparities': 64},
    {'name': 'b15_u8_d64', 'block_size': 15,
     'uniqueness_ratio': 8, 'num_disparities': 64},
    {'name': 'b5_u5_d64', 'block_size': 5,
     'uniqueness_ratio': 5, 'num_disparities': 64},
    {'name': 'b5_u12_d64', 'block_size': 5,
     'uniqueness_ratio': 12, 'num_disparities': 64},
    {'name': 'b5_u16_d64', 'block_size': 5,
     'uniqueness_ratio': 16, 'num_disparities': 64},
    {'name': 'b5_u20_d64', 'block_size': 5,
     'uniqueness_ratio': 20, 'num_disparities': 64},
)


def load_gray(path):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_MSEC, 3000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f'Could not read saved frame: {path}')
    return cv2.cvtColor(cv2.resize(frame[:, :1600], (800, 600)),
                        cv2.COLOR_BGR2GRAY)


def shifted(image, disparity, interpolation):
    return cv2.warpAffine(
        image, np.float32(((1, 0, -disparity), (0, 1, 0))),
        (image.shape[1], image.shape[0]), flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT)


def stats(values):
    return {
        'n': len(values),
        'median': round(float(statistics.median(values)), 5) if values else None,
        'p95': round(float(np.percentile(values, 95)), 5) if values else None,
    }


records = []
for scene, path in VIDEOS:
    image = load_gray(path)
    for target_i, (cx, cy, object_disparity) in enumerate(TARGETS):
        # Extra left context lets the matcher search at the object disparity.
        x0, x1 = cx - 100, cx + 100
        y0, y1 = cy - 55, cy + 55
        left_base = image[y0:y1, x0:x1]
        local_cx, local_cy = cx - x0, cy - y0
        disk = ((np.indices(left_base.shape)[1] - local_cx) ** 2 +
                (np.indices(left_base.shape)[0] - local_cy) ** 2 <= RADIUS ** 2)
        object_range = BASELINE_M * FOCAL_PX / (2 * object_disparity)
        for fraction in OCCLUSION_FRACTIONS:
            offsets = (0.0,) if fraction == 0 else OCCLUDER_DISPARITY_OFFSETS
            cutoff = local_cy + RADIUS - 2 * RADIUS * fraction
            left_mask = np.zeros_like(left_base, dtype=np.uint8)
            if fraction:
                cv2.rectangle(left_mask,
                              (local_cx - RADIUS, int(round(cutoff))),
                              (local_cx + RADIUS, local_cy + RADIUS), 255, -1)
            left = left_base.copy()
            left[left_mask > 0] = 0
            for offset in offsets:
                right_base = shifted(left_base, object_disparity,
                                     cv2.INTER_LINEAR)
                right_mask = shifted(left_mask, object_disparity + offset,
                                     cv2.INTER_NEAREST)
                right = right_base.copy()
                right[right_mask > 0] = 0
                # Pixels outside either camera's foreground mask correspond to
                # the object surface in both views; exclude the block boundary.
                visible = disk & (left_mask == 0)
                for ly, lx in zip(*np.nonzero(visible)):
                    rx = int(round(lx - object_disparity))
                    if rx < 0 or rx >= right.shape[1] or right_mask[ly, rx] > 0:
                        visible[ly, lx] = False
                if fraction:
                    visible &= np.indices(left_base.shape)[0] < int(round(cutoff)) - 4
                for config in CONFIGS:
                    block = config['block_size']
                    matcher_args = dict(
                        numDisparities=config['num_disparities'],
                        blockSize=block,
                        P1=8 * block * block,
                        P2=32 * block * block,
                        uniquenessRatio=config['uniqueness_ratio'],
                        speckleWindowSize=50,
                        speckleRange=2,
                        disp12MaxDiff=-1,
                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
                    matcher = cv2.StereoSGBM_create(
                        minDisparity=0, **matcher_args)
                    reverse_matcher = cv2.StereoSGBM_create(
                        minDisparity=-config['num_disparities'], **matcher_args)
                    disparity = matcher.compute(left, right).astype(np.float32) / 16
                    reverse = reverse_matcher.compute(right, left).astype(np.float32) / 16
                    raw = disparity[disk]
                    raw = raw[np.isfinite(raw) & (raw > 0)]
                    clean = disparity[visible]
                    clean = clean[np.isfinite(clean) & (clean > 0)]
                    raw_median = float(np.median(raw)) if raw.size >= 8 else None
                    clean_median = float(np.median(clean)) if clean.size >= 8 else None
                    mad = (float(np.median(np.abs(raw - raw_median)))
                           if raw_median is not None else None)
                    ys, xs = np.nonzero(disk & np.isfinite(disparity) &
                                        (disparity > 0))
                    lr_errors = []
                    for ly, lx in zip(ys, xs):
                        rx = int(round(lx - float(disparity[ly, lx])))
                        if 0 <= rx < reverse.shape[1]:
                            reverse_value = float(reverse[ly, rx])
                            if reverse_value < 0:
                                lr_errors.append(abs(float(disparity[ly, lx]) +
                                                     reverse_value))
                    lr_fraction = (sum(error <= 0.58 for error in lr_errors) /
                                   len(lr_errors) if lr_errors else None)
                    estimate = (BASELINE_M * FOCAL_PX / (2 * raw_median)
                                if raw_median else None)
                    records.append({
                        'scene': scene,
                        'target': target_i + 1,
                        'config': config['name'],
                        'object_disparity_half_px': object_disparity,
                        'occlusion_fraction': fraction,
                        'occluder_extra_disparity_half_px': offset,
                        'known_object_range_m': round(object_range, 5),
                        'raw_roi_median_disparity_half_px': (
                            round(raw_median, 5) if raw_median else None),
                        'visible_surface_median_disparity_half_px': (
                            round(clean_median, 5) if clean_median else None),
                        'raw_roi_disparity_mad_half_px': (
                            round(mad, 5) if mad is not None else None),
                        'lr_consistency_fraction_within_0p58_half_px': (
                            round(lr_fraction, 5)
                            if lr_fraction is not None else None),
                        'raw_roi_absolute_range_error_m': (
                            round(abs(estimate - object_range), 5)
                            if estimate is not None else None),
                        'raw_roi_valid_fraction': round(float(raw.size / disk.sum()), 4),
                        'visible_surface_valid_fraction': round(
                            float(clean.size / max(1, visible.sum())), 4),
                        'mad_warning_gt_1p25_half_px': mad is not None and mad > 1.25,
                        'lr_warning_fraction_below_0p58': (
                            lr_fraction is None or lr_fraction < 0.58),
                    })

summaries = []
for config in CONFIGS:
    for fraction in OCCLUSION_FRACTIONS:
        rows = [r for r in records if r['config'] == config['name'] and
                r['occlusion_fraction'] == fraction]
        for offset in sorted(set(r['occluder_extra_disparity_half_px'] for r in rows)):
            group = [r for r in rows
                     if r['occluder_extra_disparity_half_px'] == offset]
            errors = [r['raw_roi_absolute_range_error_m'] for r in group
                      if r['raw_roi_absolute_range_error_m'] is not None]
            summaries.append({
                'config': config['name'],
                'occlusion_fraction': fraction,
                'occluder_extra_disparity_half_px': offset,
                'samples': len(group),
                'absolute_range_error_m': stats(errors),
                'mad_warnings': sum(r['mad_warning_gt_1p25_half_px'] for r in group),
                'lr_warnings': sum(r['lr_warning_fraction_below_0p58']
                                   for r in group),
                'valid_fraction_median': round(statistics.median(
                    r['raw_roi_valid_fraction'] for r in group), 4),
            })

out = {
    'method': ('Build rectified synthetic stereo pairs from 3 s left-eye '
               'textures in both saved videos, with a known disparity for '
               'each target. Mask a lower target region in both views, then '
               'shift the right-view mask by the target disparity plus a '
               'foreground disparity offset to simulate occlusion parallax.'),
    'limitations': ('The mask is a black rectangle and the object is modeled '
                    'as a plane with one disparity. This is a controlled '
                    'matcher stress test, not physical foreground occlusion '
                    'or spherical FUEL geometry.'),
    'baseline_m': BASELINE_M,
    'focal_px': FOCAL_PX,
    'configurations': CONFIGS,
    'summaries': summaries,
    'samples': records,
}
OUTPUT.write_text(json.dumps(out, indent=2) + '\n')
for row in summaries:
    if row['config'] == 'active_b5_u8_d64':
        print(row)
print('saved', OUTPUT)
