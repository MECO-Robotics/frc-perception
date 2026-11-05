#!/usr/bin/env python3
"""Compare disparity-aware horizontal crops for FUEL stereo matching."""
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
cv2.setNumThreads(1)
DETECTIONS = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
REFERENCE = json.loads((ROOT / 'media/frc-fuel-dense-stereo-evaluation.json').read_text())
CAL = ROOT / 'media/calibration/stereo_calibration_live_order_candidate.yml'
CONF = .05
BASELINE_M = .0635
SCALE = .5
NUM_DISPARITIES = 128
BLOCK = 5
VERTICAL_PAD = 24
TARGETS = ((808, 589), (902, 591))
CONTEXT = NUM_DISPARITIES + BLOCK

fs = cv2.FileStorage(str(CAL), cv2.FILE_STORAGE_READ)
kl = fs.getNode('left_camera_matrix').mat()
dl = fs.getNode('left_distortion').mat()
kr = fs.getNode('right_camera_matrix').mat()
dr = fs.getNode('right_distortion').mat()
rotation = fs.getNode('rotation').mat()
translation = fs.getNode('translation').mat()
fs.release()
full_size = (1600, 1200)
r1, r2, p1, p2, *_ = cv2.stereoRectify(
    kl, dl, kr, dr, full_size, rotation, translation,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
map_left = cv2.initUndistortRectifyMap(kl, dl, r1, p1, full_size, cv2.CV_32FC1)
map_right = cv2.initUndistortRectifyMap(kr, dr, r2, p2, full_size, cv2.CV_32FC1)
focal = float(p1[0, 0])
matcher = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=NUM_DISPARITIES, blockSize=BLOCK,
    P1=8 * BLOCK * BLOCK, P2=32 * BLOCK * BLOCK, uniquenessRatio=8,
    speckleWindowSize=50, speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(value - center) for value in values]) if values else None


def summary(values):
    if not values:
        return None
    ordered = sorted(values)
    return {'n': len(ordered), 'median_m': round(median(ordered), 4),
            'mad_m': round(mad(ordered), 4),
            'p05_m': round(ordered[int(.05 * (len(ordered) - 1))], 4),
            'p95_m': round(ordered[int(.95 * (len(ordered) - 1))], 4)}


def matched(boxes):
    targets = [None, None]
    for box in sorted((b for b in boxes if b['confidence'] >= CONF),
                      key=lambda b: b['confidence'], reverse=True):
        cx = (box['x1'] + box['x2']) / 2
        cy = (box['y1'] + box['y2']) / 2
        distance, index = min((math.hypot(cx - x, cy - y), i)
                              for i, (x, y) in enumerate(TARGETS))
        if distance <= 60 and targets[index] is None:
            targets[index] = box
    return targets


def rectified_box(box):
    corners = np.array([[[box['x1'], box['y1']]], [[box['x2'], box['y1']]],
                         [[box['x2'], box['y2']]], [[box['x1'], box['y2']]]],
                       dtype=np.float32)
    points = cv2.undistortPoints(corners, kl, dl, R=r1, P=p1).reshape(-1, 2) * SCALE
    return (float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max()))


def disparity_range(disparity, box, x_offset, y_offset):
    x1, y1, x2, y2 = box
    rx1 = max(0, int(np.floor(x1)) - x_offset)
    rx2 = min(disparity.shape[1], int(np.ceil(x2)) - x_offset)
    ry1 = max(0, int(np.floor(y1)) - y_offset)
    ry2 = min(disparity.shape[0], int(np.ceil(y2)) - y_offset)
    roi = disparity[ry1:ry2, rx1:rx2]
    valid = roi[np.isfinite(roi) & (roi > 3 * SCALE) & (roi < 120 * SCALE)]
    if valid.size < 8:
        return None
    half_disp = float(np.median(valid))
    return BASELINE_M * focal / (half_disp / SCALE)


def unproject(box, z):
    cx, cy = (box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2
    center = np.array([[[cx, cy]]], dtype=np.float32)
    uv = cv2.undistortPoints(center, kl, dl, R=r1, P=p1)[0, 0]
    return ((uv[0] - p1[0, 2]) * z / p1[0, 0],
            (uv[1] - p1[1, 2]) * z / p1[1, 1], z)


config_specs = (
    ('full_width_band', None),
    ('box_union_with_128px_search_context', 'disparity_context'))
results = []
for clip_index, clip in enumerate(DETECTIONS['results']):
    capture = cv2.VideoCapture(str(ROOT / clip['video']))
    accumulators = {name: {'ranges': [[], []], 'range_delta': [[], []],
                           'z_gap': [], 'xyz_spacing': [], 'latency_ms': [],
                           'crop_pixels': [], 'valid_pairs': 0}
                   for name, _ in config_specs}
    frames = 0
    for row, ref_row in zip(clip['per_frame'], REFERENCE['results'][clip_index]['frames']):
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        candidates = [box for box in row['detections'] if box['confidence'] >= CONF]
        targets = matched(candidates)
        if not all(targets):
            continue
        left = cv2.cvtColor(cv2.remap(frame[:, :1600], *map_left, cv2.INTER_LINEAR),
                            cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(cv2.remap(frame[:, 1600:], *map_right, cv2.INTER_LINEAR),
                             cv2.COLOR_BGR2GRAY)
        left = cv2.resize(left, (800, 600), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (800, 600), interpolation=cv2.INTER_AREA)
        # Use all accepted candidates for band/crop bounds, just as runtime does.
        rect_candidates = [rectified_box(box) for box in candidates]
        rect_targets = [rectified_box(box) for box in targets]
        y0 = max(0, int(np.floor(min(box[1] for box in rect_candidates))) - VERTICAL_PAD)
        y1 = min(600, int(np.ceil(max(box[3] for box in rect_candidates))) + VERTICAL_PAD)
        for name, crop_mode in config_specs:
            if crop_mode is None:
                x0, x1 = 0, 800
            else:
                leftmost = min(box[0] for box in rect_candidates)
                rightmost = max(box[2] for box in rect_candidates)
                x0 = max(0, int(np.floor(leftmost)) - CONTEXT)
                x1 = min(800, int(np.ceil(rightmost)) + BLOCK)
            lroi = left[y0:y1, x0:x1]
            rroi = right[y0:y1, x0:x1]
            if lroi.shape[1] <= NUM_DISPARITIES + BLOCK or lroi.shape[0] < BLOCK:
                continue
            started = time.perf_counter()
            disparity = matcher.compute(lroi, rroi).astype(np.float32) / 16.0
            record = accumulators[name]
            record['latency_ms'].append((time.perf_counter() - started) * 1000)
            record['crop_pixels'].append(lroi.shape[0] * lroi.shape[1])
            ranges = [disparity_range(disparity, box, x0, y0) for box in rect_targets]
            for target, value in enumerate(ranges):
                if value is not None:
                    record['ranges'][target].append(value)
                    reference = ref_row['ranges_m'][target]
                    if reference is not None:
                        record['range_delta'][target].append(value - reference)
            if all(value is not None for value in ranges):
                record['valid_pairs'] += 1
                points = [unproject(box, z) for box, z in zip(targets, ranges)]
                record['z_gap'].append(abs(ranges[0] - ranges[1]))
                record['xyz_spacing'].append(math.dist(*points))
    capture.release()
    configs = []
    for name, _ in config_specs:
        record = accumulators[name]
        latency = record['latency_ms']
        configs.append({
            'method': name, 'valid_pair_frames': record['valid_pairs'],
            'range_by_target_m': [summary(values) for values in record['ranges']],
            'range_delta_vs_reference_m': [summary(values) for values in record['range_delta']],
            'absolute_z_gap_m': summary(record['z_gap']),
            'xyz_pair_spacing_m': summary(record['xyz_spacing']),
            'matcher_ms_median': round(median(latency), 2) if latency else None,
            'matcher_ms_p95': round(sorted(latency)[int(.95 * (len(latency) - 1))], 2)
            if latency else None,
            'median_crop_pixels': round(median(record['crop_pixels']))
            if record['crop_pixels'] else None})
        print(clip['scene'], configs[-1])
    results.append({'scene': clip['scene'], 'frames': frames, 'configs': configs})

output = {
    'method': 'Compare current full-width vertical-band SGBM with a horizontal crop spanning all confidence-0.05 detection boxes plus 128 half-resolution pixels to the left (the complete disparity search) and a block-size margin to the right. Both configs use the same rectification, 24 px vertical context, 128 disparities, block 5, uniqueness 8, and target box aggregation.',
    'confidence': CONF, 'match_scale': SCALE,
    'num_disparities_half_px': NUM_DISPARITIES, 'block_size': BLOCK,
    'vertical_context_half_px': VERTICAL_PAD,
    'horizontal_search_context_half_px': CONTEXT,
    'baseline_m': BASELINE_M,
    'limitations': 'Two stationary balls at one pose/range; comparison measures matcher speed and repeatability, not absolute distance accuracy. Cropping assumes all useful detections are included in the confidence-0.05 boxes; close-range and edge-of-image coverage need separate checks.',
    'results': results}
path = ROOT / 'media/frc-fuel-stereo-horizontal-roi-sweep.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print('saved', path)
