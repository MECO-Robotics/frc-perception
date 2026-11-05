#!/usr/bin/env python3
"""Compare SGBM block sizes under matched synthetic partial occlusion."""
import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COUNTS = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
OUTPUT = ROOT / 'media/frc-fuel-stereo-block-occlusion-evaluation.json'
CALIBRATION = ROOT / 'media/calibration/stereo_calibration_live_order_candidate.yml'
TARGETS = ((808, 589), (902, 591))
TIMES = tuple(1.0 + 0.4 * i for i in range(15))
OCCLUSIONS = (0.0, 0.25, 0.5, 0.75)
BLOCKS = (3, 5)
PAD = 24
SCALE = 0.5
BASELINE_M = 0.0635
MATCHERS = {}

fs = cv2.FileStorage(str(CALIBRATION), cv2.FILE_STORAGE_READ)
kl, dl = fs.getNode('left_camera_matrix').mat(), fs.getNode('left_distortion').mat()
kr, dr = fs.getNode('right_camera_matrix').mat(), fs.getNode('right_distortion').mat()
rotation, translation = fs.getNode('rotation').mat(), fs.getNode('translation').mat()
fs.release()
size = (1600, 1200)
r1, r2, p1, p2, *_ = cv2.stereoRectify(
    kl, dl, kr, dr, size, rotation, translation,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
map_left = cv2.initUndistortRectifyMap(kl, dl, r1, p1, size, cv2.CV_32FC1)
map_right = cv2.initUndistortRectifyMap(kr, dr, r2, p2, size, cv2.CV_32FC1)
focal = float(p1[0, 0])

for block in BLOCKS:
    MATCHERS[block] = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=block,
        P1=8 * block * block, P2=32 * block * block,
        uniquenessRatio=8, speckleWindowSize=50, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def matched(row):
    result = [None, None]
    for box in sorted((b for b in row['detections'] if b['confidence'] >= 0.1),
                      key=lambda b: -b['confidence']):
        cx, cy = (box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2
        distances = [math.hypot(cx - x, cy - y) for x, y in TARGETS]
        target = min(range(2), key=distances.__getitem__)
        if distances[target] <= 60 and result[target] is None:
            result[target] = box
    return result


def rect_box(box):
    corners = np.array([[[box['x1'], box['y1']], [box['x2'], box['y1']],
                         [box['x2'], box['y2']], [box['x1'], box['y2']]]],
                       dtype=np.float32).reshape(-1, 1, 2)
    points = cv2.undistortPoints(corners, kl, dl, R=r1, P=p1).reshape(-1, 2) * SCALE
    return (max(0, int(np.floor(points[:, 0].min()))),
            max(0, int(np.floor(points[:, 1].min()))),
            min(800, int(np.ceil(points[:, 0].max()))),
            min(600, int(np.ceil(points[:, 1].max()))))


def disparity_range(disparity, bounds, y_offset):
    x1, y1, x2, y2 = bounds
    roi = disparity[max(0, y1 - y_offset):max(0, y2 - y_offset), x1:x2]
    valid = roi[np.isfinite(roi) & (roi > 3.0 * SCALE) & (roi < 120.0 * SCALE)]
    if valid.size < 8:
        return None, 0.0
    dpx = float(np.median(valid) / SCALE)
    return BASELINE_M * focal / dpx, float(valid.size / max(1, roi.size))


def rectified_images(left, right):
    l = cv2.remap(left, *map_left, cv2.INTER_LINEAR)
    r = cv2.remap(right, *map_right, cv2.INTER_LINEAR)
    return (cv2.resize(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY), (800, 600), interpolation=cv2.INTER_AREA),
            cv2.resize(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY), (800, 600), interpolation=cv2.INTER_AREA))


def summarize(values):
    values = [float(x) for x in values if x is not None and math.isfinite(x)]
    if not values:
        return {'n': 0}
    median = statistics.median(values)
    return {'n': len(values), 'median_m': round(median, 5),
            'mad_m': round(statistics.median(abs(x - median) for x in values), 5),
            'p95_m': round(float(np.percentile(values, 95)), 5)}


records = {}
for clip in COUNTS['results']:
    capture = cv2.VideoCapture(str(ROOT / clip['video']))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_rows = clip['per_frame']
    for time_s in TIMES:
        index = round(time_s * fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        boxes = matched(frame_rows[index])
        if not all(boxes):
            continue
        left, right = frame[:, :1600].copy(), frame[:, 1600:].copy()
        raw_ranges = {}
        for block in BLOCKS:
            gl, gr = rectified_images(left, right)
            y0 = max(0, min(rect_box(b)[1] for b in boxes) - PAD)
            y1 = min(600, max(rect_box(b)[3] for b in boxes) + PAD)
            matcher = MATCHERS[block]
            clean_disp = matcher.compute(gl[y0:y1], gr[y0:y1]).astype(np.float32) / 16
            clean = [disparity_range(clean_disp, rect_box(b), y0)[0] for b in boxes]
            raw_ranges[block] = clean
            for fraction in OCCLUSIONS:
                masked_left, masked_right = left.copy(), right.copy()
                if fraction:
                    for box, z in zip(boxes, clean):
                        if z is None:
                            continue
                        y_start = max(0, box['y2'] - round((box['y2'] - box['y1']) * fraction))
                        y_end = min(1200, box['y2'])
                        x1, x2 = max(0, box['x1']), min(1600, box['x2'])
                        masked_left[y_start:y_end, x1:x2] = 0
                        disparity_px = BASELINE_M * focal / z
                        rx1 = max(0, round(x1 - disparity_px))
                        rx2 = min(1600, round(x2 - disparity_px))
                        masked_right[y_start:y_end, rx1:rx2] = 0
                ml, mr = rectified_images(masked_left, masked_right)
                disp = matcher.compute(ml[y0:y1], mr[y0:y1]).astype(np.float32) / 16
                ranges, valid_fraction = [], []
                for box in boxes:
                    z, valid = disparity_range(disp, rect_box(box), y0)
                    ranges.append(z)
                    valid_fraction.append(valid)
                pair_spacing = None
                if all(ranges):
                    points = []
                    for box, z in zip(boxes, ranges):
                        center = np.array([[[float(box['x1'] + box['x2']) / 2,
                                             float(box['y1'] + box['y2']) / 2]]], np.float32)
                        u, v = cv2.undistortPoints(center, kl, dl, R=r1, P=p1)[0, 0]
                        points.append(((u - p1[0, 2]) * z / p1[0, 0],
                                       (v - p1[1, 2]) * z / p1[1, 1], z))
                    pair_spacing = math.dist(points[0], points[1])
                key = (clip['scene'], block, fraction)
                records.setdefault(key, {'sampled_frames': 0,
                                         'range_errors': [[], []], 'ranges': [[], []],
                                         'valid': [0, 0], 'pair_spacing': [],
                                         'pair_spacing_error': []})
                row = records[key]
                row['sampled_frames'] += 1
                for i in range(2):
                    if ranges[i] is not None:
                        row['valid'][i] += 1
                        row['ranges'][i].append(ranges[i])
                        if raw_ranges[block][i] is not None:
                            row['range_errors'][i].append(abs(ranges[i] - raw_ranges[block][i]))
                if pair_spacing is not None:
                    row['pair_spacing'].append(pair_spacing)
                    clean_spacing = None
                    if all(raw_ranges[block]):
                        clean_points = []
                        for box, z in zip(boxes, raw_ranges[block]):
                            center = np.array([[[float(box['x1'] + box['x2']) / 2,
                                                 float(box['y1'] + box['y2']) / 2]]], np.float32)
                            u, v = cv2.undistortPoints(center, kl, dl, R=r1, P=p1)[0, 0]
                            clean_points.append(((u - p1[0, 2]) * z / p1[0, 0],
                                                 (v - p1[1, 2]) * z / p1[1, 1], z))
                        clean_spacing = math.dist(clean_points[0], clean_points[1])
                    if clean_spacing is not None:
                        row['pair_spacing_error'].append(abs(pair_spacing - clean_spacing))
    capture.release()

results = []
for (scene, block, fraction), row in sorted(records.items()):
    results.append({
        'scene': scene, 'block_size': block, 'occlusion_fraction': fraction,
        'sampled_frames': row['sampled_frames'],
        'valid_depth_frames_by_target': row['valid'],
        'range_m_by_target': [summarize(x) for x in row['ranges']],
        'absolute_range_shift_from_clean_m_by_target': [summarize(x) for x in row['range_errors']],
        'pair_spacing_m': summarize(row['pair_spacing']),
        'pair_spacing_shift_from_clean_m': summarize(row['pair_spacing_error']),
    })
report = {
    'method': ('On 15 fixed timestamps per clip, keep detector boxes fixed, mask '
               'the lower 0/25/50/75% of each target in both views (right mask '
               'shifted by the same matcher clean range), then compare rectified '
               'SGBM block sizes 3 and 5 using the active 128-disparity, 0.5-scale, '
               '24 px vertical-band ROI and 63.5 mm baseline.'),
    'limitations': ('Synthetic bottom masks are not physical occluder parallax and '
                    'the detector is not rerun after masking. Range error is relative '
                    'to each matcher’s clean same-frame estimate, not ground truth.'),
    'results': results,
}
OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
for row in results:
    if row['occlusion_fraction'] in (0.0, 0.5, 0.75):
        print(row['scene'], 'block', row['block_size'], 'mask', row['occlusion_fraction'],
              'valid', row['valid_depth_frames_by_target'],
              'range shifts', [x.get('median_m') for x in row['absolute_range_shift_from_clean_m_by_target']],
              'pair shift', row['pair_spacing_shift_from_clean_m'].get('median_m'))
print('saved', OUTPUT)
