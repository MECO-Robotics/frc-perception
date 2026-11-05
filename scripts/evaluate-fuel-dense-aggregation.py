#!/usr/bin/env python3
"""Compare dense-frame stereo disparity aggregators on saved FUEL clips."""
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DENSE_PATH = ROOT / 'media/frc-fuel-dense-count-evaluation.json'
CALIBRATION = ROOT / 'media/calibration/stereo_calibration_live_order_candidate.yml'
AGGREGATORS = ('median', 'mean', 'trimmed10', 'trimmed25', 'winsor10', 'mode_halfpx')
BASELINE_M = 0.0635
MIN_DISPARITY_HALF = 3.0
MAX_DISPARITY_HALF = 120.0


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(value - center) for value in values]) if values else None


def summary(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {
        'valid_samples': len(ordered),
        'median': round(median(ordered), 4),
        'mad': round(mad(ordered), 4),
        'p05': round(ordered[int(.05 * (len(ordered) - 1))], 4),
        'p95': round(ordered[int(.95 * (len(ordered) - 1))], 4),
    }


def aggregate(values, method):
    values = np.asarray(values, np.float64)
    ordered = np.sort(values)
    if method == 'median':
        return float(np.median(ordered))
    if method == 'mean':
        return float(np.mean(ordered))
    if method in ('trimmed10', 'trimmed25'):
        fraction = .10 if method == 'trimmed10' else .25
        trim = int(len(ordered) * fraction)
        retained = ordered[trim:len(ordered) - trim] if trim else ordered
        return float(np.mean(retained))
    if method == 'winsor10':
        low, high = np.quantile(ordered, (.10, .90))
        return float(np.mean(np.clip(ordered, low, high)))
    if method == 'mode_halfpx':
        # Bins are 0.25 half-resolution px wide = 0.5 full-resolution px.
        bin_width = .25
        origin = math.floor(float(ordered.min()) / bin_width) * bin_width
        bins = np.floor((ordered - origin) / bin_width).astype(int)
        counts = np.bincount(bins)
        modal = int(np.argmax(counts))
        center = origin + (modal + .5) * bin_width
        neighborhood = ordered[np.abs(ordered - center) <= bin_width]
        return float(np.mean(neighborhood if neighborhood.size else ordered[bins == modal]))
    raise ValueError(method)


def match_targets(items):
    matched = [None, None]
    for box in sorted((item for item in items if item['confidence'] >= .10),
                      key=lambda item: item['confidence'], reverse=True):
        cx, cy = (box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2
        distance, index = min((math.hypot(cx - x, cy - y), i)
                              for i, (x, y) in enumerate(((808, 589), (902, 591))))
        if distance <= 60 and matched[index] is None:
            matched[index] = box
    return matched


dense = json.loads(DENSE_PATH.read_text())
fs = cv2.FileStorage(str(CALIBRATION), cv2.FILE_STORAGE_READ)
kl = fs.getNode('left_camera_matrix').mat()
dl = fs.getNode('left_distortion').mat()
kr = fs.getNode('right_camera_matrix').mat()
dr = fs.getNode('right_distortion').mat()
rotation = fs.getNode('rotation').mat()
translation = fs.getNode('translation').mat()
fs.release()
size = (1600, 1200)
r1, r2, p1, p2, *_ = cv2.stereoRectify(
    kl, dl, kr, dr, size, rotation, translation,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
map_left = cv2.initUndistortRectifyMap(kl, dl, r1, p1, size, cv2.CV_32FC1)
map_right = cv2.initUndistortRectifyMap(kr, dr, r2, p2, size, cv2.CV_32FC1)
matcher = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=128, blockSize=5, P1=200, P2=800,
    uniquenessRatio=8, speckleWindowSize=50, speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
focal_px = float(p1[0, 0])
output = {
    'method': 'Reuse all 1280x960 detector boxes at confidence 0.10. Compute one rectified 800x600 SGBM b5/u8 disparity map per synchronized video frame, then compare six summaries inside identical full detection-box ROIs.',
    'calibration': str(CALIBRATION.relative_to(ROOT)),
    'baseline_m': BASELINE_M,
    'rectified_focal_px': focal_px,
    'limitations': 'Only two mostly stationary visible FUEL balls at about 1.4-1.6 m. No measured metric range or true ball centers. Spread measures repeatability, not absolute accuracy.',
    'results': [],
}

for clip in dense['results']:
    capture = cv2.VideoCapture(str(ROOT / clip['video']))
    records = {method: [[], []] for method in AGGREGATORS}
    pair_rows = {method: {} for method in AGGREGATORS}
    matcher_times = []
    frames = clip['per_frame']
    started = time.perf_counter()
    processed = 0
    for index, row in enumerate(frames):
        ok, frame = capture.read()
        if not ok:
            break
        processed += 1
        boxes = match_targets(row['detections'])
        left, right = frame[:, :1600], frame[:, 1600:]
        gray_left = cv2.cvtColor(cv2.remap(left, *map_left, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(cv2.remap(right, *map_right, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
        gray_left = cv2.resize(gray_left, (800, 600), interpolation=cv2.INTER_AREA)
        gray_right = cv2.resize(gray_right, (800, 600), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        disparity = matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
        matcher_times.append((time.perf_counter() - t0) * 1000)
        target_rows = {method: [None, None] for method in AGGREGATORS}
        for target_index, box in enumerate(boxes):
            if box is None:
                continue
            corners = np.array([[[box['x1'], box['y1']], [box['x2'], box['y1']],
                                 [box['x2'], box['y2']], [box['x1'], box['y2']]]],
                               dtype=np.float32)
            rect = cv2.undistortPoints(corners, kl, dl, R=r1, P=p1).reshape(-1, 2) / 2
            x1, y1 = np.floor(rect.min(axis=0)).astype(int)
            x2, y2 = np.ceil(rect.max(axis=0)).astype(int)
            y1, y2 = max(0, y1), min(disparity.shape[0], y2)
            x1, x2 = max(0, x1), min(disparity.shape[1], x2)
            roi = disparity[y1:y2, x1:x2]
            valid = roi[np.isfinite(roi) & (roi > MIN_DISPARITY_HALF) &
                        (roi < MAX_DISPARITY_HALF)]
            if valid.size < 8:
                continue
            center = np.array([[[ (box['x1'] + box['x2']) / 2,
                                   (box['y1'] + box['y2']) / 2 ]]], dtype=np.float32)
            u, v = cv2.undistortPoints(center, kl, dl, R=r1, P=p1)[0, 0]
            for method in AGGREGATORS:
                disp_half = aggregate(valid, method)
                distance = BASELINE_M * focal_px / (2 * disp_half)
                point = ((u - p1[0, 2]) * distance / p1[0, 0],
                         (v - p1[1, 2]) * distance / p1[1, 1], distance)
                records[method][target_index].append(distance)
                target_rows[method][target_index] = {'point': point,
                    'range_m': distance, 'disparity_half_px': disp_half}
        for method in AGGREGATORS:
            pair = target_rows[method]
            spacing = None
            if all(item is not None for item in pair):
                spacing = math.dist(pair[0]['point'], pair[1]['point'])
            pair_rows[method][index] = {
                'time_s': row['time_s'],
                'ranges_m': [item['range_m'] if item else None for item in pair],
                'pair_spacing_m': spacing,
                'disparities_half_px': [item['disparity_half_px'] if item else None
                                        for item in pair],
            }
    capture.release()
    clip_result = {'scene': clip['scene'], 'video': clip['video'],
                   'frames_processed': processed,
                   'median_sgbm_compute_ms': round(median(matcher_times), 2),
                   'p95_sgbm_compute_ms': round(sorted(matcher_times)[int(.95*(len(matcher_times)-1))], 2),
                   'elapsed_seconds': round(time.perf_counter() - started, 2),
                   'aggregators': []}
    for method in AGGREGATORS:
        spacing = [item['pair_spacing_m'] for item in pair_rows[method].values()
                   if item['pair_spacing_m'] is not None]
        merged = {str(radius): sum(value <= radius for value in spacing)
                  for radius in (.15, .20, .25)}
        clip_result['aggregators'].append({
            'method': method,
            'ranges_by_target_m': [summary(values) for values in records[method]],
            'pair_spacing_m': summary(spacing),
            'paired_frames': len(spacing),
            'merged_pair_frames_at_radius_m': merged,
            'frames': pair_rows[method],
        })
    output['results'].append(clip_result)

out_path = ROOT / 'media/frc-fuel-dense-aggregation.json'
out_path.write_text(json.dumps(output, indent=2) + '\n')
for clip in output['results']:
    print(clip['scene'], 'SGBM ms', clip['median_sgbm_compute_ms'], clip['p95_sgbm_compute_ms'])
    for row in clip['aggregators']:
        print(row['method'], row['ranges_by_target_m'], 'spacing', row['pair_spacing_m'],
              'paired', row['paired_frames'], 'merged', row['merged_pair_frames_at_radius_m'])
print('saved', out_path)
