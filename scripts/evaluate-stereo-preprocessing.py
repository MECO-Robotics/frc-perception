#!/usr/bin/env python3
"""Sweep image preprocessing options while holding FUEL boxes and SGBM fixed."""
import json
import math
import os
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from scout.detectors import TensorRTDetector

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / 'media/calibration/stereo_calibration_live_order_candidate.yml'
ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_1280_fp16.engine'
VIDEOS = ('media/frc-2026-live-two-fuel-60fps-cfr.mp4',
          'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4')
SAMPLE_TIMES = (1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.4, 5.8, 6.2, 6.6)
TARGETS = ((808, 589), (902, 591))
BASELINE_M = 0.0635
FUEL_DIAMETER_M = 0.15
MODES = ('gray', 'clahe2', 'clahe4', 'equalize', 'gaussian3', 'median3', 'gamma08', 'gamma12', 'bilateral')


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(value - center) for value in values]) if values else None


def preprocess(gray, mode):
    if mode == 'gray':
        return gray
    if mode == 'clahe2':
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if mode == 'clahe4':
        return cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
    if mode == 'equalize':
        return cv2.equalizeHist(gray)
    if mode == 'gaussian3':
        return cv2.GaussianBlur(gray, (3, 3), 0.8)
    if mode == 'median3':
        return cv2.medianBlur(gray, 3)
    if mode in ('gamma08', 'gamma12'):
        gamma = 0.8 if mode == 'gamma08' else 1.2
        lut = np.clip(np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255.0,
                      0, 255).astype(np.uint8)
        return cv2.LUT(gray, lut)
    if mode == 'bilateral':
        return cv2.bilateralFilter(gray, 5, 35, 35)
    raise ValueError(mode)


def rectified_roi(box, kl, dl, r1, p1, shape):
    corners = np.array([[[box.x1, box.y1]], [[box.x2, box.y1]],
                        [[box.x2, box.y2]], [[box.x1, box.y2]]], np.float32)
    rect = cv2.undistortPoints(corners, kl, dl, R=r1, P=p1).reshape(-1, 2) / 2.0
    x1, y1 = np.floor(rect.min(axis=0)).astype(int)
    x2, y2 = np.ceil(rect.max(axis=0)).astype(int)
    height, width = shape
    return max(0, y1), min(height, y2), max(0, x1), min(width, x2)


def summarize(values):
    return {'median': round(median(values), 4), 'mad': round(mad(values), 4)} if values else None


fs = cv2.FileStorage(str(CALIBRATION), cv2.FILE_STORAGE_READ)
kl, dl = fs.getNode('left_camera_matrix').mat(), fs.getNode('left_distortion').mat()
kr, dr = fs.getNode('right_camera_matrix').mat(), fs.getNode('right_distortion').mat()
rotation, translation = fs.getNode('rotation').mat(), fs.getNode('translation').mat()
fs.release()
size = (1600, 1200)
r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(kl, dl, kr, dr, size, rotation, translation,
                                             flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
map_left = cv2.initUndistortRectifyMap(kl, dl, r1, p1, size, cv2.CV_32FC1)
map_right = cv2.initUndistortRectifyMap(kr, dr, r2, p2, size, cv2.CV_32FC1)
focal_px = float(p1[0, 0])
matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5,
                                P1=8*5*5, P2=32*5*5, uniquenessRatio=8,
                                speckleWindowSize=50, speckleRange=2,
                                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '2.0'
detector = TensorRTDetector(str(ENGINE), confidence=0.10)
measurements = {video: {mode: [[] for _ in range(2)] for mode in MODES} for video in VIDEOS}
detector_times = []
stereo_pair_times = {mode: [] for mode in MODES}

for video_name in VIDEOS:
    capture = cv2.VideoCapture(str(ROOT / video_name))
    for time_s in SAMPLE_TIMES:
        capture.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        left, right = frame[:, :1600], frame[:, 1600:]
        detections, infer_ms = detector.infer(left)
        detector_times.append(infer_ms)
        assigned = [None, None]
        for box in detections:
            cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
            errors = [math.hypot(cx - x, cy - y) for x, y in TARGETS]
            error, target_i = min((value, i) for i, value in enumerate(errors))
            if error <= 60 and (assigned[target_i] is None or
                                box.confidence > assigned[target_i].confidence):
                assigned[target_i] = box
        rect_left = cv2.remap(left, map_left[0], map_left[1], cv2.INTER_LINEAR)
        rect_right = cv2.remap(right, map_right[0], map_right[1], cv2.INTER_LINEAR)
        gray_left = cv2.resize(cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY), (800, 600),
                               interpolation=cv2.INTER_AREA)
        gray_right = cv2.resize(cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY), (800, 600),
                                interpolation=cv2.INTER_AREA)
        for mode in MODES:
            start = time.perf_counter()
            left_input, right_input = preprocess(gray_left, mode), preprocess(gray_right, mode)
            disparity = matcher.compute(left_input, right_input).astype(np.float32) / 16.0
            stereo_pair_times[mode].append((time.perf_counter() - start) * 1000)
            for target_i, box in enumerate(assigned):
                if box is None:
                    continue
                y1, y2, x1, x2 = rectified_roi(box, kl, dl, r1, p1, disparity.shape)
                roi = disparity[y1:y2, x1:x2]
                valid = roi[np.isfinite(roi) & (roi > 3.0) & (roi < 120.0)]
                distance = None
                disp_med = None
                if valid.size >= 8:
                    disp_med = float(np.median(valid) * 2.0)
                    distance = BASELINE_M * focal_px / disp_med
                size_range = 1046.1599129444185 * FUEL_DIAMETER_M / math.sqrt(
                    max(1.0, (box.x2-box.x1) * (box.y2-box.y1)))
                point = np.array([[[ (box.x1+box.x2)/2, (box.y1+box.y2)/2 ]]], np.float32)
                rect_center = cv2.undistortPoints(point, kl, dl, R=r1, P=p1)[0, 0]
                measurements[video_name][mode][target_i].append({
                    'time_s': time_s, 'distance_m': distance, 'disparity_px': disp_med,
                    'valid_pixels': int(valid.size), 'size_proxy_m': size_range,
                    'rectified_center_px': [float(rect_center[0]), float(rect_center[1])],
                    'box_confidence': box.confidence,
                })
    capture.release()
detector.close()

results = []
for video_name in VIDEOS:
    for mode in MODES:
        target_results = []
        paired_points = {}
        for target_i, rows in enumerate(measurements[video_name][mode]):
            valid_rows = [row for row in rows if row['distance_m'] is not None]
            ranges = [row['distance_m'] for row in valid_rows]
            size_ranges = [row['size_proxy_m'] for row in rows]
            errors = [abs(row['distance_m'] - row['size_proxy_m']) for row in valid_rows]
            target_results.append({
                'target': target_i + 1,
                'valid_ranges': len(valid_rows),
                'range_m': summarize(ranges),
                'size_proxy_median': round(median(size_ranges), 4) if size_ranges else None,
                'median_abs_range_size_proxy_delta_m': round(median(errors), 4) if errors else None,
            })
            paired_points[target_i] = {round(row['time_s'], 3): row for row in valid_rows}
        shared_times = sorted(set(paired_points[0]) & set(paired_points[1]))
        pair_spacings = []
        for time_s in shared_times:
            points = []
            for target_i in range(2):
                row = paired_points[target_i][time_s]
                u, v = row['rectified_center_px']
                z = row['distance_m']
                points.append(((u-p1[0, 2])*z/p1[0, 0],
                               (v-p1[1, 2])*z/p1[1, 1], z))
            pair_spacings.append(math.dist(points[0], points[1]))
        merged = {str(radius): sum(value <= radius for value in pair_spacings)
                  for radius in (0.15, 0.20, 0.25)}
        results.append({
            'video': video_name, 'preprocessing': mode, 'targets': target_results,
            'paired_3d_spacing_m': summarize(pair_spacings),
            'paired_frames': len(pair_spacings), 'merged_pair_frames_at_radius_m': merged,
            'stereo_preprocess_plus_matcher_ms_median': round(median(stereo_pair_times[mode]), 2)
                if stereo_pair_times[mode] else None,
        })

output = {
    'method': 'Keep detector model/boxes, candidate rectification, SGBM 3WAY blockSize=5, uniquenessRatio=8, full detector box ROI, and 63.5 mm baseline fixed. Compare grayscale, CLAHE, equalization, blur/median denoising, gamma shifts, and bilateral filtering on 15 synchronized sampled frames per saved clip.',
    'limitations': 'Two stationary, unobstructed targets at about 1.4-1.6 m. Apparent-size distance is only a proxy; MAD measures repeatability, not absolute accuracy. Filter runtime is measured on this Jetson and includes preprocessing plus disparity matching.',
    'configuration': {'calibration': str(CALIBRATION.relative_to(ROOT)), 'rectified_focal_px': focal_px,
                      'baseline_m': BASELINE_M, 'sample_times_s': list(SAMPLE_TIMES),
                      'detector': 'FRC 9550 YOLO26n TensorRT 1280, confidence 0.10, aspect limit 2.0',
                      'median_detector_inference_ms': round(median(detector_times), 2)
                          if detector_times else None},
    'results': results,
}
output_path = ROOT / 'media/frc-stereo-preprocessing-sweep.json'
output_path.write_text(json.dumps(output, indent=2) + '\n')
for row in results:
    print(Path(row['video']).name, row['preprocessing'],
          [(target['range_m'], target['median_abs_range_size_proxy_delta_m'])
           for target in row['targets']], 'pair=', row['paired_3d_spacing_m'],
          'merge=', row['merged_pair_frames_at_radius_m'],
          'ms=', row['stereo_preprocess_plus_matcher_ms_median'])
