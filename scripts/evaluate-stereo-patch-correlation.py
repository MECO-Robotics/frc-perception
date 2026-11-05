#!/usr/bin/env python3
"""Compare local stereo patch correlation with production SGBM on saved clips."""
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
PATCH_SIZES = (15, 21, 27)  # pixels at half resolution
MIN_DISPARITY = 8.0
MAX_DISPARITY = 50.0
LR_TOLERANCE = 2.0
METHODS = ('gray_ccoeff', 'gradient_ccoeff')


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(value - center) for value in values]) if values else None


def feature_image(gray, method):
    if method == 'gray_ccoeff':
        return gray
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def correlate_one_direction(template_image, search_image, cx, cy, patch_size,
                            min_x_delta, max_x_delta, direction):
    """Return the strongest horizontal template match and normalized correlation."""
    half = patch_size // 2
    height, width = template_image.shape[:2]
    cx, cy = int(round(cx)), int(round(cy))
    if cx - half < 0 or cx + half >= width or cy - half < 0 or cy + half >= height:
        return None
    template = template_image[cy-half:cy+half+1, cx-half:cx+half+1]
    if direction == 'left_to_right':
        search_x1 = max(0, int(math.floor(cx - max_x_delta - half)))
        search_x2 = min(width, int(math.ceil(cx - min_x_delta + half + 1)))
    else:
        search_x1 = max(0, int(math.floor(cx + min_x_delta - half)))
        search_x2 = min(width, int(math.ceil(cx + max_x_delta + half + 1)))
    search_y1, search_y2 = max(0, cy-half), min(height, cy+half+1)
    search = search_image[search_y1:search_y2, search_x1:search_x2]
    if search.shape[0] < patch_size or search.shape[1] < patch_size:
        return None
    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(response)
    peak_x = location[0]
    subpixel_offset = 0.0
    if 0 < peak_x < response.shape[1] - 1:
        left_score = float(response[0, peak_x - 1])
        peak_score = float(response[0, peak_x])
        right_score = float(response[0, peak_x + 1])
        curvature = left_score - 2 * peak_score + right_score
        if abs(curvature) > 1e-8:
            subpixel_offset = float(np.clip(0.5 * (left_score - right_score) / curvature, -1, 1))
    matched_cx = search_x1 + peak_x + subpixel_offset + half
    delta = cx - matched_cx if direction == 'left_to_right' else matched_cx - cx
    return float(delta), float(score)


def stereo_roi(box, kl, dl, r1, p1, shape):
    x1, y1, x2, y2 = box
    corners = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], np.float32)
    rect = cv2.undistortPoints(corners, kl, dl, R=r1, P=p1).reshape(-1, 2) / 2.0
    x1, y1 = np.floor(rect.min(axis=0)).astype(int)
    x2, y2 = np.ceil(rect.max(axis=0)).astype(int)
    h, w = shape
    return max(0, y1), min(h, y2), max(0, x1), min(w, x2)


fs = cv2.FileStorage(str(CALIBRATION), cv2.FILE_STORAGE_READ)
kl, dl = fs.getNode('left_camera_matrix').mat(), fs.getNode('left_distortion').mat()
kr, dr = fs.getNode('right_camera_matrix').mat(), fs.getNode('right_distortion').mat()
rotation, translation = fs.getNode('rotation').mat(), fs.getNode('translation').mat()
fs.release()
size = (1600, 1200)
r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(kl, dl, kr, dr, size, rotation, translation,
                                             flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
ml = cv2.initUndistortRectifyMap(kl, dl, r1, p1, size, cv2.CV_32FC1)
mr = cv2.initUndistortRectifyMap(kr, dr, r2, p2, size, cv2.CV_32FC1)
focal_px = float(p1[0, 0])
matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5,
                                P1=8*5*5, P2=32*5*5, uniquenessRatio=8,
                                speckleWindowSize=50, speckleRange=2,
                                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '2.0'
detector = TensorRTDetector(str(ENGINE), confidence=0.10)
configs = [(method, patch_size, lr_check) for method in METHODS
           for patch_size in PATCH_SIZES for lr_check in (False, True)]
records = {video: {config: [[] for _ in range(2)] for config in configs} for video in VIDEOS}
detector_times = []

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
        left_gray = cv2.cvtColor(cv2.remap(left, ml[0], ml[1], cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(cv2.remap(right, mr[0], mr[1], cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
        left_gray = cv2.resize(left_gray, (800, 600), interpolation=cv2.INTER_AREA)
        right_gray = cv2.resize(right_gray, (800, 600), interpolation=cv2.INTER_AREA)
        disparity_map = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        left_features = {method: feature_image(left_gray, method) for method in METHODS}
        right_features = {method: feature_image(right_gray, method) for method in METHODS}
        for target_i, box in enumerate(assigned):
            if box is None:
                continue
            raw_center = np.array([[[ (box.x1 + box.x2) / 2,
                                      (box.y1 + box.y2) / 2 ]]], dtype=np.float32)
            rect_center = cv2.undistortPoints(raw_center, kl, dl, R=r1, P=p1)[0, 0] / 2.0
            y0, y1, x0, x1 = stereo_roi((box.x1, box.y1, box.x2, box.y2),
                                         kl, dl, r1, p1, disparity_map.shape)
            roi = disparity_map[y0:y1, x0:x1]
            valid = roi[np.isfinite(roi) & (roi > 3) & (roi < 120)]
            sgbm_disparity = float(np.median(valid)) if valid.size >= 8 else None
            sgbm_range = BASELINE_M * focal_px / (2 * sgbm_disparity) if sgbm_disparity else None
            size_range = 1046.1599129444185 * FUEL_DIAMETER_M / math.sqrt(
                max(1.0, (box.x2-box.x1) * (box.y2-box.y1)))
            for config in configs:
                method, patch_size, lr_check = config
                forward = correlate_one_direction(left_features[method], right_features[method],
                    rect_center[0], rect_center[1], patch_size, MIN_DISPARITY,
                    MAX_DISPARITY, 'left_to_right')
                measured = None
                reverse = None
                if forward is not None:
                    disp, score = forward
                    if disp > 0:
                        reverse = correlate_one_direction(right_features[method], left_features[method],
                            rect_center[0] - disp, rect_center[1], patch_size, MIN_DISPARITY,
                            MAX_DISPARITY, 'right_to_left')
                        consistent = reverse is not None and abs(reverse[0] - disp) <= LR_TOLERANCE
                        if not lr_check or consistent:
                            measured = BASELINE_M * focal_px / (2 * disp)
                    else:
                        score = None
                        consistent = False
                else:
                    disp = score = None
                    consistent = False
                records[video_name][config][target_i].append({
                    'time_s': time_s, 'disparity_half_px': disp, 'correlation': score,
                    'reverse_disparity_half_px': reverse[0] if reverse else None,
                    'left_right_consistent': consistent,
                    'patch_range_m': measured, 'sgbm_range_m': sgbm_range,
                    'size_proxy_range_m': size_range,
                    'box_confidence': box.confidence,
                    'box_xyxy_px': [box.x1, box.y1, box.x2, box.y2],
                })
    capture.release()
detector.close()

results = []
for video_name in VIDEOS:
    for config in configs:
        method, patch_size, lr_check = config
        targets = []
        for target_i, rows in enumerate(records[video_name][config]):
            patch_ranges = [row['patch_range_m'] for row in rows if row['patch_range_m'] is not None]
            sgbm_ranges = [row['sgbm_range_m'] for row in rows if row['sgbm_range_m'] is not None]
            size_ranges = [row['size_proxy_range_m'] for row in rows]
            paired = [(row['patch_range_m'], row['sgbm_range_m']) for row in rows
                      if row['patch_range_m'] is not None and row['sgbm_range_m'] is not None]
            targets.append({
                'target': target_i + 1, 'samples': len(rows), 'valid_patch_ranges': len(patch_ranges),
                'patch_range_median_m': round(median(patch_ranges), 4) if patch_ranges else None,
                'patch_range_mad_m': round(mad(patch_ranges), 4) if patch_ranges else None,
                'sgbm_range_median_m': round(median(sgbm_ranges), 4) if sgbm_ranges else None,
                'sgbm_range_mad_m': round(mad(sgbm_ranges), 4) if sgbm_ranges else None,
                'size_proxy_median_m': round(median(size_ranges), 4) if size_ranges else None,
                'median_abs_patch_sgbm_delta_m': round(median([abs(a-b) for a,b in paired]), 4)
                    if paired else None,
                'median_abs_patch_size_delta_m': round(median([abs(row['patch_range_m']-row['size_proxy_range_m'])
                    for row in rows if row['patch_range_m'] is not None]), 4) if patch_ranges else None,
                'median_correlation': round(median([row['correlation'] for row in rows
                    if row['correlation'] is not None]), 4),
                'lr_consistent_samples': sum(bool(row['left_right_consistent']) for row in rows),
            })
        results.append({'video': video_name, 'feature': method, 'patch_size_half_resolution_px': patch_size,
                        'left_right_check': lr_check, 'targets': targets,
                        'measurements': records[video_name][config]})

output = {
    'method': 'FRC 9550 TensorRT 1280 detections on 15 sampled pairs per saved video. Rectified grayscale or Sobel-magnitude patches are searched horizontally in the right image by normalized cross-correlation; optional reverse search requires disparity agreement within 2 px at half resolution. Compare with production SGBM block 5/uniqueness 8 and the nominal 15 cm apparent-box-size proxy.',
    'limitations': 'Stationary, unobstructed balls only; detector centers initialize fixed target matches. The same images provide SGBM and size proxies, neither of which is metric ground truth. Patch correlation may lock onto background texture; left/right consistency only checks match symmetry.',
    'configuration': {'calibration': str(CALIBRATION.relative_to(ROOT)), 'baseline_m': BASELINE_M,
                      'rectified_focal_px': focal_px, 'sample_times_s': list(SAMPLE_TIMES),
                      'detector': 'FRC 9550 YOLO26n TensorRT 1280, confidence 0.10, aspect 2.0',
                      'median_detector_ms': round(median(detector_times), 2) if detector_times else None},
    'results': results,
}
out_path = ROOT / 'media/frc-stereo-patch-correlation-sweep.json'
out_path.write_text(json.dumps(output, indent=2) + '\n')
for row in results:
    print(Path(row['video']).name, row['feature'], row['patch_size_half_resolution_px'],
          'lr=', row['left_right_check'], [(t['valid_patch_ranges'], t['patch_range_median_m'],
          t['patch_range_mad_m'], t['sgbm_range_median_m'], t['median_abs_patch_size_delta_m'],
          t['lr_consistent_samples']) for t in row['targets']])
