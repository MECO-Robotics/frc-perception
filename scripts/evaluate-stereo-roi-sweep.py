#!/usr/bin/env python3
"""Compare stereo disparity from full and center-cropped detector boxes."""
import json
import math
import os
import statistics
from pathlib import Path

import cv2
import numpy as np
from scout.detectors import TensorRTDetector

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / 'media/calibration/stereo_calibration_live_order_candidate.yml'
ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_1280_fp16.engine'
VIDEOS = ('media/frc-2026-live-two-fuel-60fps-cfr.mp4',
          'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4')
SAMPLE_TIMES = (1, 1.4, 1.8, 2.2, 2.6, 3, 3.4, 3.8, 4.2, 4.6, 5, 5.4, 5.8, 6.2, 6.6)
ROIS = (1.0, 0.8, 0.6, 0.4, 0.25, 0.15)
BASELINE_M = 0.0635
FUEL_DIAMETER_M = 0.15
TARGETS = ((808, 589), (902, 591))


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(x - center) for x in values]) if values else None


def rectified_roi(box, fraction, left_camera, left_dist, rotation, projection, shape):
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * fraction / 2, (y2 - y1) * fraction / 2
    corners = np.array([[[cx-half_w, cy-half_h]], [[cx+half_w, cy-half_h]],
                        [[cx+half_w, cy+half_h]], [[cx-half_w, cy+half_h]]], dtype=np.float32)
    rect = cv2.undistortPoints(corners, left_camera, left_dist,
                               R=rotation, P=projection).reshape(-1, 2) / 2.0
    x1, y1 = np.floor(rect.min(axis=0)).astype(int)
    x2, y2 = np.ceil(rect.max(axis=0)).astype(int)
    h, w = shape
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return slice(y1, y2), slice(x1, x2)


fs = cv2.FileStorage(str(CALIBRATION), cv2.FILE_STORAGE_READ)
kl = fs.getNode('left_camera_matrix').mat()
dl = fs.getNode('left_distortion').mat()
kr = fs.getNode('right_camera_matrix').mat()
dr = fs.getNode('right_distortion').mat()
rotation = fs.getNode('rotation').mat()
translation = fs.getNode('translation').mat()
fs.release()
size = (1600, 1200)
r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(kl, dl, kr, dr, size, rotation, translation,
                                             flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
ml = cv2.initUndistortRectifyMap(kl, dl, r1, p1, size, cv2.CV_32FC1)
mr = cv2.initUndistortRectifyMap(kr, dr, r2, p2, size, cv2.CV_32FC1)
focal = float(p1[0, 0])
matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5,
                                P1=8*5*5, P2=32*5*5, uniquenessRatio=8,
                                speckleWindowSize=50, speckleRange=2,
                                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '2.0'
detector = TensorRTDetector(str(ENGINE), confidence=0.10)
records = []
for video_name in VIDEOS:
    video = ROOT / video_name
    cap = cv2.VideoCapture(str(video))
    target_samples = {(fraction, target): [] for fraction in ROIS for target in range(2)}
    detector_timings_ms = []
    for sample_i, time_s in enumerate(SAMPLE_TIMES):
        cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        left, right = frame[:, :1600], frame[:, 1600:]
        detections, detector_ms = detector.infer(left)
        detector_timings_ms.append(detector_ms)
        gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        gl = cv2.remap(gl, ml[0], ml[1], cv2.INTER_LINEAR)
        gr = cv2.remap(gr, mr[0], mr[1], cv2.INTER_LINEAR)
        gl = cv2.resize(gl, (800, 600), interpolation=cv2.INTER_AREA)
        gr = cv2.resize(gr, (800, 600), interpolation=cv2.INTER_AREA)
        disparity = matcher.compute(gl, gr).astype(np.float32) / 16.0
        assigned = [None, None]
        for box in detections:
            if box.confidence < 0.10:
                continue
            x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
            cx, cy = (x1+x2)/2, (y1+y2)/2
            distances = [math.hypot(cx-x, cy-y) for x, y in TARGETS]
            distance, target_i = min((v, i) for i, v in enumerate(distances))
            if distance <= 60 and assigned[target_i] is None:
                assigned[target_i] = box
        for target_i, box in enumerate(assigned):
            if box is None:
                continue
            x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
            xyxy = (x1, y1, x2, y2)
            size_range = focal * FUEL_DIAMETER_M / math.sqrt((x2-x1)*(y2-y1))
            for fraction in ROIS:
                ys, xs = rectified_roi(xyxy, fraction, kl, dl, r1, p1, disparity.shape)
                roi = disparity[ys, xs]
                valid = roi[np.isfinite(roi) & (roi > 3.0) & (roi < 120.0)]
                range_m = None
                if valid.size >= 8:
                    disp_px = float(np.median(valid) * 2.0)
                    range_m = BASELINE_M * focal / disp_px
                target_samples[(fraction, target_i)].append({
                    'time_s': time_s, 'sample_i': sample_i, 'confidence': box.confidence,
                    'valid_disparity_pixels': int(valid.size), 'distance_m': range_m,
                    'size_range_m': size_range,
                })
    cap.release()
    for fraction in ROIS:
        for target_i in range(2):
            samples = target_samples[(fraction, target_i)]
            distances = [x['distance_m'] for x in samples if x['distance_m'] is not None]
            size_distances = [x['size_range_m'] for x in samples]
            deltas = [abs(x['distance_m'] - x['size_range_m']) for x in samples
                      if x['distance_m'] is not None]
            records.append({
                'video': video_name, 'roi_fraction': fraction,
                'target': target_i + 1, 'detected_samples': len(samples),
                'valid_range_samples': len(distances),
                'median_valid_disparity_pixels': round(median([x['valid_disparity_pixels'] for x in samples]), 1),
                'range_m': {'median': round(median(distances), 3), 'mad': round(mad(distances), 3)}
                           if distances else None,
                'size_range_proxy_median': round(median(size_distances), 3) if size_distances else None,
                'median_abs_stereo_size_difference_m': round(median(deltas), 3) if deltas else None,
                'detector_latency_median_ms': round(median(detector_timings_ms), 2),
            })
detector.close()
output = {
    'method': 'Run the active FRC 9550 TensorRT 1280 detector at confidence 0.10 and max box aspect ratio 2.0 on 15 sampled synchronized frames per video. Compute one rectified SGBM disparity map per frame, then estimate range from full or centered fraction of each box. Same 63.5 mm baseline, rectified focal length, and SGBM b5/u8 as service. Apparent-size range is a nominal 15 cm FUEL proxy, not ground truth.',
    'configuration': {'baseline_m': BASELINE_M, 'calibration_translation_baseline_m': float(np.linalg.norm(translation)),
                      'focal_px': focal, 'roi_fractions': list(ROIS), 'min_valid_pixels': 8},
    'results': records,
}
(ROOT / 'media/frc-stereo-box-roi-sweep.json').write_text(json.dumps(output, indent=2) + '\n')
print('saved', ROOT / 'media/frc-stereo-box-roi-sweep.json', 'records', len(records))
