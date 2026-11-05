#!/usr/bin/env python3
"""Compare disparity aggregation rules inside the same detector ROIs."""
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
AGGREGATORS = ('median', 'mean', 'trimmed10', 'trimmed25', 'winsor10', 'mode_halfpx')
CLUSTER_RADII_M = (0.15, 0.20, 0.25)


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(value-center) for value in values]) if values else None


def robust_summary(values):
    return {'median': round(median(values), 4), 'mad': round(mad(values), 4)} if values else None


def aggregate(values, method):
    vals = np.asarray(values, np.float64)
    if not vals.size:
        return None
    ordered = np.sort(vals)
    if method == 'median':
        return float(np.median(ordered))
    if method == 'mean':
        return float(np.mean(ordered))
    if method in ('trimmed10', 'trimmed25'):
        trim_fraction = 0.10 if method == 'trimmed10' else 0.25
        trim_n = int(len(ordered) * trim_fraction)
        retained = ordered[trim_n:len(ordered)-trim_n] if trim_n else ordered
        return float(np.mean(retained))
    if method == 'winsor10':
        low, high = np.quantile(ordered, (0.10, 0.90))
        return float(np.mean(np.clip(ordered, low, high)))
    if method == 'mode_halfpx':
        # Disparity values here are half-resolution pixels; 0.25 px bins equal 0.5 full-res px.
        bin_width = 0.25
        origin = math.floor(float(ordered.min()) / bin_width) * bin_width
        bins = np.floor((ordered - origin) / bin_width).astype(int)
        counts = np.bincount(bins)
        modal_bin = int(np.argmax(counts))
        neighborhood = ordered[np.abs(ordered - (origin + (modal_bin + 0.5) * bin_width))
                                <= bin_width]
        return float(np.mean(neighborhood if neighborhood.size else ordered[bins == modal_bin]))
    raise ValueError(method)


def rectified_roi(box, kl, dl, r1, p1, shape):
    corners = np.array([[[box.x1, box.y1]], [[box.x2, box.y1]],
                        [[box.x2, box.y2]], [[box.x1, box.y2]]], np.float32)
    rect = cv2.undistortPoints(corners, kl, dl, R=r1, P=p1).reshape(-1, 2) / 2.0
    x1, y1 = np.floor(rect.min(axis=0)).astype(int)
    x2, y2 = np.ceil(rect.max(axis=0)).astype(int)
    height, width = shape
    return max(0, y1), min(height, y2), max(0, x1), min(width, x2)


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
records = {video: {method: [[] for _ in range(2)] for method in AGGREGATORS}
           for video in VIDEOS}
detector_times = []
matcher_times = []

for video_name in VIDEOS:
    capture = cv2.VideoCapture(str(ROOT / video_name))
    for time_s in SAMPLE_TIMES:
        capture.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        left, right = frame[:, :1600], frame[:, 1600:]
        detections, det_ms = detector.infer(left)
        detector_times.append(det_ms)
        targets = [None, None]
        for box in detections:
            cx, cy = (box.x1+box.x2)/2, (box.y1+box.y2)/2
            error, target_i = min((math.hypot(cx-x,cy-y),i) for i,(x,y) in enumerate(TARGETS))
            if error <= 60 and (targets[target_i] is None or
                                box.confidence > targets[target_i].confidence):
                targets[target_i] = box
        gray_left = cv2.cvtColor(cv2.remap(left,map_left[0],map_left[1],cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(cv2.remap(right,map_right[0],map_right[1],cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY)
        gray_left = cv2.resize(gray_left,(800,600),interpolation=cv2.INTER_AREA)
        gray_right = cv2.resize(gray_right,(800,600),interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        disparity = matcher.compute(gray_left,gray_right).astype(np.float32)/16.0
        matcher_times.append((time.perf_counter()-t0)*1000)
        for target_i, box in enumerate(targets):
            if box is None:
                continue
            y1,y2,x1,x2=rectified_roi(box,kl,dl,r1,p1,disparity.shape)
            roi=disparity[y1:y2,x1:x2]
            valid=roi[np.isfinite(roi)&(roi>3)&(roi<120)]
            if valid.size < 8:
                continue
            size_range=1046.1599129444185*FUEL_DIAMETER_M/math.sqrt(
                max(1.0,(box.x2-box.x1)*(box.y2-box.y1)))
            point=np.array([[[ (box.x1+box.x2)/2,(box.y1+box.y2)/2 ]]],np.float32)
            center=cv2.undistortPoints(point,kl,dl,R=r1,P=p1)[0,0]
            for method in AGGREGATORS:
                disp_half=aggregate(valid,method)
                distance=BASELINE_M*focal_px/(2*disp_half)
                records[video_name][method][target_i].append({
                    'time_s':time_s,'range_m':distance,'disparity_half_px':disp_half,
                    'size_proxy_m':size_range,'valid_pixels':int(valid.size),
                    'rectified_center_px':[float(center[0]),float(center[1])],
                    'box_confidence':box.confidence})
    capture.release()
detector.close()

results=[]
for video_name in VIDEOS:
    for method in AGGREGATORS:
        target_results=[]
        per_target_by_time=[]
        for target_i, rows in enumerate(records[video_name][method]):
            ranges=[row['range_m'] for row in rows]
            size_delta=[abs(row['range_m']-row['size_proxy_m']) for row in rows]
            target_results.append({'target':target_i+1,'valid_ranges':len(rows),
                'range_m':robust_summary(ranges),
                'median_abs_delta_to_size_proxy_m':round(median(size_delta),4) if size_delta else None})
            per_target_by_time.append({round(row['time_s'],3):row for row in rows})
        pair_spacing=[]
        for time_s in sorted(set(per_target_by_time[0])&set(per_target_by_time[1])):
            points=[]
            for target_i in range(2):
                row=per_target_by_time[target_i][time_s]
                u,v=row['rectified_center_px'];z=row['range_m']
                points.append(((u-p1[0,2])*z/p1[0,0],(v-p1[1,2])*z/p1[1,1],z))
            pair_spacing.append(math.dist(points[0],points[1]))
        results.append({'video':video_name,'aggregation':method,'targets':target_results,
            'paired_3d_spacing_m':robust_summary(pair_spacing),'paired_frames':len(pair_spacing),
            'merged_pair_frames_at_radius_m':{str(radius):sum(value<=radius for value in pair_spacing)
                for radius in CLUSTER_RADII_M}})

output={'method':'Hold FRC 9550 YOLO26n TensorRT 1280 boxes, rectification, raw grayscale, SGBM 3WAY b5/u8, full-box ROI, and calibration fixed. Compare median, mean, trimmed/winsorized mean, and modal-bin disparity summaries on 15 sampled pairs per clip.',
 'limitations':'Two stationary visible balls at about 1.4-1.6 m. Apparent-size range is only a proxy and MAD measures repeatability, not absolute accuracy. This does not test field clusters or occlusion.',
 'configuration':{'calibration':str(CALIBRATION.relative_to(ROOT)),'baseline_m':BASELINE_M,
  'rectified_focal_px':focal_px,'sample_times_s':list(SAMPLE_TIMES),
  'detector':'FRC 9550 YOLO26n TensorRT 1280, confidence 0.10, aspect limit 2.0',
  'median_detector_ms':round(median(detector_times),2) if detector_times else None,
  'median_sgbm_compute_ms':round(median(matcher_times),2) if matcher_times else None},
 'results':results}
out_path=ROOT/'media/frc-stereo-roi-aggregation-sweep.json'
out_path.write_text(json.dumps(output,indent=2)+'\n')
for row in results:
 print(Path(row['video']).name,row['aggregation'],
  [(target['range_m'],target['median_abs_delta_to_size_proxy_m']) for target in row['targets']],
  'spacing=',row['paired_3d_spacing_m'],'merged=',row['merged_pair_frames_at_radius_m'])
