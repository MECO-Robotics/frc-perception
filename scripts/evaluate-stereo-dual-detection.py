#!/usr/bin/env python3
"""Test direct stereo ranging by matching FUEL detector boxes across both views."""
import itertools
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
SAMPLE_TIMES = (1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.4, 5.8, 6.2, 6.6)
TARGETS = ((808, 589), (902, 591))
VERTICAL_GATES_PX = (2.0, 4.0, 8.0, 12.0)
BASELINE_M = 0.0635
FUEL_DIAMETER_M = 0.15


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(value - center) for value in values]) if values else None


def summarize(values):
    return {'median': round(median(values), 4), 'mad': round(mad(values), 4)} if values else None


def rectified_box(box, camera, distortion, rotation, projection):
    corners = np.array([[[box.x1, box.y1]], [[box.x2, box.y1]],
                        [[box.x2, box.y2]], [[box.x1, box.y2]]], np.float32)
    points = cv2.undistortPoints(corners, camera, distortion, R=rotation, P=projection).reshape(-1, 2)
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    center = ((x1+x2)/2, (y1+y2)/2)
    return {'cx': float(center[0]), 'cy': float(center[1]), 'w': float(x2-x1),
            'h': float(y2-y1), 'area': float((x2-x1)*(y2-y1)), 'source': box}


def associate(left_targets, right_boxes, vertical_gate):
    """Brute-force one-to-one assignment for at most a few detected balls."""
    best = None
    for choices in itertools.product(range(-1, len(right_boxes)), repeat=len(left_targets)):
        real = [choice for choice in choices if choice >= 0]
        if len(real) != len(set(real)):
            continue
        costs, row = [], []
        for target_i, right_i in enumerate(choices):
            if left_targets[target_i] is None:
                row.append(None)
                if right_i >= 0:
                    costs.append(1e6)
                continue
            if right_i < 0:
                row.append(None)
                continue
            left, right = left_targets[target_i], right_boxes[right_i]
            disparity = left['cx'] - right['cx']
            vertical_error = abs(left['cy'] - right['cy'])
            if disparity <= 0 or vertical_error > vertical_gate:
                row.append(None)
                continue
            size_penalty = abs(math.log(max(1.0, left['area']) / max(1.0, right['area'])))
            aspect_penalty = abs(math.log(max(1e-6, left['w']/left['h']) /
                                          max(1e-6, right['w']/right['h'])))
            cost = vertical_error + 2.0 * size_penalty + aspect_penalty
            costs.append(cost)
            row.append({'right_index': right_i, 'disparity_px': disparity,
                        'vertical_error_px': vertical_error, 'cost': cost,
                        'confidence': right['source'].confidence})
        # Prefer more valid matches, then lower total geometry/size mismatch.
        valid_n = sum(item is not None for item in row)
        objective = (-valid_n, sum(costs))
        if best is None or objective < best[0]:
            best = (objective, row)
    return best[1] if best else [None] * len(left_targets)


fs = cv2.FileStorage(str(CALIBRATION), cv2.FILE_STORAGE_READ)
kl, dl = fs.getNode('left_camera_matrix').mat(), fs.getNode('left_distortion').mat()
kr, dr = fs.getNode('right_camera_matrix').mat(), fs.getNode('right_distortion').mat()
rotation, translation = fs.getNode('rotation').mat(), fs.getNode('translation').mat()
fs.release()
size = (1600, 1200)
r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(kl, dl, kr, dr, size, rotation, translation,
                                             flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
focal_px = float(p1[0, 0])
matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5,
                                P1=8*5*5, P2=32*5*5, uniquenessRatio=8,
                                speckleWindowSize=50, speckleRange=2,
                                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
map_left = cv2.initUndistortRectifyMap(kl, dl, r1, p1, size, cv2.CV_32FC1)
map_right = cv2.initUndistortRectifyMap(kr, dr, r2, p2, size, cv2.CV_32FC1)
os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '2.0'
detector = TensorRTDetector(str(ENGINE), confidence=0.10)
results = []
detector_times = []

for video_name in VIDEOS:
    capture = cv2.VideoCapture(str(ROOT / video_name))
    frame_rows = []
    for time_s in SAMPLE_TIMES:
        capture.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        left, right = frame[:, :1600], frame[:, 1600:]
        left_detections, left_ms = detector.infer(left)
        right_detections, right_ms = detector.infer(right)
        detector_times.extend((left_ms, right_ms))
        left_targets = [None, None]
        for box in left_detections:
            cx, cy = (box.x1+box.x2)/2, (box.y1+box.y2)/2
            error, target_i = min((math.hypot(cx-x, cy-y), i) for i,(x,y) in enumerate(TARGETS))
            if error <= 60 and (left_targets[target_i] is None or
                                box.confidence > left_targets[target_i]['source'].confidence):
                left_targets[target_i] = rectified_box(box, kl, dl, r1, p1)
        right_boxes = [rectified_box(box, kr, dr, r2, p2) for box in right_detections]
        rect_left = cv2.cvtColor(cv2.remap(left, map_left[0], map_left[1], cv2.INTER_LINEAR),
                                 cv2.COLOR_BGR2GRAY)
        rect_right = cv2.cvtColor(cv2.remap(right, map_right[0], map_right[1], cv2.INTER_LINEAR),
                                  cv2.COLOR_BGR2GRAY)
        rect_left = cv2.resize(rect_left, (800, 600), interpolation=cv2.INTER_AREA)
        rect_right = cv2.resize(rect_right, (800, 600), interpolation=cv2.INTER_AREA)
        disparity = matcher.compute(rect_left, rect_right).astype(np.float32) / 16.0
        sgbm_rows = [None, None]
        for target_i, box in enumerate(left_targets):
            if box is None:
                continue
            x1,y1,x2,y2 = box['cx']-box['w']/2,box['cy']-box['h']/2,box['cx']+box['w']/2,box['cy']+box['h']/2
            ix1,iy1,ix2,iy2 = [int(round(v/2)) for v in (x1,y1,x2,y2)]
            roi = disparity[max(0,iy1):min(600,iy2),max(0,ix1):min(800,ix2)]
            valid = roi[np.isfinite(roi) & (roi > 3) & (roi < 120)]
            if valid.size >= 8:
                z = BASELINE_M * focal_px / (2 * float(np.median(valid)))
                sgbm_rows[target_i] = {'range_m': z, 'valid_pixels': int(valid.size)}
        frame_row = {'time_s': time_s, 'right_detection_count': len(right_boxes), 'sgbm': sgbm_rows,
                     'associations': {}}
        for gate in VERTICAL_GATES_PX:
            association = associate(left_targets, right_boxes, gate)
            pair_rows = []
            for target_i, match in enumerate(association):
                left_box = left_targets[target_i]
                row = None
                if left_box is not None and match is not None:
                    disparity_px = match['disparity_px']
                    stereo_range = BASELINE_M * focal_px / disparity_px
                    source = left_box['source']
                    size_range = 1046.1599129444185 * FUEL_DIAMETER_M / math.sqrt(
                        max(1.0, (source.x2-source.x1)*(source.y2-source.y1)))
                    row = {'disparity_px': disparity_px, 'range_m': stereo_range,
                           'size_proxy_m': size_range, 'vertical_error_px': match['vertical_error_px'],
                           'left_confidence': source.confidence,
                           'right_confidence': match['confidence'],
                           'sgbm_range_m': sgbm_rows[target_i]['range_m'] if sgbm_rows[target_i] else None}
                    pair_rows.append((target_i, left_box, stereo_range))
                if row:
                    frame_row['associations'].setdefault(str(gate), [None,None])[target_i] = row
            if len(pair_rows) == 2:
                points = []
                for target_i, box, z in pair_rows:
                    points.append(((box['cx']-p1[0,2])*z/p1[0,0],
                                   (box['cy']-p1[1,2])*z/p1[1,1],z))
                frame_row['associations'][str(gate) + '_pair_spacing_m'] = math.dist(points[0],points[1])
        frame_rows.append(frame_row)
    capture.release()
    for gate in VERTICAL_GATES_PX:
        gate_rows = [row for row in frame_rows if row['associations'].get(str(gate))]
        target_summaries = []
        for target_i in range(2):
            rows = [row['associations'][str(gate)][target_i] for row in gate_rows
                    if row['associations'][str(gate)][target_i] is not None]
            distances = [row['range_m'] for row in rows]
            proxy_delta = [abs(row['range_m']-row['size_proxy_m']) for row in rows]
            stereo_delta = [abs(row['range_m']-row['sgbm_range_m']) for row in rows
                            if row['sgbm_range_m'] is not None]
            target_summaries.append({'target':target_i+1,'matched_frames':len(rows),
                'range_m':{'median':round(median(distances),4),'mad':round(mad(distances),4)} if distances else None,
                'median_abs_delta_to_size_proxy_m':round(median(proxy_delta),4) if proxy_delta else None,
                'median_abs_delta_to_sgbm_m':round(median(stereo_delta),4) if stereo_delta else None,
                'median_vertical_error_px':round(median([r['vertical_error_px'] for r in rows]),2) if rows else None})
        pair_spacings = [row['associations'][str(gate)+'_pair_spacing_m'] for row in frame_rows
                         if str(gate)+'_pair_spacing_m' in row['associations']]
        results.append({'video':video_name,'vertical_gate_px':gate,
            'paired_targets':sum(all(row['associations'].get(str(gate),[None,None])) for row in frame_rows),
            'targets':target_summaries,'pair_spacing_m':{'median':round(median(pair_spacings),4),
                'mad':round(mad(pair_spacings),4)} if pair_spacings else None,
            'frames':frame_rows})
detector.close()

output={'method':'Run the FRC 9550 YOLO26n TensorRT 1280 detector independently on both rectified-camera source images at confidence 0.10/aspect 2.0. Match left targets to right boxes one-to-one using vertical epipolar error plus box area/aspect similarity; sweep vertical gates, then triangulate from detector box centers. Compare with left-view SGBM and the 15 cm apparent-box-size proxy.',
 'limitations':'Only two stationary room-view balls; target windows select left detections. Box-center localization is not a labeled correspondence truth, and size/SGBM are not measured range truth. Right-view inference approximately doubles detector compute; this is an offline test.',
 'configuration':{'calibration':str(CALIBRATION.relative_to(ROOT)),'baseline_m':BASELINE_M,
  'rectified_focal_px':focal_px,'sample_times_s':list(SAMPLE_TIMES),
  'detector':'FRC 9550 YOLO26n TensorRT 1280, confidence 0.10, aspect limit 2.0',
  'median_single_view_detector_ms':round(median(detector_times),2) if detector_times else None},
 'results':results}
out_path=ROOT/'media/frc-stereo-dual-detection-sweep.json'
out_path.write_text(json.dumps(output,indent=2)+'\n')
for row in results:
 print(Path(row['video']).name,'gate=',row['vertical_gate_px'],'paired=',row['paired_targets'],
  'targets=',[(t['range_m'],t['median_abs_delta_to_sgbm_m'],t['median_vertical_error_px']) for t in row['targets']],
  'spacing=',row['pair_spacing_m'])
