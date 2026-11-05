#!/usr/bin/env python3
"""Stress horizontal stereo ROI at image edges and a synthetic close disparity."""
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
CAL = ROOT / 'media/calibration/stereo_calibration_live_order_candidate.yml'
TARGETS = ((808, 589), (902, 591))
CONF = .05
SAMPLE_INDICES = tuple(range(100, 500, 20))
EDGE_SHIFTS = (-620, 620)
MODES = (
    {'name': 'fixed64_full', 'width': 64, 'horizontal': False, 'adaptive': False},
    {'name': 'fixed64_crop', 'width': 64, 'horizontal': True, 'adaptive': False},
    {'name': 'fixed128_full', 'width': 128, 'horizontal': False, 'adaptive': False},
    {'name': 'fixed128_crop', 'width': 128, 'horizontal': True, 'adaptive': False},
    {'name': 'adaptive128_crop', 'width': 128, 'horizontal': True, 'adaptive': True})
CLOSE_RIGHT_SHIFT = -76  # Adds 38 px at the half-resolution matcher scale.

fs = cv2.FileStorage(str(CAL), cv2.FILE_STORAGE_READ)
kl = fs.getNode('left_camera_matrix').mat()
dl = fs.getNode('left_distortion').mat()
kr = fs.getNode('right_camera_matrix').mat()
dr = fs.getNode('right_distortion').mat()
rotation = fs.getNode('rotation').mat()
translation = fs.getNode('translation').mat()
fs.release()
r1, r2, p1, p2, *_ = cv2.stereoRectify(
    kl, dl, kr, dr, (1600, 1200), rotation, translation,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
maps = (
    cv2.initUndistortRectifyMap(kl, dl, r1, p1, (1600, 1200), cv2.CV_32FC1),
    cv2.initUndistortRectifyMap(kr, dr, r2, p2, (1600, 1200), cv2.CV_32FC1))

os.environ.update(
    SCOUT_CLASS_NAMES='["ball"]', SCOUT_STEREO_CALIBRATION=str(CAL),
    SCOUT_STEREO_RECTIFY='0', SCOUT_STEREO_BASELINE_M='.0635',
    SCOUT_STEREO_FOCAL_PX=str(float(p1[0, 0])),
    SCOUT_STEREO_DISPARITY_SCALE='1', SCOUT_STEREO_MATCH_SCALE='.5',
    SCOUT_STEREO_VERTICAL_ROI_PAD='24',
    SCOUT_FUEL_DIAMETER_M='0', SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR='0',
    SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU='0')
from scout.detectors import Detection
from scout.stereo import StereoDepth


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(value - center) for value in values]) if values else None


def box_from_rectification(box, shift_x):
    corners = np.array([[[box['x1'], box['y1']]], [[box['x2'], box['y1']]],
                         [[box['x2'], box['y2']]], [[box['x1'], box['y2']]]],
                       dtype=np.float32)
    corners = cv2.undistortPoints(corners, kl, dl, R=r1, P=p1).reshape(-1, 2)
    x1 = int(np.floor(corners[:, 0].min())) + shift_x
    y1 = int(np.floor(corners[:, 1].min()))
    x2 = int(np.ceil(corners[:, 0].max())) + shift_x
    y2 = int(np.ceil(corners[:, 1].max()))
    return Detection(x1, y1, x2, y2, 0, float(box['confidence']))


def select(boxes):
    chosen = [None, None]
    for box in sorted((item for item in boxes if item['confidence'] >= CONF),
                      key=lambda item: item['confidence'], reverse=True):
        cx = (box['x1'] + box['x2']) / 2
        cy = (box['y1'] + box['y2']) / 2
        distance, index = min((math.hypot(cx - x, cy - y), i)
                              for i, (x, y) in enumerate(TARGETS))
        if distance <= 60 and chosen[index] is None:
            chosen[index] = box
    return chosen


def move(image, dx):
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, 0]]),
                          (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


results = []
for clip in DATA['results']:
    cap = cv2.VideoCapture(str(ROOT / clip['video']))
    records = {(shift, close, mode['name']): {'ranges': [[], []], 'valid_pairs': 0,
                                              'estimate_ms': [], 'widths': []}
               for shift in EDGE_SHIFTS for close in (False, True)
               for mode in MODES}
    estimators = {}
    for mode in MODES:
        os.environ['SCOUT_STEREO_NUM_DISPARITIES'] = str(mode['width'])
        os.environ['SCOUT_STEREO_HORIZONTAL_ROI'] = '1' if mode['horizontal'] else '0'
        os.environ['SCOUT_STEREO_ADAPTIVE_DISPARITY_WIDTH'] = '1' if mode['adaptive'] else '0'
        estimators[mode['name']] = StereoDepth()
    for index, row in enumerate(clip['per_frame']):
        ok, frame = cap.read()
        if not ok:
            break
        if index not in SAMPLE_INDICES:
            continue
        original = select(row['detections'])
        if not all(original):
            continue
        left = cv2.remap(frame[:, :1600], *maps[0], cv2.INTER_LINEAR)
        right = cv2.remap(frame[:, 1600:], *maps[1], cv2.INTER_LINEAR)
        for shift in EDGE_SHIFTS:
            boxes = [box_from_rectification(box, shift) for box in original]
            for close in (False, True):
                left_view = move(left, shift)
                right_view = move(right, shift + (CLOSE_RIGHT_SHIFT if close else 0))
                for mode in MODES:
                    estimator = estimators[mode['name']]
                    current_boxes = [Detection(b.x1, b.y1, b.x2, b.y2,
                                               b.class_id, b.confidence) for b in boxes]
                    started = time.perf_counter()
                    estimator.estimate(left_view, right_view, current_boxes)
                    elapsed = (time.perf_counter() - started) * 1000
                    record = records[(shift, close, mode['name'])]
                    record['estimate_ms'].append(elapsed)
                    record['widths'].append(estimator.last_num_disparities)
                    targets = current_boxes
                    if all(box is not None and box.distance_m is not None for box in targets):
                        record['valid_pairs'] += 1
                        for target, box in enumerate(targets):
                            record['ranges'][target].append(float(box.distance_m))
    cap.release()
    configurations = []
    for shift in EDGE_SHIFTS:
        for close in (False, True):
            for mode in MODES:
                pair = {'horizontal_shift_px': shift,
                        'synthetic_close_disparity_shift_px': CLOSE_RIGHT_SHIFT if close else 0,
                        'estimated_operating_distance_m': 'about 0.55-0.65' if close else 'about 1.4-1.6'}
                record = records[(shift, close, mode['name'])]
                latency = record['estimate_ms']
                pair.update({'method': mode['name'], 'configured_max_disparities': mode['width'],
                             'selected_widths': sorted(set(record['widths'])),
                             'valid_pair_frames': record['valid_pairs'],
                             'range_by_target_m': [
                                 {'n': len(values), 'median': round(median(values), 3),
                                  'mad': round(mad(values), 3)} if values else None
                                 for values in record['ranges']],
                             'estimate_ms_median': round(median(latency), 2) if latency else None})
                configurations.append(pair)
                print(clip['scene'], pair)
    results.append({'scene': clip['scene'], 'frames_sampled': len(SAMPLE_INDICES),
                    'configurations': configurations})

output = {
    'method': 'Rectify saved stereo pairs, translate both rectified views and matched target boxes toward the left/right image edge, then compare fixed 64/128-disparity settings with an adaptive width capped at 128 using the leftmost box center. Fixed widths run with and without horizontal cropping; the adaptive mode uses cropping. A second condition translates only the right image an additional 76 full-resolution pixels left (38 half-scale pixels) to synthesize a closer disparity while preserving local texture. This stresses crop boundaries and configured search coverage; it is not a physical close-range recording.',
    'limitations': 'Synthetic translations do not reproduce lens vignetting, occlusion, close-range appearance or actual camera geometry. This verifies disparity-crop boundaries and fallback behavior only; metric depth accuracy needs real measured targets.',
    'results': results}
path = ROOT / 'media/frc-fuel-stereo-horizontal-roi-edge-sweep.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print('saved', path)
