#!/usr/bin/env python3
"""Compare fixed and edge-adaptive SGBM width through runtime StereoDepth."""
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
CAL = 'media/calibration/stereo_calibration_live_order_candidate.yml'
TARGETS = ((808, 589), (902, 591))
STRIDE = 20
CONF = .05
MODES = (
    {'name': 'fixed64_crop', 'width': 64, 'horizontal': True, 'adaptive': False},
    {'name': 'fixed128_crop', 'width': 128, 'horizontal': True, 'adaptive': False},
    {'name': 'adaptive128_crop', 'width': 128, 'horizontal': True, 'adaptive': True})

os.environ.update(
    SCOUT_CLASS_NAMES='["ball"]', SCOUT_STEREO_CALIBRATION=CAL,
    SCOUT_STEREO_RECTIFY='1', SCOUT_STEREO_BASELINE_M='.0635',
    SCOUT_STEREO_DISPARITY_SCALE='1', SCOUT_STEREO_MATCH_SCALE='.5',
    SCOUT_STEREO_VERTICAL_ROI_PAD='24', SCOUT_FUEL_DIAMETER_M='.15',
    SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR='.55',
    SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU='.90')
from scout.detectors import Detection
from scout.stereo import StereoDepth


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(x - center) for x in values]) if values else None


def match(boxes):
    result = [None, None]
    extras = 0
    for box in sorted(boxes, key=lambda item: item.confidence, reverse=True):
        cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        distance, index = min((math.hypot(cx - x, cy - y), i)
                              for i, (x, y) in enumerate(TARGETS))
        if distance <= 60 and result[index] is None:
            result[index] = box
        else:
            extras += 1
    return result, extras


def to_boxes(row):
    return [Detection(int(box['x1']), int(box['y1']), int(box['x2']),
                      int(box['y2']), 0, float(box['confidence']))
            for box in row['detections'] if box['confidence'] >= CONF]


output = {
    'method': 'On every twentieth saved frame, run all confidence-0.05 detections through runtime StereoDepth at fixed 64, fixed 128, and adaptive-to-leftmost-box-center (maximum 128) disparity widths, all with horizontal ROI and the active 24 px vertical band. Apply the active 55% nominal-size/stereo gate and duplicate suppression before target scoring.',
    'sample_stride_frames': STRIDE, 'confidence': CONF,
    'settings': {'horizontal_roi': True, 'adaptive_max_width': 128,
                 'vertical_pad': 24, 'match_scale': .5,
                 'size_gate_relative_tolerance': .55},
    'limitations': 'Fixed stationary pair at one camera pose/range; no tape-measured distance. Width 64 theoretical near limit is about 0.55 m. This sample does not validate response during target motion or false boxes near the image edge.',
    'results': []}

for clip in DATA['results']:
    estimators = {}
    for mode in MODES:
        os.environ['SCOUT_STEREO_NUM_DISPARITIES'] = str(mode['width'])
        os.environ['SCOUT_STEREO_HORIZONTAL_ROI'] = '1' if mode['horizontal'] else '0'
        os.environ['SCOUT_STEREO_ADAPTIVE_DISPARITY_WIDTH'] = '1' if mode['adaptive'] else '0'
        estimators[mode['name']] = StereoDepth()
    caps = cv2.VideoCapture(str(ROOT / clip['video']))
    recs = {mode['name']: {'hist': {0: 0, 1: 0, 2: 0}, 'ranges': [[], []],
                           'spacing': [], 'widths': {}, 'latency_ms': [],
                           'off_target': 0, 'gate_rejected': 0,
                           'duplicate_rejected': 0}
            for mode in MODES}
    frame_count = 0
    for index, row in enumerate(clip['per_frame']):
        ok, frame = caps.read()
        if not ok:
            break
        frame_count += 1
        if index % STRIDE:
            continue
        for mode in MODES:
            estimator = estimators[mode['name']]
            boxes = to_boxes(row)
            started = time.perf_counter()
            estimator.estimate(frame[:, :1600], frame[:, 1600:], boxes)
            rec = recs[mode['name']]
            rec['latency_ms'].append((time.perf_counter() - started) * 1000)
            width = estimator.last_num_disparities
            rec['widths'][width] = rec['widths'].get(width, 0) + 1
            filtered = estimator.filter_fuel_size_inconsistent(boxes)
            rec['gate_rejected'] += estimator.last_size_gate_rejected
            rec['duplicate_rejected'] += estimator.last_size_gate_duplicate_rejected
            targets, extras = match(filtered)
            rec['off_target'] += extras
            rec['hist'][sum(box is not None for box in targets)] += 1
            points = []
            for target, box in enumerate(targets):
                if box is None or box.distance_m is None:
                    points.append(None)
                    continue
                z = float(box.distance_m)
                rec['ranges'][target].append(z)
                center = np.array([[[float(box.x1 + box.x2) / 2,
                                     float(box.y1 + box.y2) / 2]]], np.float32)
                _, _, _, kl, r1, p1 = estimator.maps
                uv = cv2.undistortPoints(
                    center, kl, estimator.calibration['left_distortion'],
                    R=r1, P=p1)[0, 0]
                points.append(((uv[0] - p1[0, 2]) * z / p1[0, 0],
                               (uv[1] - p1[1, 2]) * z / p1[1, 1], z))
            if all(point is not None for point in points):
                rec['spacing'].append(math.dist(*points))
    caps.release()
    clip_result = {'scene': clip['scene'], 'frames_decoded': frame_count,
                   'sampled_frames': len(range(0, frame_count, STRIDE)),
                   'configurations': []}
    for mode in MODES:
        rec = recs[mode['name']]
        lat = rec['latency_ms']
        config = {'method': mode['name'],
                  'target_count_histogram': {str(k): v for k, v in rec['hist'].items()},
                  'valid_pair_frames': len(rec['spacing']),
                  'off_target_boxes': rec['off_target'],
                  'gate_rejected': rec['gate_rejected'],
                  'duplicate_rejected': rec['duplicate_rejected'],
                  'selected_width_histogram': {str(k): v for k, v in rec['widths'].items()},
                  'range_by_target_m': [
                      {'n': len(values), 'median': round(median(values), 4),
                       'mad': round(mad(values), 4)} if values else None
                      for values in rec['ranges']],
                  'pair_spacing_m': {'n': len(rec['spacing']),
                                     'median': round(median(rec['spacing']), 4),
                                     'mad': round(mad(rec['spacing']), 4)}
                  if rec['spacing'] else None,
                  'estimate_ms_median': round(median(lat), 2) if lat else None,
                  'estimate_ms_p95': round(sorted(lat)[int(.95 * (len(lat) - 1))], 2)
                  if lat else None}
        clip_result['configurations'].append(config)
        print(clip['scene'], config)
    output['results'].append(clip_result)

path = ROOT / 'media/frc-fuel-stereo-adaptive-width-runtime-check.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print('saved', path)
