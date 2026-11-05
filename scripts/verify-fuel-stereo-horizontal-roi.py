#!/usr/bin/env python3
"""Verify the optional horizontal SGBM crop through the runtime StereoDepth class."""
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
os.environ.update(
    SCOUT_CLASS_NAMES='["ball"]', SCOUT_STEREO_CALIBRATION=CAL,
    SCOUT_STEREO_RECTIFY='1', SCOUT_STEREO_BASELINE_M='.0635',
    SCOUT_STEREO_DISPARITY_SCALE='1', SCOUT_STEREO_MATCH_SCALE='.5',
    SCOUT_STEREO_NUM_DISPARITIES='128', SCOUT_STEREO_VERTICAL_ROI_PAD='24',
    SCOUT_FUEL_DIAMETER_M='.15', SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR='.55',
    SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU='.90')
from scout.detectors import Detection
from scout.stereo import StereoDepth


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    m = median(values)
    return median([abs(x - m) for x in values]) if values else None


def match(boxes):
    result = [None, None]
    for box in sorted(boxes, key=lambda x: x.confidence, reverse=True):
        x = (box.x1 + box.x2) / 2
        y = (box.y1 + box.y2) / 2
        distance, index = min((math.hypot(x - tx, y - ty), i)
                              for i, (tx, ty) in enumerate(TARGETS))
        if distance <= 60 and result[index] is None:
            result[index] = box
    return result


def make_boxes(row):
    return [Detection(int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2']),
                      0, float(b['confidence']))
            for b in row['detections'] if b['confidence'] >= CONF]


output = {'method': 'Sample every twentieth decoded frame from both clips and run identical confidence-0.05 boxes through separate instances of the runtime StereoDepth implementation, with horizontal ROI disabled/enabled. Compare valid 3D pairs, range and pair-spacing repeatability, same-frame range deltas, full estimate latency, and gate rejections.',
          'sample_stride_frames': STRIDE,
          'settings': {'confidence': CONF, 'vertical_pad': 24,
                       'disparities': 128, 'match_scale': .5,
                       'size_gate_relative_error': .55},
          'limitations': 'Sampled static two-ball clips only. This verifies code-path parity and repeatability; it does not establish absolute depth accuracy or edge-of-image behavior.',
          'results': []}

for clip in DATA['results']:
    estimators = {}
    for enabled in (False, True):
        os.environ['SCOUT_STEREO_HORIZONTAL_ROI'] = '1' if enabled else '0'
        estimators[enabled] = StereoDepth()
    caps = cv2.VideoCapture(str(ROOT / clip['video']))
    records = {enabled: {'ranges': [[], []], 'spacing': [], 'latency_ms': [],
                         'valid_pairs': 0, 'gate_rejected': 0,
                         'duplicates_rejected': 0, 'count_hist': {0: 0, 1: 0, 2: 0}}
               for enabled in (False, True)}
    deltas = [[], []]
    for index, row in enumerate(clip['per_frame']):
        ok, frame = caps.read()
        if not ok:
            break
        if index % STRIDE:
            continue
        outputs = {}
        for enabled, estimator in estimators.items():
            boxes = make_boxes(row)
            started = time.perf_counter()
            estimator.estimate(frame[:, :1600], frame[:, 1600:], boxes)
            rec = records[enabled]
            rec['latency_ms'].append((time.perf_counter() - started) * 1000)
            filtered = estimator.filter_fuel_size_inconsistent(boxes)
            rec['gate_rejected'] += estimator.last_size_gate_rejected
            rec['duplicates_rejected'] += estimator.last_size_gate_duplicate_rejected
            targets = match(filtered)
            rec['count_hist'][sum(box is not None for box in targets)] += 1
            outputs[enabled] = targets
            if all(box is not None and box.distance_m is not None for box in targets):
                rec['valid_pairs'] += 1
                xyz = []
                _, _, _, kl, r1, p1 = estimator.maps
                for target, box in enumerate(targets):
                    z = float(box.distance_m)
                    rec['ranges'][target].append(z)
                    center = np.array([[[float(box.x1 + box.x2) / 2,
                                         float(box.y1 + box.y2) / 2]]], np.float32)
                    uv = cv2.undistortPoints(
                        center, kl, estimator.calibration['left_distortion'],
                        R=r1, P=p1)[0, 0]
                    xyz.append(((uv[0] - p1[0, 2]) * z / p1[0, 0],
                                (uv[1] - p1[1, 2]) * z / p1[1, 1], z))
                rec['spacing'].append(math.dist(*xyz))
        if all(box is not None and box.distance_m is not None
               for boxes in outputs.values() for box in boxes):
            for target in range(2):
                deltas[target].append(
                    outputs[True][target].distance_m - outputs[False][target].distance_m)
    caps.release()
    clip_result = {'scene': clip['scene'], 'configs': []}
    for enabled in (False, True):
        rec = records[enabled]
        lat = rec['latency_ms']
        config = {'horizontal_roi_enabled': enabled,
                  'valid_pair_frames': rec['valid_pairs'],
                  'target_count_histogram': {str(k): v for k, v in rec['count_hist'].items()},
                  'range_by_target_m': [{'n': len(v), 'median': round(median(v), 4),
                                         'mad': round(mad(v), 4)} if v else None
                                        for v in rec['ranges']],
                  'pair_spacing_m': {'n': len(rec['spacing']),
                                     'median': round(median(rec['spacing']), 4),
                                     'mad': round(mad(rec['spacing']), 4)}
                  if rec['spacing'] else None,
                  'full_estimate_ms_median': round(median(lat), 2) if lat else None,
                  'full_estimate_ms_p95': round(sorted(lat)[int(.95 * (len(lat) - 1))], 2)
                  if lat else None,
                  'gate_rejected': rec['gate_rejected'],
                  'duplicates_rejected': rec['duplicates_rejected']}
        clip_result['configs'].append(config)
        print(clip['scene'], config)
    clip_result['same_frame_range_delta_roi_minus_full_m'] = [
        {'n': len(values), 'median': round(median(values), 4),
         'max_abs': round(max(abs(x) for x in values), 4)} if values else None
        for values in deltas]
    output['results'].append(clip_result)

path = ROOT / 'media/frc-fuel-stereo-horizontal-roi-runtime-check.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print('saved', path)
