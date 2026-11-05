#!/usr/bin/env python3
"""Check stereo depth and 3D grouping for an luma-adaptive confidence rule."""
import json
import math
import os
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update({
    'SCOUT_CLASS_NAMES': '["ball"]', 'SCOUT_STEREO_ENABLED': '1',
    'SCOUT_STEREO_RECTIFY': '1',
    'SCOUT_STEREO_CALIBRATION': str(ROOT / 'media/calibration/stereo_calibration_live_order_candidate.yml'),
    'SCOUT_STEREO_BASELINE_M': '0.0635', 'SCOUT_STEREO_DISPARITY_SCALE': '1',
    'SCOUT_STEREO_MATCH_SCALE': '0.5', 'SCOUT_STEREO_NUM_DISPARITIES': '128',
    'SCOUT_STEREO_VERTICAL_ROI_PAD': '24', 'SCOUT_STEREO_HORIZONTAL_ROI': '0',
    'SCOUT_STEREO_ADAPTIVE_DISPARITY_WIDTH': '0', 'SCOUT_FUEL_DIAMETER_M': '0.15',
    'SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR': '0.55',
    'SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU': '0.90',
})
from scout.detectors import Detection
from scout.stereo import StereoDepth

CONF_FIXED, CONF_LOW, CONF_HIGH, LUMA_CUTOFF = 0.05, 0.05, 0.075, 18.0
STRIDE = 10
TARGETS = ((808.0, 589.0), (902.0, 591.0))


def make_boxes(items, confidence):
    return [Detection(int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2']),
                      0, float(b['confidence'])) for b in items
            if b['confidence'] >= confidence]


def select(boxes):
    chosen = [None, None]
    extras = 0
    for box in sorted(boxes, key=lambda b: b.confidence, reverse=True):
        cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        distance, index = min((math.dist((cx, cy), target), i)
                              for i, target in enumerate(TARGETS))
        if distance <= 60 and chosen[index] is None:
            chosen[index] = box
        else:
            extras += 1
    return chosen, extras


def summary(values):
    if not values:
        return {'n': 0}
    median = statistics.median(values)
    return {'n': len(values), 'median_m': round(median, 5),
            'mad_m': round(statistics.median(abs(v - median) for v in values), 5),
            'p95_abs_dev_m': round(sorted(abs(v - median) for v in values)[
                math.ceil(.95 * len(values)) - 1], 5)}


def camera_point(box, estimator):
    _, _, _, camera_matrix, rectification, projection = estimator.maps
    center = np.array([[[float(box.x1 + box.x2) / 2,
                         float(box.y1 + box.y2) / 2]]], dtype=np.float32)
    u, v = cv2.undistortPoints(center, camera_matrix,
        estimator.calibration['left_distortion'], R=rectification,
        P=projection)[0, 0]
    z = float(box.distance_m)
    return ((u - projection[0, 2]) * z / projection[0, 0],
            (v - projection[1, 2]) * z / projection[1, 1], z)


source = json.loads((ROOT / 'media/frc-fuel-low-confidence-evaluation.json').read_text())
luma_data = json.loads((ROOT / 'media/frc-fuel-luma-adaptive-confidence-evaluation.json').read_text())
output = {
    'method': ('Every tenth frame, compare fixed 0.05 confidence with a rule '
               'using 0.05 at mean luma <=18 and 0.075 otherwise. Run both '
               'candidate sets through rectified 128-disparity / 24 px band '
               'stereo, the active 55% size gate, target matching and 26 cm '
               '3D clustering.'),
    'limitations': ('60 samples per clip from the same two stationary videos. '
                    'Fixed target centers; 26 cm cluster radius is tuned to '
                    'these adjacent balls. No independent range truth or '
                    'separate nearby clusters.'),
    'settings': {'stride_frames': STRIDE, 'fixed_confidence': CONF_FIXED,
                 'low_luma_confidence': CONF_LOW, 'high_luma_confidence': CONF_HIGH,
                 'luma_cutoff': LUMA_CUTOFF, 'cluster_radius_m': 0.26,
                 'size_gate_relative_tolerance': 0.55},
    'results': [],
}

for clip, luma_clip in zip(source['results'], luma_data['results']):
    capture = cv2.VideoCapture(str(ROOT / clip['video']))
    records = {name: {'ranges': [[], []], 'pairs': 0, 'clusters': [],
                      'extras': 0, 'gate_rejected': 0, 'frame_count': 0}
               for name in ('fixed_0.05', 'adaptive_luma_18')}
    range_deltas = [[], []]
    lumas = luma_clip['per_frame_luma']
    for index, detections in enumerate(clip['per_frame']):
        ok, frame = capture.read()
        if not ok:
            break
        if index % STRIDE:
            continue
        image_luma = lumas[index]
        confidence = CONF_LOW if image_luma <= LUMA_CUTOFF else CONF_HIGH
        candidates = {
            'fixed_0.05': make_boxes(detections, CONF_FIXED),
            'adaptive_luma_18': make_boxes(detections, confidence),
        }
        for name, boxes in candidates.items():
            estimator = StereoDepth()
            estimator.estimate(frame[:, :1600], frame[:, 1600:], boxes)
            boxes = estimator.filter_fuel_size_inconsistent(boxes)
            record = records[name]
            record['frame_count'] += 1
            record['gate_rejected'] += estimator.last_size_gate_rejected
            record['extras'] += select(boxes)[1]
            targets, _ = select(boxes)
            valid = [box for box in targets if box is not None and box.distance_m is not None]
            if len(valid) == 2:
                record['pairs'] += 1
                distance = math.dist(camera_point(targets[0], estimator),
                                     camera_point(targets[1], estimator))
                record['clusters'].append(1 if distance <= 0.26 else 2)
            for target, box in enumerate(targets):
                if box is not None and box.distance_m is not None:
                    record['ranges'][target].append(float(box.distance_m))
        for target in range(2):
            a = records['fixed_0.05']['ranges'][target]
            b = records['adaptive_luma_18']['ranges'][target]
            if a and b:
                range_deltas[target].append(b[-1] - a[-1])
    capture.release()
    configs = {}
    for name, record in records.items():
        configs[name] = {
            'sampled_frames': record['frame_count'],
            'valid_two_target_depth_frames': record['pairs'],
            'target_ranges_m': [summary(values) for values in record['ranges']],
            'cluster_count_histogram_at_0.26m': {
                str(count): record['clusters'].count(count) for count in (1, 2)},
            'off_target_boxes_after_gate': record['extras'],
            'size_gate_rejected': record['gate_rejected'],
        }
    output['results'].append({'scene': clip['scene'], 'video': clip['video'],
                              'configurations': configs,
                              'range_delta_adaptive_minus_fixed_m': [summary(x) for x in range_deltas]})
    print(clip['scene'], json.dumps(output['results'][-1]))

path = ROOT / 'media/frc-fuel-luma-adaptive-stereo-evaluation.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print('saved', path)
