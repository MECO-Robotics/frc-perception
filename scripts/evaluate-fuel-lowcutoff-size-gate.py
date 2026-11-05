#!/usr/bin/env python3
"""Evaluate lower detector cutoffs combined with the FUEL size/stereo gate."""
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
INPUT = ROOT / 'media/frc-fuel-low-confidence-evaluation.json'
CAL = 'media/calibration/stereo_calibration_live_order_candidate.yml'
THRESHOLDS = (0.02, 0.03, 0.05)
TARGETS = ((808, 589), (902, 591))
MATCH_RADIUS = 60
RADIUS_M = 0.26

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
    center = median(values)
    return median([abs(value - center) for value in values]) if values else None


def summary(values):
    if not values:
        return None
    return {'n': len(values), 'median_m': round(median(values), 4),
            'mad_m': round(mad(values), 4),
            'p05_m': round(sorted(values)[int(.05 * (len(values) - 1))], 4),
            'p95_m': round(sorted(values)[int(.95 * (len(values) - 1))], 4)}


def score(boxes):
    targets = [None, None]
    extras = 0
    for box in sorted(boxes, key=lambda item: item.confidence, reverse=True):
        cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        distance, index = min((math.hypot(cx - x, cy - y), i)
                              for i, (x, y) in enumerate(TARGETS))
        if distance <= MATCH_RADIUS and targets[index] is None:
            targets[index] = box
        else:
            extras += 1
    return targets, extras


data = json.loads(INPUT.read_text())
stereo = StereoDepth()
output = {
    'method': 'Replay the saved low-confidence TensorRT candidates through calibrated 24 px vertical-band SGBM. Compare detector cutoffs 0.02/0.03/0.05 both raw and after a 55% relative apparent-size/stereo range gate plus 0.90 IoU and 7.5 cm 3D duplicate suppression. Match against the two manually labeled centers and report their stereo range and separation repeatability.',
    'thresholds': list(THRESHOLDS), 'size_gate_relative_tolerance': .55,
    'duplicate_iou': .90, 'cluster_radius_m': RADIUS_M,
    'limitations': 'Two fixed stationary balls at one pose and range; target labels use fixed 60 px windows. Nominal FUEL diameter is a proxy, not independently measured range truth. The saved candidate export begins at confidence 0.005 and shares the active model/post-NMS outputs.',
    'results': []}

for clip in data['results']:
    capture = cv2.VideoCapture(str(ROOT / clip['video']))
    configurations = []
    for threshold in THRESHOLDS:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        raw_counts = {0: 0, 1: 0, 2: 0}
        gated_counts = {0: 0, 1: 0, 2: 0}
        raw_extras = gated_extras = gate_rejected = duplicate_rejected = 0
        ranges = [[], []]
        spacings = []
        valid_pairs = merged_pairs = 0
        matcher_ms = []
        for index, row in enumerate(clip['per_frame']):
            ok, frame = capture.read()
            if not ok:
                break
            candidates = [box for box in row if box['confidence'] >= threshold]
            if not candidates:
                raw_counts[0] += 1
                gated_counts[0] += 1
                continue
            boxes = [Detection(int(item['x1']), int(item['y1']),
                               int(item['x2']), int(item['y2']), 0,
                               float(item['confidence'])) for item in candidates]
            left, right = frame[:, :1600], frame[:, 1600:]
            start = cv2.getTickCount()
            stereo.estimate(left, right, boxes)
            matcher_ms.append((cv2.getTickCount() - start) * 1000 / cv2.getTickFrequency())
            raw_targets, extras = score(boxes)
            raw_count = sum(box is not None for box in raw_targets)
            raw_counts[raw_count] += 1
            raw_extras += extras
            filtered = stereo.filter_fuel_size_inconsistent(boxes)
            gate_rejected += stereo.last_size_gate_rejected
            duplicate_rejected += stereo.last_size_gate_duplicate_rejected
            targets, extras = score(filtered)
            count = sum(box is not None for box in targets)
            gated_counts[count] += 1
            gated_extras += extras
            for target, box in enumerate(targets):
                if box is not None and box.distance_m is not None:
                    ranges[target].append(float(box.distance_m))
            if all(box is not None and box.distance_m is not None for box in targets):
                valid_pairs += 1
                # Use calibrated camera-space XYZ, not image-space pixel distance.
                _, _, _, kl, r1, p1 = stereo.maps
                points = []
                for box in targets:
                    center = np.array(
                        [[[(box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2]]],
                        dtype=np.float32)
                    uv = cv2.undistortPoints(
                        center, kl, stereo.calibration['left_distortion'],
                        R=r1, P=p1)[0, 0]
                    z = float(box.distance_m)
                    points.append(((uv[0] - p1[0, 2]) * z / p1[0, 0],
                                   (uv[1] - p1[1, 2]) * z / p1[1, 1], z))
                separation = math.dist(*points)
                spacings.append(separation)
                merged_pairs += separation <= RADIUS_M
        configurations.append({
            'confidence': threshold,
            'raw_target_count_histogram': {str(k): v for k, v in raw_counts.items()},
            'raw_frames_with_two_target_matches': raw_counts[2],
            'raw_off_target_detections': raw_extras,
            'gated_target_count_histogram': {str(k): v for k, v in gated_counts.items()},
            'gated_frames_with_two_target_matches': gated_counts[2],
            'gated_off_target_detections': gated_extras,
            'gate_rejected_boxes': gate_rejected,
            'duplicate_rejected_boxes': duplicate_rejected,
            'valid_3d_pair_frames': valid_pairs,
            'range_by_target': [summary(values) for values in ranges],
            'pair_spacing': summary(spacings),
            'single_cluster_frames_at_26cm': merged_pairs,
            'matcher_ms_median': round(median(matcher_ms), 2) if matcher_ms else None,
            'matcher_ms_p95': round(sorted(matcher_ms)[int(.95 * (len(matcher_ms) - 1))], 2)
            if matcher_ms else None})
        print(clip['scene'], configurations[-1])
    capture.release()
    output['results'].append({'scene': clip['scene'], 'video': clip['video'],
                              'frames': clip['frames'], 'configurations': configurations})

path = ROOT / 'media/frc-fuel-lowcutoff-size-gate-sweep.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print('saved', path)
