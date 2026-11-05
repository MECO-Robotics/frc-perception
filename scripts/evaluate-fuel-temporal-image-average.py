#!/usr/bin/env python3
"""Compare raw FUEL detection with causal short-window image denoising."""
import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--methods', default='raw,causal_mean_3,causal_mean_5',
                    help='comma-separated raw, causal_mean_3, causal_mean_5, causal_ema_50, causal_ema_75')
parser.add_argument('--output', default='media/frc-fuel-temporal-image-average-sweep.json')
parser.add_argument('--start-frame', type=int, default=60)
parser.add_argument('--end-frame', type=int, default=None, help='exclusive end frame')
parser.add_argument('--stride', type=int, default=6)
args = parser.parse_args()
VIDEOS = (
    ('earlier', 'media/frc-2026-live-two-fuel-60fps-cfr.mp4'),
    ('corrected-settings', 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'))
ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
CAL = 'media/calibration/stereo_calibration_live_order_candidate.yml'
TARGETS = ((808, 589), (902, 591))
MATCH_RADIUS_PX = 60
SAMPLE_STRIDE = args.stride
FIRST_SAMPLE = args.start_frame
if SAMPLE_STRIDE <= 0 or FIRST_SAMPLE < 0:
    parser.error('--stride must be positive and --start-frame nonnegative')
METHODS = tuple(item.strip() for item in args.methods.split(',') if item.strip())
allowed_methods = {'raw', 'causal_mean_3', 'causal_mean_5',
                   'causal_ema_50', 'causal_ema_75'}
if not METHODS or any(method not in allowed_methods for method in METHODS):
    parser.error(f'--methods must be selected from {sorted(allowed_methods)}')

os.environ.update(
    SCOUT_CLASS_NAMES='["ball"]', SCOUT_MAX_BOX_ASPECT_RATIO='2.0',
    SCOUT_STEREO_CALIBRATION=CAL, SCOUT_STEREO_RECTIFY='1',
    SCOUT_STEREO_BASELINE_M='.0635', SCOUT_STEREO_DISPARITY_SCALE='1',
    SCOUT_STEREO_MATCH_SCALE='.5', SCOUT_STEREO_NUM_DISPARITIES='128',
    SCOUT_STEREO_VERTICAL_ROI_PAD='24', SCOUT_FUEL_DIAMETER_M='.15',
    SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR='.55',
    SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU='.90')

from scout.detectors import TensorRTDetector
from scout.stereo import StereoDepth


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(value - center) for value in values]) if values else None


def percentile(values, pct):
    return sorted(values)[int(pct * (len(values) - 1))] if values else None


def summary(values):
    if not values:
        return None
    return {'n': len(values), 'median': round(median(values), 4),
            'mad': round(mad(values), 4),
            'p05': round(percentile(values, .05), 4),
            'p95': round(percentile(values, .95), 4)}


def score(boxes):
    targets = [None, None]
    extras = 0
    errors = [None, None]
    for box in sorted(boxes, key=lambda item: item.confidence, reverse=True):
        cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        distance, index = min((math.hypot(cx - x, cy - y), i)
                              for i, (x, y) in enumerate(TARGETS))
        if distance <= MATCH_RADIUS_PX and targets[index] is None:
            targets[index] = box
            errors[index] = distance
        else:
            extras += 1
    return targets, extras, errors


def point(box, stereo):
    _, _, _, kl, r1, p1 = stereo.maps
    center = np.array([[[float(box.x1 + box.x2) / 2,
                         float(box.y1 + box.y2) / 2]]], np.float32)
    uv = cv2.undistortPoints(center, kl, stereo.calibration['left_distortion'],
                             R=r1, P=p1)[0, 0]
    z = float(box.distance_m)
    return ((uv[0] - p1[0, 2]) * z / p1[0, 0],
            (uv[1] - p1[1, 2]) * z / p1[1, 1], z)


detector = TensorRTDetector(str(ENGINE), confidence=.05)
stereo = StereoDepth()
results = []
try:
    for scene, relative_video in VIDEOS:
        capture = cv2.VideoCapture(str(ROOT / relative_video))
        history = deque(maxlen=5)
        states = {method: {'counts': {0: 0, 1: 0, 2: 0}, 'extras': 0,
                           'exact_two': 0, 'valid_pair_3d': 0,
                           'ranges': [[], []], 'spacing': [],
                           'center_error_px': [[], []], 'confidence': [[], []],
                           'gate_rejected': 0, 'duplicate_rejected': 0}
                  for method in METHODS}
        samples = 0
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            left, right = frame[:, :1600], frame[:, 1600:]
            history.append((left.copy(), right.copy()))
            in_sample_range = (index >= FIRST_SAMPLE and
                               (args.end_frame is None or index < args.end_frame))
            if (in_sample_range and (index - FIRST_SAMPLE) % SAMPLE_STRIDE == 0 and
                    len(history) == 5):
                samples += 1
                left_images = [pair[0] for pair in history]
                right_images = [pair[1] for pair in history]
                inputs = {}
                if 'raw' in METHODS:
                    inputs['raw'] = (left_images[-1], right_images[-1])
                if 'causal_mean_3' in METHODS:
                    inputs['causal_mean_3'] = (
                        np.mean(np.stack(left_images[-3:]), axis=0).astype(np.uint8),
                        np.mean(np.stack(right_images[-3:]), axis=0).astype(np.uint8))
                if 'causal_mean_5' in METHODS:
                    inputs['causal_mean_5'] = (
                        np.mean(np.stack(left_images), axis=0).astype(np.uint8),
                        np.mean(np.stack(right_images), axis=0).astype(np.uint8))
                for method, alpha in (('causal_ema_50', .5), ('causal_ema_75', .75)):
                    if method not in METHODS:
                        continue
                    weights = np.array([alpha * (1 - alpha) ** age
                                        for age in range(5)], dtype=np.float32)
                    weights /= weights.sum()
                    ema_left = sum(image.astype(np.float32) * weight
                                   for image, weight in zip(reversed(left_images), weights))
                    ema_right = sum(image.astype(np.float32) * weight
                                    for image, weight in zip(reversed(right_images), weights))
                    inputs[method] = (np.clip(ema_left, 0, 255).astype(np.uint8),
                                      np.clip(ema_right, 0, 255).astype(np.uint8))
                for method, (image, matching_image) in inputs.items():
                    boxes, _ = detector.infer(image)
                    stereo.estimate(image, matching_image, boxes)
                    boxes = stereo.filter_fuel_size_inconsistent(boxes)
                    state = states[method]
                    state['gate_rejected'] += stereo.last_size_gate_rejected
                    state['duplicate_rejected'] += stereo.last_size_gate_duplicate_rejected
                    targets, extras, errors = score(boxes)
                    count = sum(box is not None for box in targets)
                    state['counts'][count] += 1
                    state['extras'] += extras
                    state['exact_two'] += count == 2 and extras == 0
                    for target, box in enumerate(targets):
                        if box is None:
                            continue
                        state['center_error_px'][target].append(errors[target])
                        state['confidence'][target].append(float(box.confidence))
                        if box.distance_m is not None:
                            state['ranges'][target].append(float(box.distance_m))
                    if all(box is not None and box.distance_m is not None for box in targets):
                        state['valid_pair_3d'] += 1
                        xyz = [point(box, stereo) for box in targets]
                        state['spacing'].append(math.dist(*xyz))
            index += 1
        capture.release()
        configs = []
        for method in METHODS:
            state = states[method]
            configs.append({
                'method': method,
                'count_histogram': {str(k): v for k, v in state['counts'].items()},
                'exact_two_frames': state['exact_two'],
                'off_target_boxes': state['extras'],
                'valid_pair_3d_frames': state['valid_pair_3d'],
                'range_by_target_m': [summary(x) for x in state['ranges']],
                'pair_spacing_m': summary(state['spacing']),
                'center_error_px_by_target': [summary(x) for x in state['center_error_px']],
                'confidence_by_target': [summary(x) for x in state['confidence']],
                'gate_rejected_boxes': state['gate_rejected'],
                'duplicate_rejected_boxes': state['duplicate_rejected']})
            print(scene, configs[-1])
        results.append({'scene': scene, 'video': relative_video,
                        'frames_decoded': index, 'sampled_frames': samples,
                        'sample_stride': SAMPLE_STRIDE,
                        'configurations': configs})
finally:
    detector.close()

output = {
    'method': 'On the same stable sample times from both saved 59.9 fps stereo videos, compare selected causal image filters on synchronized stereo pairs. Run the same FRC YOLO26n 1280x960 FP16 detector, calibrated vertical-band SGBM, then the active nominal-size/stereo gate and duplicate suppression. Score two fixed target centers and camera-space pair spacing.',
    'sample_start_frame': FIRST_SAMPLE, 'sample_end_frame_exclusive': args.end_frame,
    'sample_stride_frames': SAMPLE_STRIDE,
    'detector_confidence': .05, 'size_gate_relative_tolerance': .55,
    'limitations': 'Stationary balls at one pose/range; these clips cannot predict motion ghosting or validate absolute depth. The mean input uses only current and prior frames, so it is causal; the SGBM right image remains the current unaveraged frame. Pair spacing is a repeatability measure, not known distance truth.',
    'results': results}
path = ROOT / args.output
path.write_text(json.dumps(output, indent=2) + '\n')
print('saved', path)
