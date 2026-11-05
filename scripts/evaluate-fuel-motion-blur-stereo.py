#!/usr/bin/env python3
"""Test detector and stereo performance under matched stereo motion blur."""
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
os.environ.update({
    'SCOUT_CLASS_NAMES': '["ball"]', 'SCOUT_MAX_BOX_ASPECT_RATIO': '2.0',
    'SCOUT_STEREO_ENABLED': '1', 'SCOUT_STEREO_RECTIFY': '1',
    'SCOUT_STEREO_CALIBRATION': str(ROOT / 'media/calibration/stereo_calibration_live_order_candidate.yml'),
    'SCOUT_STEREO_BASELINE_M': '0.0635', 'SCOUT_STEREO_DISPARITY_SCALE': '1',
    'SCOUT_STEREO_MATCH_SCALE': '0.5', 'SCOUT_STEREO_NUM_DISPARITIES': '128',
    'SCOUT_STEREO_VERTICAL_ROI_PAD': '24', 'SCOUT_STEREO_HORIZONTAL_ROI': '0',
    'SCOUT_STEREO_ADAPTIVE_DISPARITY_WIDTH': '0', 'SCOUT_FUEL_DIAMETER_M': '0.15',
    'SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR': '0.55',
    'SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU': '0.90',
})
from scout.detectors import TensorRTDetector
from scout.stereo import StereoDepth

VIDEOS = (
    ('earlier', ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr.mp4'),
    ('corrected-settings', ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'),
)
ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
TARGETS = ((808.0, 589.0), (902.0, 591.0))
MATCH_RADIUS = 60
KERNELS = (1, 3, 5, 9)
FRAME_INDICES = np.linspace(30, 590, 30, dtype=int).tolist()


def blur(image, length):
    if length == 1:
        return image
    kernel = np.zeros((1, length), np.float32)
    kernel[0] = 1.0 / length
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def match(detections):
    chosen = [None, None]
    extras = 0
    for box in sorted(detections, key=lambda item: item.confidence, reverse=True):
        center = ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)
        distance, index = min((math.dist(center, target), i)
                              for i, target in enumerate(TARGETS))
        if distance <= MATCH_RADIUS and chosen[index] is None:
            chosen[index] = box
        else:
            extras += 1
    return chosen, extras


def xyz(box, estimator):
    _, _, _, camera_matrix, rotation, projection = estimator.maps
    center = np.array([[[float(box.x1 + box.x2) / 2,
                         float(box.y1 + box.y2) / 2]]], dtype=np.float32)
    u, v = cv2.undistortPoints(center, camera_matrix,
        estimator.calibration['left_distortion'], R=rotation,
        P=projection)[0, 0]
    z = float(box.distance_m)
    return ((u - projection[0, 2]) * z / projection[0, 0],
            (v - projection[1, 2]) * z / projection[1, 1], z)


def summary(values):
    if not values:
        return {'n': 0}
    median = statistics.median(values)
    return {'n': len(values), 'median': round(median, 5),
            'mad': round(statistics.median(abs(v - median) for v in values), 5),
            'p95_abs_deviation': round(sorted(abs(v - median) for v in values)[
                math.ceil(.95 * len(values)) - 1], 5)}


detector = TensorRTDetector(str(ENGINE), confidence=0.05)
estimators = {length: StereoDepth() for length in KERNELS}
warm = cv2.VideoCapture(str(VIDEOS[0][1]))
warm_ok, warm_frame = warm.read()
warm.release()
if warm_ok:
    detector.infer(warm_frame[:, :1600])

output = {
    'method': ('Run fresh FRC 1280x960 TensorRT inference and rectified stereo '
               'depth on 30 evenly spaced frames per clip, using matched '
               'horizontal box-filter blur in left and right images. Apply the '
               'active 55% size/depth gate and group the known adjacent pair at '
               '26 cm. Compare ranges to unblurred same-frame estimates.'),
    'settings': {'engine': str(ENGINE.relative_to(ROOT)), 'confidence': 0.05,
                 'blur_kernel_horizontal_px': KERNELS, 'sample_frames': FRAME_INDICES,
                 'match_radius_px': MATCH_RADIUS, 'cluster_radius_m': 0.26,
                 'size_gate_relative_tolerance': 0.55,
                 'stereo': {'baseline_m': 0.0635, 'num_disparities': 128,
                            'match_scale': 0.5, 'vertical_roi_pad': 24}},
    'limitations': ('Synthetic uniform horizontal blur on stationary clips, not '
                    'measured object/camera motion blur. Only 30 frames per clip, '
                    'two fixed targets, and no absolute range truth.'),
    'results': [],
}

try:
    for scene, video in VIDEOS:
        cap = cv2.VideoCapture(str(video))
        per_config = {length: {'count_hist': {0: 0, 1: 0, 2: 0}, 'exact': 0,
            'extras': 0, 'gate_rejected': 0, 'ranges': [[], []], 'pair_spacing': [],
            'cluster_hist': {1: 0, 2: 0}, 'range_delta': [[], []],
            'detector_ms': [], 'stereo_ms': [], 'blur_ms': []} for length in KERNELS}
        index, sample_index = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if sample_index >= len(FRAME_INDICES) or index != FRAME_INDICES[sample_index]:
                index += 1
                continue
            left, right = frame[:, :1600], frame[:, 1600:]
            baseline_ranges = None
            for length in KERNELS:
                started = time.perf_counter()
                blurred_left = blur(left, length)
                blurred_right = blur(right, length)
                blur_ms = (time.perf_counter() - started) * 1000
                detections, detector_ms = detector.infer(blurred_left)
                estimator = estimators[length]
                started = time.perf_counter()
                estimator.estimate(blurred_left, blurred_right, detections)
                detections = estimator.filter_fuel_size_inconsistent(detections)
                stereo_ms = (time.perf_counter() - started) * 1000
                targets, extras = match(detections)
                record = per_config[length]
                valid_targets = [box for box in targets if box is not None and box.distance_m is not None]
                count = len(valid_targets)
                record['count_hist'][count] += 1
                record['exact'] += int(count == 2 and extras == 0)
                record['extras'] += extras
                record['gate_rejected'] += estimator.last_size_gate_rejected
                record['detector_ms'].append(detector_ms)
                record['stereo_ms'].append(stereo_ms)
                record['blur_ms'].append(blur_ms)
                for target, box in enumerate(targets):
                    if box is not None and box.distance_m is not None:
                        record['ranges'][target].append(float(box.distance_m))
                if count == 2:
                    points = [xyz(targets[target], estimator) for target in range(2)]
                    spacing = math.dist(points[0], points[1])
                    record['pair_spacing'].append(spacing)
                    record['cluster_hist'][1 if spacing <= 0.26 else 2] += 1
                if length == 1:
                    baseline_ranges = [box.distance_m if box is not None else None
                                       for box in targets]
                elif baseline_ranges is not None:
                    for target, box in enumerate(targets):
                        if (box is not None and box.distance_m is not None and
                                baseline_ranges[target] is not None):
                            record['range_delta'][target].append(
                                float(box.distance_m - baseline_ranges[target]))
            sample_index += 1
            index += 1
        cap.release()
        configs = {}
        for length, record in per_config.items():
            configs[str(length)] = {
                'frames': len(FRAME_INDICES), 'depth_target_count_histogram': {
                    str(k): v for k, v in record['count_hist'].items()},
                'exact_two_no_extra_frames': record['exact'],
                'off_target_boxes_after_gate': record['extras'],
                'size_gate_rejected': record['gate_rejected'],
                'range_by_target_m': [summary(v) for v in record['ranges']],
                'pair_spacing_m': summary(record['pair_spacing']),
                'cluster_count_histogram_at_0.26m': {
                    str(k): v for k, v in record['cluster_hist'].items()},
                'range_delta_vs_unblurred_m_by_target': [summary(v)
                                                        for v in record['range_delta']],
                'latency_ms': {name: round(statistics.median(record[field]), 2)
                               if record[field] else None for name, field in (
                                   ('detector_median', 'detector_ms'),
                                   ('stereo_median', 'stereo_ms'),
                                   ('blur_median', 'blur_ms'))},
            }
        output['results'].append({'scene': scene, 'video': str(video.relative_to(ROOT)),
                                  'configurations': configs})
        print(scene, json.dumps(configs))
finally:
    detector.close()

path = ROOT / 'media/frc-fuel-motion-blur-stereo-evaluation.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print('saved', path)
