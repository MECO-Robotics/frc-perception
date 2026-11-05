#!/usr/bin/env python3
"""Compare Hough-diameter and SGBM range fusion for FUEL positions."""
import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOUGH = json.loads((ROOT / 'media/frc-fuel-known-diameter-range-evaluation.json').read_text())
DETECTIONS = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
STEREO = json.loads((ROOT / 'media/frc-fuel-dense-stereo-evaluation.json').read_text())
CAL = ROOT / 'media/calibration/stereo_calibration_live_order_candidate.yml'
TARGETS = ((808, 589), (902, 591))
MATCH_RADIUS_PX = 60
PARAM2_VALUES = (20, 25)
STEREO_WEIGHTS = (0.0, .25, .5, .75, 1.0)
CLUSTER_RADIUS_M = .26

fs = cv2.FileStorage(str(CAL), cv2.FILE_STORAGE_READ)
kl = fs.getNode('left_camera_matrix').mat()
dl = fs.getNode('left_distortion').mat()
kr = fs.getNode('right_camera_matrix').mat()
dr = fs.getNode('right_distortion').mat()
rotation = fs.getNode('rotation').mat()
translation = fs.getNode('translation').mat()
fs.release()
r1, _, p1, _, *_ = cv2.stereoRectify(
    kl, dl, kr, dr, (1600, 1200), rotation, translation,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
fx, fy = float(p1[0, 0]), float(p1[1, 1])
cx, cy = float(p1[0, 2]), float(p1[1, 2])


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(value - center) for value in values]) if values else None


def summary(values):
    if not values:
        return {'n': 0, 'median_m': None, 'mad_m': None,
                'p05_m': None, 'p95_m': None}
    ordered = sorted(values)
    return {'n': len(ordered), 'median_m': round(median(ordered), 4),
            'mad_m': round(mad(ordered), 4),
            'p05_m': round(ordered[int(.05 * (len(ordered) - 1))], 4),
            'p95_m': round(ordered[int(.95 * (len(ordered) - 1))], 4)}


def match(detections):
    targets = [None, None]
    for box in sorted((item for item in detections if item['confidence'] >= .05),
                      key=lambda item: item['confidence'], reverse=True):
        center = ((box['x1'] + box['x2']) / 2,
                  (box['y1'] + box['y2']) / 2)
        distance, index = min((math.dist(center, point), i)
                              for i, point in enumerate(TARGETS))
        if distance <= MATCH_RADIUS_PX and targets[index] is None:
            targets[index] = box
    return targets


def camera_xyz(box, z):
    center = np.array([[[float(box['x1'] + box['x2']) / 2,
                         float(box['y1'] + box['y2']) / 2]]], np.float32)
    uv = cv2.undistortPoints(center, kl, dl, R=r1, P=p1)[0, 0]
    return np.array(((uv[0] - cx) * z / fx,
                     (uv[1] - cy) * z / fy, z), dtype=np.float64)


output = {
    'method': ('For frames where both Hough and SGBM ranges exist, compute '
               'fused Z = w*stereo + (1-w)*Hough diameter range. Unproject the '
               'same detector-box centers to camera XYZ, then compare per-target '
               'range repeatability and the fraction of the known adjacent pair '
               'inside a 26 cm 3D cluster radius.'),
    'stereo_weights': STEREO_WEIGHTS,
    'hough_param2_values': PARAM2_VALUES,
    'cluster_radius_m': CLUSTER_RADIUS_M,
    'limitations': ('No tape-measured range is available. The known-diameter '
                    'Hough range and stereo range are both imperfect proxies; '
                    'their agreement or repeatability cannot establish which '
                    'fusion weight is more accurate. The cluster labels assume '
                    'the two visible balls form one cluster and do not measure '
                    'false merges with unrelated pieces.'),
    'results': [],
}

for hough_clip, detector_clip, stereo_clip in zip(
        HOUGH['results'], DETECTIONS['results'], STEREO['results']):
    if not (hough_clip['scene'] == detector_clip['scene'] == stereo_clip['scene']):
        raise ValueError('Hough, detector, and stereo scenes are not aligned')
    hough_frames = {item['frame']: item for item in hough_clip['frame_results']}
    config_results = []
    for param2 in PARAM2_VALUES:
        frame_inputs = []
        for index, (det_row, stereo_row) in enumerate(
                zip(detector_clip['per_frame'], stereo_clip['frames'])):
            boxes = match(det_row['detections'])
            hough_row = hough_frames[index]
            hough_ranges = [target['hough'][str(param2)]['range_m']
                            if target['hough'][str(param2)] else None
                            for target in hough_row['targets']]
            stereo_ranges = stereo_row['ranges_m']
            if (all(boxes) and all(value is not None for value in hough_ranges + stereo_ranges)):
                frame_inputs.append((boxes, hough_ranges, stereo_ranges))
        weights = []
        for weight in STEREO_WEIGHTS:
            per_target = [[], []]
            spacing, z_gap = [], []
            merged = 0
            for boxes, hough_ranges, stereo_ranges in frame_inputs:
                fused = [weight * stereo + (1 - weight) * hough
                         for stereo, hough in zip(stereo_ranges, hough_ranges)]
                for target in range(2):
                    per_target[target].append(fused[target])
                points = [camera_xyz(box, z) for box, z in zip(boxes, fused)]
                distance = math.dist(*points)
                spacing.append(distance)
                z_gap.append(abs(fused[0] - fused[1]))
                merged += int(distance <= CLUSTER_RADIUS_M)
            weights.append({
                'stereo_weight': weight, 'frames': len(frame_inputs),
                'range_by_target_m': [summary(values) for values in per_target],
                'pair_spacing_m': summary(spacing),
                'depth_gap_m': summary(z_gap),
                'one_cluster_frames_at_26cm': merged,
                'one_cluster_fraction_at_26cm': round(merged / len(spacing), 5)
                if spacing else None,
            })
        config_results.append({'hough_param2': param2,
                               'same_frame_samples': len(frame_inputs),
                               'fusion_weights': weights})
    output['results'].append({'scene': hough_clip['scene'],
                              'video': hough_clip['video'],
                              'configurations': config_results})

path = ROOT / 'media/frc-fuel-hough-stereo-fusion-evaluation.json'
path.write_text(json.dumps(output, indent=2) + '\n')
for clip in output['results']:
    print(clip['scene'])
    for config in clip['configurations']:
        print('param2', config['hough_param2'])
        for row in config['fusion_weights']:
            print('w', row['stereo_weight'], 'range MAD',
                  [item['mad_m'] for item in row['range_by_target_m']],
                  'spacing', row['pair_spacing_m'], 'z-gap', row['depth_gap_m']['median_m'],
                  'one cluster@26cm', row['one_cluster_fraction_at_26cm'])
print('saved', path)
