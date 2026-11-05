#!/usr/bin/env python3
"""Evaluate camera-relative piece spacing and simple Euclidean cluster radii."""
import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / 'media/calibration/stereo_calibration_live_order_candidate.yml'
INPUTS = (
    ('earlier', 'media/frc-2026-live-two-fuel-60fps-cfr-aspect2-replay-detections.jsonl'),
    ('corrected-settings', 'media/frc-2026-fuel-camera-settings-fixed-60fps-aspect2-replay-detections.jsonl'),
)
TARGETS = ((808 / 1600, 589 / 1200), (902 / 1600, 591 / 1200))
MATCH_RADIUS_PX = 60
CLUSTER_RADII_M = (0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30)


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    m = median(values)
    return median([abs(x - m) for x in values]) if values else None


fs = cv2.FileStorage(str(CALIBRATION), cv2.FILE_STORAGE_READ)
k = fs.getNode('left_camera_matrix').mat()
d = fs.getNode('left_distortion').mat()
rotation = fs.getNode('rotation').mat()
translation = fs.getNode('translation').mat()
kr = fs.getNode('right_camera_matrix').mat()
dr = fs.getNode('right_distortion').mat()
fs.release()
r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(k, d, kr, dr, (1600, 1200), rotation, translation,
                                               flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

results = []
for scene, relative_path in INPUTS:
    per_target = [[], []]
    pair_rows = []
    events = [json.loads(line) for line in (ROOT / relative_path).read_text().splitlines()
              if line.strip()]
    for event in events:
        chosen = [None, None]
        for box in event.get('boxes', []):
            if box.get('distance_m') is None:
                continue
            cx, cy = box['x'] + box['w'] / 2, box['y'] + box['h'] / 2
            dists = [math.hypot((cx - tx) * 1600, (cy - ty) * 1200) for tx, ty in TARGETS]
            error, i = min((v, i) for i, v in enumerate(dists))
            if error <= MATCH_RADIUS_PX and (chosen[i] is None or
                                             box['confidence'] > chosen[i]['box']['confidence']):
                chosen[i] = {'box': box, 'cx': cx, 'cy': cy}
        if not all(chosen):
            continue
        points = []
        for item in chosen:
            box, cx, cy = item['box'], item['cx'], item['cy']
            pixel = np.array([[[cx * 1600, cy * 1200]]], dtype=np.float32)
            rectified = cv2.undistortPoints(pixel, k, d, R=r1, P=p1)[0, 0]
            z = float(box['distance_m'])
            x = float((rectified[0] - p1[0, 2]) * z / p1[0, 0])
            y = float((rectified[1] - p1[1, 2]) * z / p1[1, 1])
            point = {'x_m': x, 'y_m': y, 'z_m': z,
                     'confidence': box['confidence'], 'center_px': [cx * 1600, cy * 1200]}
            points.append(point)
            per_target[len(points) - 1].append(point)
        delta = [points[1][axis] - points[0][axis] for axis in ('x_m', 'y_m', 'z_m')]
        xy_distance = math.hypot(delta[0], delta[1])
        xyz_distance = math.sqrt(sum(x * x for x in delta))
        pair_rows.append({'elapsed_s': event.get('elapsed_s'), 'delta_x_m': delta[0],
                          'delta_y_m': delta[1], 'delta_z_m': delta[2],
                          'xy_distance_m': xy_distance, 'xyz_distance_m': xyz_distance})
    spacing = [row['xyz_distance_m'] for row in pair_rows]
    xy_spacing = [row['xy_distance_m'] for row in pair_rows]
    depth_gap = [abs(row['delta_z_m']) for row in pair_rows]
    radii = []
    for threshold in CLUSTER_RADII_M:
        radii.append({'radius_m': threshold,
                      'merged_pair_frames': sum(row['xyz_distance_m'] <= threshold for row in pair_rows),
                      'pair_frames': len(pair_rows),
                      'merged_fraction': round(sum(row['xyz_distance_m'] <= threshold
                                                   for row in pair_rows) / len(pair_rows), 4)
                      if pair_rows else 0})
    lateral_radii = []
    for threshold in CLUSTER_RADII_M:
        lateral_radii.append({'radius_m': threshold,
                              'merged_pair_frames': sum(row['xy_distance_m'] <= threshold for row in pair_rows),
                              'pair_frames': len(pair_rows),
                              'merged_fraction': round(sum(row['xy_distance_m'] <= threshold
                                                           for row in pair_rows) / len(pair_rows), 4)
                              if pair_rows else 0})
    coordinates = []
    for point_i, samples in enumerate(per_target):
        coordinates.append({
            'piece': point_i + 1, 'valid_points': len(samples),
            'median_camera_xyz_m': {axis: round(median([p[axis] for p in samples]), 3)
                                    for axis in ('x_m', 'y_m', 'z_m')} if samples else None,
            'median_confidence': round(median([p['confidence'] for p in samples]), 3) if samples else None,
        })
    results.append({
        'scene': scene, 'video': relative_path.replace('-aspect2-replay-detections.jsonl', '.mp4'),
        'paired_valid_3d_frames': len(pair_rows), 'camera_frame_coordinates': coordinates,
        'lateral_xy_spacing_m': {'median': round(median(xy_spacing), 3),
                                 'mad': round(mad(xy_spacing), 3)} if xy_spacing else None,
        'depth_gap_m': {'median': round(median(depth_gap), 3),
                        'mad': round(mad(depth_gap), 3)} if depth_gap else None,
        'full_xyz_spacing_m': {'median': round(median(spacing), 3),
                               'mad': round(mad(spacing), 3),
                               'min': round(min(spacing), 3), 'max': round(max(spacing), 3)}
                              if spacing else None,
        'lateral_xy_cluster_radius_sweep': lateral_radii,
        'cluster_radius_sweep': radii,
    })
output = {
    'method': 'Unproject each detector-box center through candidate stereo rectification and intrinsics, using the service stereo range as Z. Compare the two known visible balls with simple Euclidean thresholds. Points are camera-relative and box-center based, not robot-frame or verified ball-center truth.',
    'limitations': 'Only the same two stationary balls are present in these saved clips. Treat them as one expected visual group based on their adjacent placement; this is not a labeled field-cluster benchmark. Absolute coordinates remain subject to range/calibration error.',
    'calibration': {'file': 'media/calibration/stereo_calibration_live_order_candidate.yml',
                    'baseline_m': 0.06303240043010157,
                    'rectified_focal_px': float(p1[0, 0]),
                    'rectified_principal_px': [float(p1[0, 2]), float(p1[1, 2])]},
    'profile': {'model': 'FRC 9550 YOLO26n TensorRT 1280', 'confidence': 0.10,
                'max_box_aspect_ratio': 2.0, 'stereo': 'SGBM block 5 / uniqueness 8'},
    'results': results,
}
(ROOT / 'media/frc-two-ball-cluster-radius-sweep.json').write_text(json.dumps(output, indent=2) + '\n')
for result in results:
    print(result['scene'], result['full_xyz_spacing_m'], result['lateral_xy_spacing_m'], result['depth_gap_m'])
    print(result['cluster_radius_sweep'])
