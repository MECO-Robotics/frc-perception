#!/usr/bin/env python3
"""Re-evaluate 3D pair spacing and cluster radii with Hough-derived Z."""
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
PARAM2_VALUES = (20, 25, 30)
CLUSTER_RADII_M = (.15, .16, .18, .20, .22, .24, .26, .28)

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
                'p05_m': None, 'p95_m': None, 'max_m': None}
    ordered = sorted(values)
    return {'n': len(ordered), 'median_m': round(median(ordered), 4),
            'mad_m': round(mad(ordered), 4),
            'p05_m': round(ordered[int(.05 * (len(ordered) - 1))], 4),
            'p95_m': round(ordered[int(.95 * (len(ordered) - 1))], 4),
            'max_m': round(max(ordered), 4)}


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
                         float(box['y1'] + box['y2']) / 2]]], dtype=np.float32)
    uv = cv2.undistortPoints(center, kl, dl, R=r1, P=p1)[0, 0]
    return np.array(((uv[0] - cx) * z / fx,
                     (uv[1] - cy) * z / fy, z), dtype=np.float64)


output = {
    'method': ('For frames with both detector matches, both Hough-circle '
               'diameter ranges, and both SGBM ranges, unproject the detector '
               'box centers with Hough-derived Z and compare their 3D pair '
               'spacing with the same-frame stereo-Z spacing. Sweep Euclidean '
               'radii and report whether the known two-ball pair merges.'),
    'limitations': ('The 0.15 m nominal diameter is a geometry proxy, not a '
                    'measured range. XY uses detector-box centers; Hough centers '
                    'are not substituted. The clips have one fixed two-ball '
                    'arrangement and no unrelated objects, so false cluster '
                    'merges are not measured.'),
    'cluster_radii_m': CLUSTER_RADII_M,
    'results': [],
}

for hough_clip, detector_clip, stereo_clip in zip(
        HOUGH['results'], DETECTIONS['results'], STEREO['results']):
    if not (hough_clip['scene'] == detector_clip['scene'] == stereo_clip['scene']):
        raise ValueError('Hough, detector, and stereo scenes are not aligned')
    frame_records = {item['frame']: item for item in hough_clip['frame_results']}
    config_results = []
    for param2 in PARAM2_VALUES:
        pair_distance_hough = []
        pair_distance_stereo = []
        z_gap_hough = []
        z_gap_stereo = []
        centroid_hough = []
        merged = {radius: 0 for radius in CLUSTER_RADII_M}
        count = 0
        for index, (det_row, stereo_row) in enumerate(
                zip(detector_clip['per_frame'], stereo_clip['frames'])):
            hough_row = frame_records[index]
            boxes = match(det_row['detections'])
            hough_ranges = [target['hough'][str(param2)]['range_m']
                            if target['hough'][str(param2)] else None
                            for target in hough_row['targets']]
            stereo_ranges = stereo_row['ranges_m']
            if not all(boxes) or not all(value is not None for value in hough_ranges + stereo_ranges):
                continue
            points_hough = [camera_xyz(box, z) for box, z in zip(boxes, hough_ranges)]
            points_stereo = [camera_xyz(box, z) for box, z in zip(boxes, stereo_ranges)]
            spacing_hough = math.dist(*points_hough)
            spacing_stereo = math.dist(*points_stereo)
            count += 1
            pair_distance_hough.append(spacing_hough)
            pair_distance_stereo.append(spacing_stereo)
            z_gap_hough.append(abs(hough_ranges[0] - hough_ranges[1]))
            z_gap_stereo.append(abs(stereo_ranges[0] - stereo_ranges[1]))
            centroid_hough.append(np.mean(points_hough, axis=0))
            for radius in CLUSTER_RADII_M:
                merged[radius] += int(spacing_hough <= radius)
        centroid_mad = None
        if centroid_hough:
            center = np.median(np.stack(centroid_hough), axis=0)
            centroid_mad = [round(float(np.median(np.abs(
                np.stack(centroid_hough)[:, axis] - center[axis]))), 5)
                for axis in range(3)]
        config_results.append({
            'param2': param2, 'valid_pair_frames': count,
            'pair_spacing_hough_z_m': summary(pair_distance_hough),
            'pair_spacing_stereo_z_m': summary(pair_distance_stereo),
            'depth_gap_hough_z_m': summary(z_gap_hough),
            'depth_gap_stereo_z_m': summary(z_gap_stereo),
            'cluster_count_hough_z': {
                str(radius): {'one_cluster': merged[radius],
                              'two_clusters': count - merged[radius],
                              'one_cluster_fraction': round(merged[radius] / count, 5)
                              if count else None}
                for radius in CLUSTER_RADII_M},
            'cluster_centroid_camera_xyz_component_mad_m': centroid_mad,
        })
    output['results'].append({'scene': hough_clip['scene'],
                              'video': hough_clip['video'],
                              'configurations': config_results})

path = ROOT / 'media/frc-fuel-hough-range-clustering-evaluation.json'
path.write_text(json.dumps(output, indent=2) + '\n')
for clip in output['results']:
    print(clip['scene'])
    for item in clip['configurations']:
        print('p2', item['param2'], 'N', item['valid_pair_frames'],
              'spacing Hough/stereo', item['pair_spacing_hough_z_m'],
              item['pair_spacing_stereo_z_m'],
              'depth gap Hough/stereo', item['depth_gap_hough_z_m']['median_m'],
              item['depth_gap_stereo_z_m']['median_m'],
              'merge@.20/.24/.26', [item['cluster_count_hough_z'][str(radius)]['one_cluster_fraction']
                                    for radius in (.20, .24, .26)])
print('saved', path)
