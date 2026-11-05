#!/usr/bin/env python3
"""Compare stereo geometry for frames affected by cascade thresholds 0.0575/0.06."""
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'media/frc-fuel-resolution-cascade-runtime.json'
OUTPUT = ROOT / 'media/frc-fuel-cascade-trigger-stereo-breakpoints.json'
CLIPS = {
    'earlier': ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr.mp4',
    'corrected-settings': ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4',
}

os.environ.update({
    'SCOUT_STEREO_CALIBRATION': 'media/calibration/stereo_calibration_live_order_candidate.yml',
    'SCOUT_STEREO_RECTIFY': '1',
    'SCOUT_STEREO_DISPARITY_SCALE': '1',
    'SCOUT_STEREO_NUM_DISPARITIES': '128',
    'SCOUT_STEREO_MATCH_SCALE': '0.5',
    'SCOUT_STEREO_VERTICAL_ROI_PAD': '24',
    'SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR': '0.55',
    'SCOUT_FUEL_DIAMETER_M': '0.15',
})

from scout.detectors import Detection
from scout.stereo import StereoDepth

TARGETS = ((808, 589), (902, 591))
TRIGGER_CANDIDATE = 0.0575
TRIGGER_REFERENCE = 0.06
RADIUS_SWEEP_M = (0.18, 0.20, 0.21, 0.22, 0.24, 0.26)


def camera_point(stereo, box):
    if not getattr(box, 'distance_m', None):
        return None
    _, _, _, kl, rotation, projection = stereo.maps
    center = np.array([[[float(box.x1 + box.x2) / 2,
                         float(box.y1 + box.y2) / 2]]], dtype=np.float32)
    u, v = cv2.undistortPoints(
        center, kl, stereo.calibration['left_distortion'],
        R=rotation, P=projection)[0, 0]
    z = float(box.distance_m)
    return ((float(u - projection[0, 2]) * z / float(projection[0, 0])),
            (float(v - projection[1, 2]) * z / float(projection[1, 1])), z)


def estimate(stereo, left, right, raw_boxes):
    boxes = [Detection(int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2']),
                       0, float(b['confidence'])) for b in raw_boxes]
    boxes.sort(key=lambda box: (box.x1 + box.x2) / 2)
    stereo.estimate(left, right, boxes)
    points = [camera_point(stereo, box) for box in boxes]
    return {
        'ranges_m': [getattr(box, 'distance_m', None) for box in boxes],
        'disparities_px': [getattr(box, 'disparity_px', None) for box in boxes],
        'camera_xyz_m': points,
        'pair_distance_3d_m': (math.dist(points[0], points[1])
                               if len(points) == 2 and all(points) else None),
    }


source = json.loads(SOURCE.read_text())
stereo = StereoDepth()
rows = []
for clip in source['results']:
    for frame_row in clip['frames_detail']:
        fast_boxes = frame_row['fast_boxes']
        if len(fast_boxes) != 2:
            continue
        minimum = min(box['confidence'] for box in fast_boxes)
        if not TRIGGER_CANDIDATE <= minimum < TRIGGER_REFERENCE:
            continue
        capture = cv2.VideoCapture(str(CLIPS[clip['scene']]))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_row['frame'])
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"Failed to read {clip['scene']} frame {frame_row['frame']}")
        left, right = frame[:, :1600], frame[:, 1600:]
        fast = estimate(stereo, left, right, fast_boxes)
        fallback = estimate(stereo, left, right, frame_row['selected_boxes'])
        pair_distances = {'0.0575': fast['pair_distance_3d_m'],
                          '0.06': fallback['pair_distance_3d_m']}
        rows.append({
            'scene': clip['scene'],
            'frame': frame_row['frame'],
            'minimum_640_confidence': minimum,
            '0.0575_fast_stereo': fast,
            '0.06_fallback_stereo': fallback,
            'cluster_count_by_radius': {
                str(radius): {
                    str(trigger): (1 if distance is not None and distance <= radius else 2)
                    for trigger, distance in pair_distances.items()
                } for radius in RADIUS_SWEEP_M
            },
        })

report = {
    'method': ('Find frames whose 640 minimum confidence falls in [0.0575, 0.06), '
               'then replay each saved SBS frame through the production rectified '
               'SGBM estimator once with its 640 boxes and once with its saved '
               '1280 fallback boxes. Reconstruct box-center camera XYZ and compare '
               'pair spacing and Euclidean cluster counts.'),
    'stereo_settings': {
        'calibration': 'media/calibration/stereo_calibration_live_order_candidate.yml',
        'disparities': 128, 'match_scale': 0.5, 'vertical_roi_pad_px': 24,
        'horizontal_roi': False, 'size_gate_relative_error': 0.55,
    },
    'limitations': ('Only threshold-affected frames in two stationary clips are '
                    'included. Cluster thresholds are exploratory; no absolute '
                    'range truth or varied cluster labels are available.'),
    'frames': rows,
}
OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
for row in rows:
    print(row['scene'], row['frame'],
          'ranges 0.0575/0.06', row['0.0575_fast_stereo']['ranges_m'],
          row['0.06_fallback_stereo']['ranges_m'],
          'pair', row['0.0575_fast_stereo']['pair_distance_3d_m'],
          row['0.06_fallback_stereo']['pair_distance_3d_m'])
print('saved', OUTPUT)
