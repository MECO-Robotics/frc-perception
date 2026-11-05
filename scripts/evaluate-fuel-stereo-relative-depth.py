#!/usr/bin/env python3
"""Compare stereo matchers on every saved frame, including relative target depth."""
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
COUNTS = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
OUTPUT = ROOT / 'media/frc-fuel-stereo-relative-depth-evaluation.json'
CALIBRATION = 'media/calibration/stereo_calibration_live_order_candidate.yml'
TARGETS = ((808, 589), (902, 591))
CONFIGS = ('SGBM_b3', 'SGBM_b5', 'StereoBM_b31')
CLUSTER_RADII_M = (0.15, 0.22, 0.26)

os.environ.update({
    'SCOUT_STEREO_CALIBRATION': CALIBRATION,
    'SCOUT_STEREO_RECTIFY': '1',
    'SCOUT_STEREO_DISPARITY_SCALE': '1',
    'SCOUT_STEREO_BASELINE_M': '0.0635',
    'SCOUT_STEREO_NUM_DISPARITIES': '128',
    'SCOUT_STEREO_MATCH_SCALE': '0.5',
    'SCOUT_STEREO_VERTICAL_ROI_PAD': '24',
    'SCOUT_STEREO_HORIZONTAL_ROI': '0',
    'SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR': '0',
})

from scout.detectors import Detection
from scout.stereo import StereoDepth


def stats(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not values:
        return {'n': 0}
    median = statistics.median(values)
    return {
        'n': len(values), 'median_m': round(median, 5),
        'mad_m': round(statistics.median(abs(v - median) for v in values), 5),
        'p05_m': round(float(np.percentile(values, 5)), 5),
        'p95_m': round(float(np.percentile(values, 95)), 5),
    }


def detections_for_frame(detections):
    hits = [None, None]
    for box in sorted((d for d in detections if d['confidence'] >= 0.05),
                      key=lambda d: -d['confidence']):
        cx, cy = (box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2
        distances = [math.hypot(cx - x, cy - y) for x, y in TARGETS]
        target = min(range(2), key=distances.__getitem__)
        if distances[target] <= 60 and hits[target] is None:
            hits[target] = box
    return hits


def make_stereo(config):
    stereo = StereoDepth()
    if config.startswith('SGBM'):
        block = int(config[-1])
        stereo.matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=128, blockSize=block,
            P1=8 * block * block, P2=32 * block * block,
            disp12MaxDiff=-1, preFilterCap=31, uniquenessRatio=8,
            speckleWindowSize=50, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    else:
        block = 31
        matcher = cv2.StereoBM_create(numDisparities=128, blockSize=block)
        matcher.setMinDisparity(0)
        matcher.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
        matcher.setPreFilterSize(9)
        matcher.setPreFilterCap(31)
        matcher.setTextureThreshold(10)
        matcher.setUniquenessRatio(15)
        matcher.setSpeckleWindowSize(50)
        matcher.setSpeckleRange(2)
        matcher.setDisp12MaxDiff(1)
        stereo.matcher = matcher
    return stereo


def make_boxes(targets):
    return [Detection(int(d['x1']), int(d['y1']), int(d['x2']), int(d['y2']),
                      0, float(d['confidence'])) for d in targets]


def xyz(stereo, box):
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


all_results = []
for clip in COUNTS['results']:
    capture = cv2.VideoCapture(str(ROOT / clip['video']))
    matchers = {name: make_stereo(name) for name in CONFIGS}
    rows = {
        name: {'ranges': [[], []], 'signed_z_gap': [], 'absolute_z_gap': [],
               'pair_distance': [], 'times_ms': [], 'valid_pairs': 0}
        for name in CONFIGS
    }
    frame_index = 0
    started = time.monotonic()
    frame_rows = clip['per_frame']
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        detections = frame_rows[frame_index]['detections']
        targets = detections_for_frame(detections)
        frame_index += 1
        if not all(targets):
            continue
        left, right = frame[:, :1600], frame[:, 1600:]
        for name, stereo in matchers.items():
            boxes = make_boxes(targets)
            started_match = time.monotonic()
            stereo.estimate(left, right, boxes)
            rows[name]['times_ms'].append((time.monotonic() - started_match) * 1000)
            z = [getattr(box, 'distance_m', None) for box in boxes]
            for target, value in enumerate(z):
                rows[name]['ranges'][target].append(value)
            if all(z):
                points = [xyz(stereo, box) for box in boxes]
                if all(points):
                    rows[name]['valid_pairs'] += 1
                    rows[name]['signed_z_gap'].append(z[0] - z[1])
                    rows[name]['absolute_z_gap'].append(abs(z[0] - z[1]))
                    rows[name]['pair_distance'].append(math.dist(points[0], points[1]))
    capture.release()
    summary = {}
    for name, result in rows.items():
        summary[name] = {
            'target_range_m': [stats(values) for values in result['ranges']],
            'signed_target1_minus_target2_z_gap_m': stats(result['signed_z_gap']),
            'absolute_target_z_gap_m': stats(result['absolute_z_gap']),
            'pair_center_distance_3d_m': stats(result['pair_distance']),
            'valid_pair_frames': result['valid_pairs'],
            'matcher_time_ms_median': round(statistics.median(result['times_ms']), 2),
        }
        for radius in CLUSTER_RADII_M:
            summary[name].setdefault('pair_cluster_count_by_radius', {})[str(radius)] = {
                'one_cluster_frames': sum(x <= radius for x in result['pair_distance']),
                'two_cluster_frames': sum(x > radius for x in result['pair_distance']),
                'valid_pair_frames': result['valid_pairs'],
            }
    all_results.append({
        'scene': clip['scene'], 'video_frames': frame_index,
        'elapsed_s': round(time.monotonic() - started, 2),
        'configurations': summary,
    })
    print(clip['scene'], json.dumps(summary))

report = {
    'method': ('Replay every frame with two manually matched detector targets, '
               'then run full-width rectified vertical-band SGBM block 3/5 or '
               'StereoBM block 31 on the same boxes.'),
    'configurations': list(CONFIGS),
    'baseline_m': 0.0635,
    'limitations': ('The two clips show the same stationary pair at one pose. '
                    'Equal target depth is a hypothesis pending user confirmation; '
                    'no measured camera-to-ball range exists. Matcher variation '
                    'measures repeatability and relative geometry, not absolute accuracy.'),
    'results': all_results,
}
OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
print('saved', OUTPUT)
