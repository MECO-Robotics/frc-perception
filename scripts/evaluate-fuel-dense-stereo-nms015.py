#!/usr/bin/env python3
"""Run stereo range and 3D spacing on all detections from the saved FUEL clips."""
import argparse
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
DETECTIONS_PATH = ROOT / 'media/frc-fuel-dense-count-nms015.json'
CALIBRATION = 'media/calibration/stereo_calibration_live_order_candidate.yml'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--width', type=int, choices=(64, 96, 128), default=128,
                    help='SGBM disparity search width in matching-image pixels')
parser.add_argument('--scale', type=float, choices=(0.25, 0.33, 0.4, 0.5), default=0.5,
                    help='matching image scale relative to each rectified view')
parser.add_argument('--vertical-pad', type=int, choices=(0, 8, 16, 24, 32), default=0,
                    help='vertical matcher context in matching-image pixels around all detections')
parser.add_argument('--block-size', type=int, choices=(3, 5, 7), default=5,
                    help='SGBM matching window size')
parser.add_argument('--output', default='media/frc-fuel-dense-stereo-nms015-conf005.json')
args = parser.parse_args()
os.environ['SCOUT_STEREO_CALIBRATION'] = CALIBRATION
os.environ['SCOUT_STEREO_RECTIFY'] = '1'
os.environ['SCOUT_STEREO_BASELINE_M'] = '0.0635'
os.environ['SCOUT_STEREO_DISPARITY_SCALE'] = '1'
os.environ['SCOUT_STEREO_NUM_DISPARITIES'] = str(args.width)
os.environ['SCOUT_STEREO_MATCH_SCALE'] = str(args.scale)
os.environ['SCOUT_STEREO_VERTICAL_ROI_PAD'] = str(args.vertical_pad)

from scout.detectors import Detection
from scout.stereo import StereoDepth


def med(values):
    return statistics.median(values) if values else None


def mad(values):
    center = med(values)
    return med([abs(value - center) for value in values]) if values else None


def summary(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {
        'valid_samples': len(ordered),
        'median': round(med(ordered), 4),
        'mad': round(mad(ordered), 4),
        'p05': round(ordered[int(.05 * (len(ordered) - 1))], 4),
        'p95': round(ordered[int(.95 * (len(ordered) - 1))], 4),
    }


def match_targets(detections):
    targets = ((808, 589), (902, 591))
    matched = [None, None]
    for box in sorted(detections, key=lambda item: item.confidence, reverse=True):
        cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        distance, index = min((math.hypot(cx - x, cy - y), i)
                              for i, (x, y) in enumerate(targets))
        if distance <= 60 and matched[index] is None:
            matched[index] = box
    return matched


dense = json.loads(DETECTIONS_PATH.read_text())
stereo = StereoDepth()
if args.block_size != 5:
    block = args.block_size
    stereo.matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=args.width, blockSize=block,
        P1=8 * block * block, P2=32 * block * block,
        uniquenessRatio=8, speckleWindowSize=50, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
output = {
    'method': f'Reuse all per-frame 1280x960 detector outputs from the NMS IoU 0.15 confidence sweep at confidence 0.05 from the dense count evaluation and compute rectified SGBM b{args.block_size}/u8 stereo depth at match scale {args.scale} with search width {args.width}. Unproject detection centers with the same rectified P1 calibration and stereo Z.',
    'calibration': CALIBRATION,
    'num_disparities_match_px': args.width,
    'match_scale': args.scale,
    'vertical_roi_pad_match_px': args.vertical_pad,
    'block_size': args.block_size,
    'baseline_m': stereo.baseline_m,
    'baseline_source': 'SCOUT_STEREO_BASELINE_M=0.0635, matching the service startup profile',
    'distance_limitations': 'Stationary recorded balls with no tape-measured range. Stereo range repeatability and 3D spacing do not establish absolute metric accuracy.',
    'results': [],
}

for clip in dense['results']:
    capture = cv2.VideoCapture(str(ROOT / clip['video']))
    valid_ranges = [[], []]
    pair_spacings = []
    valid_pairs = 0
    frame_rows = []
    stereo_times = []
    frames = clip['per_frame']
    started = time.perf_counter()
    for index, row in enumerate(frames):
        ok, frame = capture.read()
        if not ok:
            break
        left, right = frame[:, :1600], frame[:, 1600:]
        boxes = [Detection(**{key: int(item[key]) for key in ('x1', 'y1', 'x2', 'y2')},
                           class_id=0, confidence=float(item['confidence']))
                 for item in row['detections'] if item['confidence'] >= 0.05]
        matched = match_targets(boxes)
        t0 = time.perf_counter()
        stereo.estimate(left, right, [box for box in matched if box is not None])
        stereo_times.append((time.perf_counter() - t0) * 1000)

        ranges = []
        points = []
        for target_index, box in enumerate(matched):
            if box is None or box.distance_m is None:
                ranges.append(None)
                points.append(None)
                continue
            z = float(box.distance_m)
            valid_ranges[target_index].append(z)
            ranges.append(z)
            center = np.array([[[float(box.x1 + box.x2) / 2,
                                 float(box.y1 + box.y2) / 2]]], dtype=np.float32)
            _, _, _, kl, r1, p1 = stereo.maps
            uv = cv2.undistortPoints(center, kl, stereo.calibration['left_distortion'],
                                     R=r1, P=p1)[0, 0]
            points.append(((uv[0] - p1[0, 2]) * z / p1[0, 0],
                           (uv[1] - p1[1, 2]) * z / p1[1, 1], z))
        spacing = None
        if all(point is not None for point in points):
            spacing = math.dist(points[0], points[1])
            pair_spacings.append(spacing)
            valid_pairs += 1
        frame_rows.append({
            'frame': index,
            'time_s': row['time_s'],
            'ranges_m': [round(value, 5) if value is not None else None for value in ranges],
            'disparities_px': [round(float(box.disparity_px), 3)
                               if box is not None and box.disparity_px is not None else None
                               for box in matched],
            'pair_spacing_m': round(spacing, 5) if spacing is not None else None,
        })
    capture.release()
    elapsed = time.perf_counter() - started
    output['results'].append({
        'scene': clip['scene'],
        'video': clip['video'],
        'frames_processed': len(frame_rows),
        'valid_pair_frames': valid_pairs,
        'range_by_target_m': [summary(values) for values in valid_ranges],
        'pair_spacing_m': summary(pair_spacings),
        'stereo_compute_ms_median': round(med(stereo_times), 2) if stereo_times else None,
        'stereo_compute_ms_p95': round(sorted(stereo_times)[int(.95 * (len(stereo_times) - 1))], 2)
        if stereo_times else None,
        'elapsed_seconds': round(elapsed, 2),
        'frames': frame_rows,
    })

out_path = ROOT / args.output
out_path.write_text(json.dumps(output, indent=2) + '\n')
for result in output['results']:
    print(result['scene'], 'valid pairs', result['valid_pair_frames'], '/', result['frames_processed'])
    print(' ranges', result['range_by_target_m'], 'spacing', result['pair_spacing_m'])
    print(' SGBM ms median/p95', result['stereo_compute_ms_median'], result['stereo_compute_ms_p95'])
print('saved', out_path)
