#!/usr/bin/env python3
"""Compare NVIDIA VPI CUDA SGM with the current OpenCV SGBM on saved FUEL video."""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import vpi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', default='media/frc-2026-fuel-camera-settings-fixed-60fps.mp4')
parser.add_argument('--output', default='media/frc-fuel-vpi-cuda-stereo-evaluation.json')
parser.add_argument('--samples', type=int, default=30)
args = parser.parse_args()

os.environ.setdefault('SCOUT_STEREO_RECTIFY', '1')
os.environ.setdefault('SCOUT_STEREO_CALIBRATION', 'media/calibration/stereo_calibration_live_order_candidate.yml')
os.environ.setdefault('SCOUT_STEREO_BASELINE_M', '0.0635')
os.environ.setdefault('SCOUT_STEREO_BLOCK_SIZE', '3')
os.environ.setdefault('SCOUT_STEREO_MATCH_SCALE', '0.5')

from scout.stereo import StereoDepth  # noqa: E402
from scout.detectors import Detection  # noqa: E402

stereo = StereoDepth()
cap = cv2.VideoCapture(str((ROOT / args.input).resolve()))
if not cap.isOpened():
    raise SystemExit(f'Unable to open {args.input}')
fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
timestamps = np.linspace(min(0.5, duration / 10), max(min(0.5, duration / 10), duration - 0.5),
                         num=args.samples)

# Median normalized API boxes from this stationary scene, in x/y/width/height form.
target_boxes = ((0.471, 0.444, 0.066, 0.084), (0.529, 0.446, 0.071, 0.092))
left_images, right_images, roi_specs = [], [], []
for timestamp in timestamps:
    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000)
    ok, frame = cap.read()
    if not ok:
        continue
    width = frame.shape[1] // 2
    left = frame[:, :width]
    right = frame[:, width:]
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    if stereo.maps is None:
        dummy = [Detection(int(.45 * width), int(.4 * left.shape[0]),
                           int(.55 * width), int(.5 * left.shape[0]), 0, .9)]
        stereo.estimate(left, right_gray, dummy)
    _, map_left, map_right, _, _, _ = stereo.maps
    left_rect = cv2.remap(left_gray, map_left[0], map_left[1], cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_gray, map_right[0], map_right[1], cv2.INTER_LINEAR)
    match_size = (round(width * stereo.match_scale),
                  round(left.shape[0] * stereo.match_scale))
    left_match = cv2.resize(left_rect, match_size, interpolation=cv2.INTER_AREA)
    right_match = cv2.resize(right_rect, match_size, interpolation=cv2.INTER_AREA)
    left_images.append(np.ascontiguousarray(left_match))
    right_images.append(np.ascontiguousarray(right_match))
    roi_specs.append([(round((bx + bw / 2) * match_size[0]),
                       round((by + bh / 2) * match_size[1]),
                       round(bw * match_size[0]), round(bh * match_size[1]))
                      for bx, by, bw, bh in target_boxes])
cap.release()
if not left_images:
    raise SystemExit('No video frames were decoded')

matcher = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=stereo.num_disparities,
    blockSize=stereo.block_size,
    P1=8 * stereo.block_size * stereo.block_size,
    P2=32 * stereo.block_size * stereo.block_size,
    uniquenessRatio=8, speckleWindowSize=50, speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
width, height = left_images[0].shape[1], left_images[0].shape[0]
vpi_output = vpi.Image((width, height), vpi.Format.S16)
vpi_confidence = vpi.Image((width, height), vpi.Format.U16)

def vpi_disparity(left, right):
    left_image = vpi.asimage(left, format=vpi.Format.U8)
    right_image = vpi.asimage(right, format=vpi.Format.U8)
    with vpi.Backend.CUDA:
        result = vpi.stereodisp(
            left_image, right_image, backend=vpi.Backend.CUDA,
            out=vpi_output, out_confmap=vpi_confidence,
            window=5, maxdisp=stereo.num_disparities,
            confthreshold=1, uniqueness=-1, includediagonals=True)
    # cpu() synchronizes the CUDA work; include its result transfer in timing.
    return result.cpu().astype(np.float32) / 32.0

# Warm CUDA/VPI initialization and kernel compilation before collecting timings.
for index in range(min(3, len(left_images))):
    vpi_disparity(left_images[index], right_images[index])

opencv_times, vpi_times = [], []
per_frame = []
for left, right, rois in zip(left_images, right_images, roi_specs):
    started = time.perf_counter()
    cv_disparity = matcher.compute(left, right).astype(np.float32) / 16.0
    opencv_times.append((time.perf_counter() - started) * 1000)
    started = time.perf_counter()
    gpu_disparity = vpi_disparity(left, right)
    vpi_times.append((time.perf_counter() - started) * 1000)
    row = {'opencv_disparity_px': [], 'vpi_disparity_px': [], 'vpi_valid_fraction': []}
    for x, y, box_width, box_height in rois:
        x1, x2 = max(0, x - box_width // 2), min(width, x + box_width // 2)
        y1, y2 = max(0, y - box_height // 2), min(height, y + box_height // 2)
        cv_roi = cv_disparity[y1:y2, x1:x2]
        vpi_roi = gpu_disparity[y1:y2, x1:x2]
        cv_valid = cv_roi[np.isfinite(cv_roi) & (cv_roi > 1.5) & (cv_roi < 60)]
        vpi_valid = vpi_roi[np.isfinite(vpi_roi) & (vpi_roi > 1.5) & (vpi_roi < 60)]
        row['opencv_disparity_px'].append(float(np.median(cv_valid) / stereo.match_scale)
                                          if cv_valid.size >= 8 else None)
        row['vpi_disparity_px'].append(float(np.median(vpi_valid) / stereo.match_scale)
                                       if vpi_valid.size >= 8 else None)
        row['vpi_valid_fraction'].append(float(vpi_valid.size / max(1, vpi_roi.size)))
    per_frame.append(row)

def summarize(values):
    median = statistics.median(values)
    return {'n': len(values), 'median_ms': round(median, 3),
            'p95_ms': round(float(np.percentile(values, 95)), 3),
            'fps_from_median': round(1000 / median, 2)}

range_rows = []
for row in per_frame:
    ranges = {}
    for matcher_name in ('opencv_disparity_px', 'vpi_disparity_px'):
        values = row[matcher_name]
        ranges[matcher_name] = [round(stereo.baseline_m * stereo.focal_px / value, 4)
                                if value else None for value in values]
    range_rows.append(ranges)

result = {
    'input': str((ROOT / args.input).resolve()),
    'samples': len(left_images),
    'input_resolution_per_eye': [int(left_images[0].shape[1] / stereo.match_scale),
                                 int(left_images[0].shape[0] / stereo.match_scale)],
    'matcher_resolution': [width, height],
    'stereo': {'backend': 'VPI CUDA SGM', 'window': 5,
               'max_disparity': stereo.num_disparities,
               'downscale': stereo.match_scale, 'confidence_threshold': 1,
               'uniqueness': -1, 'diagonals': True},
    'opencv_sgbm_timing': summarize(opencv_times),
    'vpi_cuda_timing': summarize(vpi_times),
    'per_frame': per_frame,
    'per_frame_range_m': range_rows,
    'notes': 'Timings include host-to-GPU input use and CPU disparity-map retrieval; exclude video decode and rectification. No tape-measured distance is available.'}
out_path = ROOT / args.output
out_path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('per_frame', 'per_frame_range_m')}, indent=2))
print('saved', out_path)
