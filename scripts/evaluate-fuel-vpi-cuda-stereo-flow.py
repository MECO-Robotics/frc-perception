#!/usr/bin/env python3
"""Evaluate VPI CUDA sparse LK tracking for per-frame stereo range updates."""
import json
import math
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
os.environ.setdefault('SCOUT_STEREO_CALIBRATION',
                       'media/calibration/stereo_calibration_live_order_candidate.yml')
os.environ.setdefault('SCOUT_STEREO_RECTIFY', '1')
os.environ.setdefault('SCOUT_STEREO_BASELINE_M', '0.0635')
os.environ.setdefault('SCOUT_STEREO_MATCH_SCALE', '0.5')
os.environ.setdefault('SCOUT_STEREO_NUM_DISPARITIES', '128')
os.environ.setdefault('SCOUT_STEREO_BLOCK_SIZE', '3')
os.environ.setdefault('SCOUT_STEREO_VERTICAL_ROI_PAD', '24')
os.environ.setdefault('SCOUT_STEREO_HORIZONTAL_ROI', '1')
os.environ.setdefault('SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR', '0')
os.environ.setdefault('SCOUT_FUEL_DIAMETER_M', '0.15')

from scout.detectors import Detection  # noqa: E402
from scout.stereo import StereoDepth  # noqa: E402

DETECTIONS = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
DENSE_DEPTH = json.loads((ROOT / 'media/frc-fuel-dense-stereo-evaluation.json').read_text())
VIDEOS = {
    'earlier': ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr.mp4',
    'corrected-settings': ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4',
}
TARGETS = ((808.0, 589.0), (902.0, 591.0))
CONFIDENCE = 0.05
DETECTOR_STRIDE = 3
STEREO_STRIDE = 9
MAX_AGE = 6
SCALE = 0.5
KP_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4')])
cv2.setNumThreads(1)


def median(values):
    return statistics.median(values) if values else None


def make_rectifier(stereo):
    c = stereo.calibration
    size = (1600, 1200)
    kl = c['left_camera_matrix'].copy()
    kr = c['right_camera_matrix'].copy()
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        kl, c['left_distortion'], kr, c['right_distortion'], size,
        c['rotation'], c['translation'], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    ml = cv2.initUndistortRectifyMap(kl, c['left_distortion'], r1, p1,
                                     size, cv2.CV_32FC1)
    mr = cv2.initUndistortRectifyMap(kr, c['right_distortion'], r2, p2,
                                     size, cv2.CV_32FC1)
    return kl, kr, r1, r2, p1, p2, ml, mr


def rectify_point(raw_half, camera, stereo, rectifier):
    kl, kr, r1, r2, p1, p2, _, _ = rectifier
    raw = np.asarray([[[raw_half[0] / SCALE, raw_half[1] / SCALE]]],
                     dtype=np.float32)
    if camera == 0:
        output = cv2.undistortPoints(raw, kl, stereo.calibration['left_distortion'],
                                     R=r1, P=p1)[0, 0]
    else:
        output = cv2.undistortPoints(raw, kr, stereo.calibration['right_distortion'],
                                     R=r2, P=p2)[0, 0]
    return float(output[0]) * SCALE, float(output[1]) * SCALE


def right_raw_from_rectified(rect_xy, rectifier):
    mr = rectifier[-1]
    u = int(np.clip(round(rect_xy[0] / SCALE), 0, mr[0].shape[1] - 1))
    v = int(np.clip(round(rect_xy[1] / SCALE), 0, mr[0].shape[0] - 1))
    return float(mr[0][v, u]) * SCALE, float(mr[1][v, u]) * SCALE


def seed_tracker(gray, center, frame, state=None):
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (round(center[0]), round(center[1])), 17, 255, -1)
    points = cv2.goodFeaturesToTrack(gray, maxCorners=20, qualityLevel=0.003,
                                     minDistance=2, mask=mask, blockSize=5)
    if points is None or len(points) < 3:
        return None
    points = points.reshape(-1, 2)
    host_points = np.empty(len(points), dtype=KP_DTYPE)
    host_points['x'], host_points['y'] = points[:, 0], points[:, 1]
    keypoints = vpi.asarray(host_points, type=vpi.Type.KEYPOINT_F32)
    if state is not None:
        state.update(features=keypoints, points=points, center=center)
        return state
    with vpi.Backend.CUDA:
        tracker = vpi.OpticalFlowPyrLK(frame, keypoints, 3,
                                       backend=vpi.Backend.CUDA)
    return {'tracker': tracker, 'features': keypoints,
            'points': points, 'center': center}


def flow_step(state, frame):
    features, status = state['tracker'](frame, keypoints=state['features'],
                                        winsize=15, maxiter=6)
    with features.rlock_cpu() as data:
        raw_moved = np.array(data, copy=True)
        if raw_moved.dtype.fields:
            moved = np.column_stack((raw_moved['x'], raw_moved['y']))
        else:
            moved = raw_moved.reshape(-1, 2).astype(np.float32, copy=False)
    with status.rlock_cpu() as data:
        valid = np.array(data, copy=True).reshape(-1) == 0
    old = state['points']
    valid_old = old[valid] if len(valid) == len(old) else old
    if len(moved) == len(valid_old):
        old = valid_old
    elif len(moved) == len(old):
        moved = moved[valid]
        old = old[valid]
    else:
        aligned = min(len(moved), len(valid_old))
        moved, old = moved[:aligned], valid_old[:aligned]
    finite = np.isfinite(moved).all(axis=1)
    moved, old = moved[finite], old[finite]
    if not len(moved):
        return False
    delta = np.median(moved - old, axis=0)
    state['center'] = (state['center'][0] + float(delta[0]),
                       state['center'][1] + float(delta[1]))
    state['points'] = moved
    state['features'] = features
    return len(moved) >= 3


def run_scene(scene):
    det_rows = next(item for item in DETECTIONS['results']
                    if item['scene'] == scene)['per_frame']
    depth_rows = next(item for item in DENSE_DEPTH['results']
                      if item['scene'] == scene)['frames']
    video = cv2.VideoCapture(str(VIDEOS[scene]))
    stereo = StereoDepth()
    rectifier = make_rectifier(stereo)
    states = [None, None]
    previous_images = [None, None]
    range_values = [[], []]
    range_deltas = [[], []]
    count_hist = {}
    flow_failures = 0
    processing_s = 0.0
    frame_i = 0
    started = time.perf_counter()
    while True:
        ok, side = video.read()
        if not ok:
            break
        frame_start = time.perf_counter()
        left, right = side[:, :1600], side[:, 1600:]
        left_gray = cv2.resize(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
                               (800, 600), interpolation=cv2.INTER_AREA)
        right_gray = cv2.resize(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY),
                                (800, 600), interpolation=cv2.INTER_AREA)
        with vpi.Backend.CUDA:
            left_frame = vpi.asimage(left_gray, vpi.Format.U8)
            right_frame = vpi.asimage(right_gray, vpi.Format.U8)
        images = (left_frame, right_frame)

        # Track the prior anchors to this frame, including detector-anchor
        # frames; fresh detector boxes below then correct/reseed the tracks.
        if previous_images[0] is not None:
            for state in states:
                if state is None:
                    continue
                left_ok = flow_step(state['left'], left_frame)
                right_ok = flow_step(state['right'], right_frame)
                if not (left_ok and right_ok):
                    flow_failures += 1

        is_detector_anchor = frame_i % DETECTOR_STRIDE == 0
        is_stereo_anchor = frame_i % STEREO_STRIDE == 0
        assigned = [None, None]
        boxes = []
        if is_detector_anchor:
            for item in det_rows[frame_i]['detections']:
                if item['confidence'] >= CONFIDENCE:
                    boxes.append(Detection(item['x1'], item['y1'], item['x2'], item['y2'],
                                           0, item['confidence']))
            for box in sorted(boxes, key=lambda item: item.confidence, reverse=True):
                center = ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)
                target_i = min(range(2), key=lambda i: math.dist(center, TARGETS[i]))
                if math.dist(center, TARGETS[target_i]) <= 60 and assigned[target_i] is None:
                    assigned[target_i] = box
        if is_stereo_anchor and boxes:
            stereo.estimate(left, right, boxes)

        if is_detector_anchor:
            for target_i, box in enumerate(assigned):
                if box is None:
                    if (states[target_i] is not None and
                            frame_i - states[target_i]['anchor_frame'] > MAX_AGE):
                        states[target_i] = None
                    continue
                if is_stereo_anchor and box.disparity_px and box.distance_m:
                    disparity = box.disparity_px
                    anchor_range = box.distance_m
                elif states[target_i] is not None:
                    disparity = states[target_i]['disparity_px']
                    anchor_range = states[target_i]['anchor_range_m']
                else:
                    continue
                left_raw = ((box.x1 + box.x2) * 0.5 * SCALE,
                            (box.y1 + box.y2) * 0.5 * SCALE)
                left_rect = rectify_point(left_raw, 0, stereo, rectifier)
                right_rect = (left_rect[0] - disparity * SCALE, left_rect[1])
                right_raw = right_raw_from_rectified(right_rect, rectifier)
                prior_left = states[target_i]['left'] if states[target_i] is not None else None
                prior_right = states[target_i]['right'] if states[target_i] is not None else None
                left_state = seed_tracker(left_gray, left_raw, left_frame, prior_left)
                right_state = seed_tracker(right_gray, right_raw, right_frame, prior_right)
                if left_state is None or right_state is None:
                    if states[target_i] is None:
                        continue
                    left_state, right_state = states[target_i]['left'], states[target_i]['right']
                states[target_i] = {
                    'left': left_state, 'right': right_state,
                    'disparity_px': disparity,
                    'anchor_range_m': anchor_range,
                    'anchor_frame': frame_i,
                }

        ranges = [None, None]
        for target_i, state in enumerate(states):
            if state is None or frame_i - state['anchor_frame'] > MAX_AGE:
                continue
            left_rect = rectify_point(state['left']['center'], 0, stereo, rectifier)
            right_rect = rectify_point(state['right']['center'], 1, stereo, rectifier)
            disparity = (left_rect[0] - right_rect[0]) / SCALE
            if disparity <= 0:
                continue
            ranges[target_i] = stereo.baseline_m * stereo.focal_px / disparity
            range_values[target_i].append(ranges[target_i])
            reference = depth_rows[frame_i]['ranges_m'][target_i]
            if reference is not None:
                range_deltas[target_i].append(abs(ranges[target_i] - reference))
        count = sum(value is not None for value in ranges)
        count_hist[str(count)] = count_hist.get(str(count), 0) + 1
        previous_images = images
        frame_i += 1
        processing_s += time.perf_counter() - frame_start
    video.release()
    return {
        'scene': scene,
        'frames': frame_i,
        'detector_anchor_rate_hz': 20.0,
        'stereo_sgbm_anchor_rate_hz': 6.67,
        'simulated_position_and_range_output_rate_hz': 60.0,
        'elapsed_s': round(time.perf_counter() - started, 3),
        'processing_s_excluding_decode': round(processing_s, 3),
        'processing_rate_excluding_decode_hz': round(frame_i / processing_s, 2)
        if processing_s else None,
        'count_histogram': count_hist,
        'flow_track_failures': flow_failures,
        'ranges': [
            {'n': len(values), 'median_m': median(values),
             'mad_m': median([abs(value - median(values)) for value in values])
             if values else None,
             'median_abs_delta_vs_same_frame_sgbm_m': median(range_deltas[i]),
             'p95_abs_delta_vs_same_frame_sgbm_m': sorted(range_deltas[i])[
                 min(len(range_deltas[i]) - 1,
                     math.ceil(.95 * len(range_deltas[i])) - 1)]
             if range_deltas[i] else None}
            for i, values in enumerate(range_values)],
        'limitations': 'Stationary clips only. Detector boxes are cached from every frame but used at 20 Hz; SGBM runs at 6.67 Hz. VPI CUDA LK tracks sparse features in both raw views; per-frame ranges are compared with cached same-frame SGBM, not measured distance truth. No concurrent TensorRT runtime, motion, crossing, or real occlusion.',
    }


output = {
    'method': 'Use cached 1280 TensorRT detections every 3 frames to reset positions. Run runtime StereoDepth SGBM every 9 frames for depth anchors. Between anchors, track points with VPI CUDA Pyramidal LK in both raw half-resolution views, rectify tracked centers, and recompute metric range each frame.',
    'vpi_version': '2.4.8',
    'anchor_confidence': CONFIDENCE,
    'max_anchor_age_frames': MAX_AGE,
    'results': [run_scene(scene) for scene in ('earlier', 'corrected-settings')],
}
path = ROOT / 'media/frc-fuel-vpi-cuda-stereo-flow-evaluation.json'
path.write_text(json.dumps(output, indent=2) + '\n')
for result in output['results']:
    print(result['scene'], result['count_histogram'], result['ranges'],
          result['processing_rate_excluding_decode_hz'])
print('saved', path)
