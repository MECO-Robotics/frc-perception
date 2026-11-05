#!/usr/bin/env python3
"""Measure sparse SGBM anchors with 60 Hz stereo feature tracking."""
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
cv2.setNumThreads(1)


def median(values):
    return statistics.median(values) if values else None


def make_rectifier(stereo):
    c = stereo.calibration
    size = (1600, 1200)
    kl = c['left_camera_matrix'].copy()
    kr = c['right_camera_matrix'].copy()
    kl[0, :] *= 1.0; kl[1, :] *= 1.0
    kr[0, :] *= 1.0; kr[1, :] *= 1.0
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        kl, c['left_distortion'], kr, c['right_distortion'], size,
        c['rotation'], c['translation'], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    ml = cv2.initUndistortRectifyMap(kl, c['left_distortion'], r1, p1,
                                     size, cv2.CV_32FC1)
    mr = cv2.initUndistortRectifyMap(kr, c['right_distortion'], r2, p2,
                                     size, cv2.CV_32FC1)
    return (kl, kr, r1, r2, p1, p2, ml, mr)


def rectify_point(raw_half, camera, rectifier):
    kl, kr, r1, r2, p1, p2, _, _ = rectifier
    raw_full = np.asarray([[[raw_half[0] / SCALE, raw_half[1] / SCALE]]],
                          dtype=np.float32)
    if camera == 0:
        output = cv2.undistortPoints(
            raw_full, kl, np.asarray(STEREO.calibration['left_distortion']),
            R=r1, P=p1)[0, 0]
    else:
        output = cv2.undistortPoints(
            raw_full, kr, np.asarray(STEREO.calibration['right_distortion']),
            R=r2, P=p2)[0, 0]
    return float(output[0]) * SCALE, float(output[1]) * SCALE


def right_raw_from_rectified(rect_xy, rectifier):
    *_, mr = rectifier
    u = int(np.clip(round(rect_xy[0] / SCALE), 0, mr[0].shape[1] - 1))
    v = int(np.clip(round(rect_xy[1] / SCALE), 0, mr[0].shape[0] - 1))
    return float(mr[0][v, u]) * SCALE, float(mr[1][v, u]) * SCALE


def seed(gray, center):
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (round(center[0]), round(center[1])), 17, 255, -1)
    return cv2.goodFeaturesToTrack(gray, maxCorners=20, qualityLevel=0.003,
                                   minDistance=2, mask=mask, blockSize=5)


def track(previous, current, state):
    points = state['points']
    if points is None or not len(points):
        return False
    moved, status, error = cv2.calcOpticalFlowPyrLK(
        previous, current, points, None, winSize=(15, 15), maxLevel=0,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
        minEigThreshold=1e-4)
    if moved is None:
        state['points'] = None
        return False
    keep = status.reshape(-1).astype(bool) & (error.reshape(-1) < 35)
    old = points.reshape(-1, 2)[keep]
    new = moved.reshape(-1, 2)[keep]
    if not len(new):
        state['points'] = None
        return False
    delta = np.median(new - old, axis=0)
    state['center'] = (state['center'][0] + float(delta[0]),
                       state['center'][1] + float(delta[1]))
    state['points'] = new.reshape(-1, 1, 2)
    return len(new) >= 3


def run_scene(scene):
    global STEREO
    det_rows = next(item for item in DETECTIONS['results']
                    if item['scene'] == scene)['per_frame']
    depth_rows = next(item for item in DENSE_DEPTH['results']
                      if item['scene'] == scene)['frames']
    video = cv2.VideoCapture(str(VIDEOS[scene]))
    stereo = StereoDepth()
    STEREO = stereo
    rectifier = make_rectifier(stereo)
    states = [None, None]
    prev_left = prev_right = None
    count_hist = {}
    range_values = [[], []]
    range_abs_delta = [[], []]
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
        is_detector_anchor = frame_i % DETECTOR_STRIDE == 0
        is_stereo_anchor = frame_i % STEREO_STRIDE == 0
        boxes = []
        assigned = [None, None]
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
        for state in states:
            if state is None or prev_left is None:
                continue
            left_ok = track(prev_left, left_gray, state['left'])
            right_ok = track(prev_right, right_gray, state['right'])
            if not (left_ok and right_ok):
                flow_failures += 1

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
                left_raw = (box.x1 + box.x2) * 0.5 * SCALE, (box.y1 + box.y2) * 0.5 * SCALE
                left_rect = rectify_point(left_raw, 0, rectifier)
                right_rect = (left_rect[0] - disparity * SCALE, left_rect[1])
                right_raw = right_raw_from_rectified(right_rect, rectifier)
                states[target_i] = {
                    'left': {'center': left_raw, 'points': seed(left_gray, left_raw)},
                    'right': {'center': right_raw, 'points': seed(right_gray, right_raw)},
                    'disparity_px': disparity,
                    'anchor_range_m': anchor_range,
                    'anchor_frame': frame_i,
                }

        ranges_this_frame = [None, None]
        for target_i, state in enumerate(states):
            if state is None or frame_i - state['anchor_frame'] > MAX_AGE:
                continue
            left_rect = rectify_point(state['left']['center'], 0, rectifier)
            right_rect = rectify_point(state['right']['center'], 1, rectifier)
            disparity = (left_rect[0] - right_rect[0]) / SCALE
            if disparity <= 0:
                continue
            range_m = stereo.baseline_m * stereo.focal_px / disparity
            ranges_this_frame[target_i] = range_m
            range_values[target_i].append(range_m)
            reference = depth_rows[frame_i]['ranges_m'][target_i]
            if reference is not None:
                range_abs_delta[target_i].append(abs(range_m - reference))
        count = sum(value is not None for value in ranges_this_frame)
        count_hist[str(count)] = count_hist.get(str(count), 0) + 1
        prev_left, prev_right = left_gray, right_gray
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
             'median_abs_delta_vs_same_frame_sgbm_m': median(range_abs_delta[i]),
             'p95_abs_delta_vs_same_frame_sgbm_m': sorted(range_abs_delta[i])[
                 min(len(range_abs_delta[i]) - 1,
                     math.ceil(.95 * len(range_abs_delta[i])) - 1)]
             if range_abs_delta[i] else None}
            for i, values in enumerate(range_values)],
        'limitations': 'Stationary clips only. TensorRT boxes are cached from every frame but used only at 20 Hz; SGBM runs at 6.67 Hz. Between stereo anchors, range is derived from independent left/right KLT tracks and compared with cached full-frame SGBM, not measured ground truth. No concurrent detector, camera callback, motion, crossings, or real occlusion.',
    }


output = {
    'method': 'Use cached 1280 TensorRT detections every 3 frames to reset image positions. Run runtime-calibrated SGBM only every 9 frames for range anchors. Between anchors, track LK features in both raw half-resolution views; rectify only each tracked center and compute metric range from horizontal disparity on every frame.',
    'anchor_confidence': CONFIDENCE,
    'max_anchor_age_frames': MAX_AGE,
    'results': [run_scene(scene) for scene in ('earlier', 'corrected-settings')],
}
path = ROOT / 'media/frc-fuel-sparse-stereo-flow-evaluation.json'
path.write_text(json.dumps(output, indent=2) + '\n')
for result in output['results']:
    print(result['scene'], result['count_histogram'], result['ranges'],
          result['processing_rate_excluding_decode_hz'])
print('saved', path)
