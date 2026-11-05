#!/usr/bin/env python3
"""Evaluate stereo KLT tracks for per-frame range between detector anchors."""
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
os.environ.setdefault('SCOUT_STEREO_HORIZONTAL_ROI', '0')
os.environ.setdefault('SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR', '0')
os.environ.setdefault('SCOUT_FUEL_DIAMETER_M', '0.15')
os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'

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
STRIDE = 3
FLOW_SCALE = 0.5
MAX_AGE = 6
cv2.setNumThreads(1)


def median(values):
    return statistics.median(values) if values else None


def feature_points(gray, center, radius=17):
    x, y = center
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (round(x), round(y)), radius, 255, -1)
    return cv2.goodFeaturesToTrack(gray, maxCorners=20, qualityLevel=0.003,
                                   minDistance=2, mask=mask, blockSize=5)


def track_one(previous, current, state):
    if state['points'] is None or len(state['points']) == 0:
        return False
    tracked, status, error = cv2.calcOpticalFlowPyrLK(
        previous, current, state['points'], None,
        winSize=(15, 15), maxLevel=0,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
        minEigThreshold=1e-4)
    if tracked is None:
        state['points'] = None
        return False
    keep = status.reshape(-1).astype(bool) & (error.reshape(-1) < 35)
    old = state['points'].reshape(-1, 2)[keep]
    new = tracked.reshape(-1, 2)[keep]
    if len(new) == 0:
        state['points'] = None
        return False
    delta = np.median(new - old, axis=0)
    state['center'] = (state['center'][0] + float(delta[0]),
                       state['center'][1] + float(delta[1]))
    state['points'] = new.reshape(-1, 1, 2)
    return len(new) >= 3


def point_rectify(point, stereo):
    _, _, _, kl, rotation_left, projection_left = stereo.maps
    xy = np.asarray([[[point[0], point[1]]]], dtype=np.float32)
    result = cv2.undistortPoints(
        xy, kl, stereo.calibration['left_distortion'],
        R=rotation_left, P=projection_left)[0, 0]
    return float(result[0]) * FLOW_SCALE, float(result[1]) * FLOW_SCALE


def run_scene(scene):
    det_clip = next(item for item in DETECTIONS['results'] if item['scene'] == scene)
    depth_clip = next(item for item in DENSE_DEPTH['results'] if item['scene'] == scene)
    det_rows, depth_rows = det_clip['per_frame'], depth_clip['frames']
    video = cv2.VideoCapture(str(VIDEOS[scene]))
    stereo = StereoDepth()
    states = [None, None]
    previous_left = previous_right = None
    tracked_ranges = [[], []]
    abs_deltas_to_sgbm = [[], []]
    anchor_depth_deltas = [[], []]
    counts = []
    flow_fallbacks = 0
    frame_i = 0
    processing_s = 0.0
    started = time.perf_counter()
    while True:
        ok, side_by_side = video.read()
        if not ok:
            break
        frame_started = time.perf_counter()
        left = side_by_side[:, :1600]
        right = side_by_side[:, 1600:]
        is_anchor = frame_i % STRIDE == 0
        boxes = []
        assigned = [None, None]
        if is_anchor:
            for item in det_rows[frame_i]['detections']:
                if item['confidence'] >= CONFIDENCE:
                    boxes.append(Detection(item['x1'], item['y1'], item['x2'], item['y2'],
                                           0, item['confidence']))
            stereo.estimate(left, right, boxes)
            for box in sorted(boxes, key=lambda item: item.confidence, reverse=True):
                cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
                target_i = min(range(2), key=lambda i:
                               math.hypot(cx - TARGETS[i][0], cy - TARGETS[i][1]))
                if math.dist((cx, cy), TARGETS[target_i]) <= 60 and assigned[target_i] is None:
                    assigned[target_i] = box
        if stereo.maps is not None:
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            _, ml, mr, *_ = stereo.maps
            left_rect = cv2.resize(cv2.remap(left_gray, ml[0], ml[1], cv2.INTER_LINEAR),
                                   (800, 600), interpolation=cv2.INTER_AREA)
            right_rect = cv2.resize(cv2.remap(right_gray, mr[0], mr[1], cv2.INTER_LINEAR),
                                    (800, 600), interpolation=cv2.INTER_AREA)
        else:
            left_rect = right_rect = None
        if previous_left is not None and left_rect is not None:
            for state in states:
                if state is None:
                    continue
                left_ok = track_one(previous_left, left_rect, state['left'])
                right_ok = track_one(previous_right, right_rect, state['right'])
                if not (left_ok and right_ok):
                    flow_fallbacks += 1

        if is_anchor and stereo.maps is not None:
            for target_i, box in enumerate(assigned):
                if box is not None and box.disparity_px and box.distance_m:
                    left_center = point_rectify(((box.x1 + box.x2) / 2,
                                                 (box.y1 + box.y2) / 2), stereo)
                    right_center = (left_center[0] - box.disparity_px * FLOW_SCALE,
                                    left_center[1])
                    states[target_i] = {
                        'left': {'center': left_center,
                                 'points': feature_points(left_rect, left_center),
                                 'anchor_frame': frame_i},
                        'right': {'center': right_center,
                                  'points': feature_points(right_rect, right_center),
                                  'anchor_frame': frame_i},
                        'anchor_range_m': box.distance_m,
                    }
                elif (states[target_i] is not None and
                      frame_i - states[target_i]['left']['anchor_frame'] > MAX_AGE):
                    states[target_i] = None

        frame_count = 0
        for target_i, state in enumerate(states):
            if (state is None or frame_i - state['left']['anchor_frame'] > MAX_AGE or
                    left_rect is None or right_rect is None):
                continue
            disparity_small = state['left']['center'][0] - state['right']['center'][0]
            disparity = disparity_small / FLOW_SCALE
            if disparity <= 0:
                continue
            range_m = stereo.baseline_m * stereo.focal_px / disparity
            tracked_ranges[target_i].append(range_m)
            reference = depth_rows[frame_i]['ranges_m'][target_i]
            if reference is not None:
                abs_deltas_to_sgbm[target_i].append(abs(range_m - reference))
            if frame_i % STRIDE == 0:
                anchor_depth_deltas[target_i].append(abs(range_m - state['anchor_range_m']))
            frame_count += 1
        counts.append(frame_count)
        previous_left, previous_right = left_rect, right_rect
        frame_i += 1
        processing_s += time.perf_counter() - frame_started
    video.release()
    return {
        'scene': scene, 'frames': frame_i, 'detector_anchor_rate_hz': 20.0,
        'simulated_output_rate_hz': 60.0,
        'elapsed_s': round(time.perf_counter() - started, 3),
        'processing_s_excluding_decode': round(processing_s, 3),
        'processing_rate_excluding_decode_hz': round(frame_i / processing_s, 2)
        if processing_s else None,
        'valid_two_range_frames': sum(count == 2 for count in counts),
        'count_histogram': {str(n): counts.count(n) for n in sorted(set(counts))},
        'flow_track_failures': flow_fallbacks,
        'target_range_median_mad': [
            {'n': len(values), 'median_m': median(values),
             'mad_m': median([abs(value - median(values)) for value in values])
             if values else None,
             'median_abs_delta_from_same_frame_sgbm_m': median(abs_deltas_to_sgbm[i]),
             'p95_abs_delta_from_same_frame_sgbm_m': sorted(abs_deltas_to_sgbm[i])[
                 min(len(abs_deltas_to_sgbm[i]) - 1,
                     math.ceil(.95 * len(abs_deltas_to_sgbm[i])) - 1)]
             if abs_deltas_to_sgbm[i] else None,
             'median_anchor_range_difference_m': median(anchor_depth_deltas[i])}
            for i, values in enumerate(tracked_ranges)],
        'limitations': 'Stationary two-ball videos only. Detector boxes and full-frame SGBM are cached references. Flow-derived stereo uses tracked image features in both rectified views and metric calibration, but is compared with the same-frame SGBM output, not measured distance truth. No motion, real occlusion, crossing, or concurrent TensorRT runtime is included.',
    }


output = {
    'method': 'Use cached 1280 TensorRT detections only every 3rd frame. At each detector anchor, use runtime StereoDepth SGBM to initialize rectified left/right feature tracks and anchor range. Between anchors, propagate sparse LK features independently in the two rectified views and recompute range from tracked horizontal disparity at every frame.',
    'anchor_confidence': CONFIDENCE,
    'max_track_age_frames': MAX_AGE,
    'results': [run_scene(scene) for scene in ('earlier', 'corrected-settings')],
}
path = ROOT / 'media/frc-fuel-stereo-flow-between-anchors.json'
path.write_text(json.dumps(output, indent=2) + '\n')
for result in output['results']:
    print(result['scene'], result['count_histogram'],
          result['target_range_median_mad'],
          result['processing_rate_excluding_decode_hz'])
print('saved', path)
