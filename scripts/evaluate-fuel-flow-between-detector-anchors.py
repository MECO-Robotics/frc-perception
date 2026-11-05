#!/usr/bin/env python3
"""Evaluate sparse detector anchors plus LK flow on the two fixed FUEL clips.

This is an offline 60 Hz output-rate experiment. Detector boxes are read from
the per-frame 1280 TensorRT cache, but used only every N frames. LK flow tracks
the central image patch between anchors. It cannot validate moving targets or
fresh stereo range because both saved scenes are stationary.
"""
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
VIDEOS = {
    'earlier': ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr.mp4',
    'corrected-settings': ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4',
}
TARGETS = ((808.0, 589.0), (902.0, 591.0))
GATE_PX = 60.0
STRIDES = (3,)
CONFIDENCE = 0.05
MAX_TRACK_AGE = 6
FLOW_SCALE = 0.5
FLOW_TARGETS = tuple((x * FLOW_SCALE, y * FLOW_SCALE) for x, y in TARGETS)
FLOW_MAX_LEVEL = 0
FLOW_WINDOW = (15, 15)
cv2.setNumThreads(1)


def median(values):
    return statistics.median(values) if values else None


def frame_detections(scene):
    result = next(item for item in DATA['results'] if item['scene'] == scene)
    return result['per_frame']


def assign_detections(detections):
    assigned = [None, None]
    extras = 0
    for box in sorted((box for box in detections
                       if box['confidence'] >= CONFIDENCE),
                      key=lambda box: box['confidence'], reverse=True):
        cx = (box['x1'] + box['x2']) * 0.5
        cy = (box['y1'] + box['y2']) * 0.5
        target_i = min(range(2), key=lambda i:
                       math.hypot(cx - TARGETS[i][0], cy - TARGETS[i][1]))
        if math.hypot(cx - TARGETS[target_i][0],
                      cy - TARGETS[target_i][1]) > GATE_PX or assigned[target_i]:
            extras += 1
            continue
        assigned[target_i] = (cx, cy, box)
    return assigned, extras


def seed_points(gray, center, box=None):
    cx, cy = center
    if box:
        radius = max(12, min(box['x2'] - box['x1'], box['y2'] - box['y1']) * 0.36)
    else:
        radius = 36
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (round(cx), round(cy)), round(radius), 255, -1)
    points = cv2.goodFeaturesToTrack(gray, maxCorners=25, qualityLevel=0.003,
                                     minDistance=3, mask=mask, blockSize=5,
                                     useHarrisDetector=False)
    return points


def run_scene(scene, stride):
    rows = frame_detections(scene)
    video = cv2.VideoCapture(str(VIDEOS[scene]))
    previous_gray = None
    states = [None, None]
    errors = [[], []]
    fallback_frames = 0
    detector_anchor_frames = 0
    anchor_pair_frames = 0
    counts = []
    center_records = []
    elapsed = []
    flow_cpu_s = 0.0
    started = time.perf_counter()
    frame_i = 0
    while True:
        ok, side_by_side = video.read()
        if not ok:
            break
        flow_started = time.perf_counter()
        gray = cv2.resize(cv2.cvtColor(side_by_side[:, :1600], cv2.COLOR_BGR2GRAY),
                          (800, 600), interpolation=cv2.INTER_AREA)
        is_anchor = frame_i % stride == 0
        assigned, extras = assign_detections(rows[frame_i]['detections'])
        if is_anchor:
            detector_anchor_frames += 1
            anchor_pair_frames += sum(item is not None for item in assigned) == 2

        if previous_gray is not None:
            for target_i, state in enumerate(states):
                if state is None:
                    continue
                points = state['points']
                if points is None or len(points) == 0:
                    new_points = np.array(state['center'], dtype=np.float32).reshape(1, 1, 2)
                else:
                    new_points = points
                tracked, status, lk_error = cv2.calcOpticalFlowPyrLK(
                    previous_gray, gray, new_points, None,
                    winSize=FLOW_WINDOW, maxLevel=FLOW_MAX_LEVEL,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              20, 0.03), minEigThreshold=1e-4)
                if tracked is not None:
                    valid = status.reshape(-1).astype(bool)
                    if points is not None and len(points):
                        valid &= lk_error.reshape(-1) < 35
                    valid_points = tracked.reshape(-1, 2)[valid]
                else:
                    valid_points = np.empty((0, 2), dtype=np.float32)
                if len(valid_points) >= 1:
                    delta = valid_points - new_points.reshape(-1, 2)[valid]
                    dx, dy = np.median(delta, axis=0)
                    state['center'] = (state['center'][0] + float(dx),
                                       state['center'][1] + float(dy))
                    # Carry the tracked features forward. Re-detecting features
                    # every frame made this path CPU-heavy; detector anchors
                    # reseed the local feature set.
                    state['points'] = valid_points.reshape(-1, 1, 2)
                    state['flow_ok'] = len(valid_points) >= 3
                else:
                    state['points'] = None
                    state['flow_ok'] = False
        if is_anchor:
            for target_i, detection in enumerate(assigned):
                if detection is not None:
                    cx, cy, box = detection
                    cx *= FLOW_SCALE
                    cy *= FLOW_SCALE
                    scaled_box = {key: box[key] * FLOW_SCALE
                                  for key in ('x1', 'y1', 'x2', 'y2')}
                    states[target_i] = {
                        'center': (cx, cy),
                        'points': seed_points(gray, (cx, cy), scaled_box),
                        'last_anchor': frame_i,
                        'flow_ok': True,
                    }
                elif states[target_i] is not None and frame_i - states[target_i]['last_anchor'] > MAX_TRACK_AGE:
                    states[target_i] = None

        centers = []
        for target_i, state in enumerate(states):
            if state is None or frame_i - state['last_anchor'] > MAX_TRACK_AGE:
                centers.append(None)
                continue
            centers.append(state['center'])
            errors[target_i].append(math.dist(state['center'], FLOW_TARGETS[target_i]) /
                                    FLOW_SCALE)
        if any(value is not None and not states[i]['flow_ok']
               for i, value in enumerate(centers)):
            fallback_frames += 1
        counts.append(sum(value is not None for value in centers))
        center_records.append(centers)
        elapsed.append(frame_i / 60.0)
        previous_gray = gray
        flow_cpu_s += time.perf_counter() - flow_started
        frame_i += 1
    video.release()
    all_pair = sum(value == 2 for value in counts)
    return {
        'scene': scene,
        'stride': stride,
        'frames': frame_i,
        'elapsed_s': round(time.perf_counter() - started, 3),
        'per_frame_processing_s_excluding_decode': round(flow_cpu_s, 3),
        'processing_rate_excluding_decode_hz': round(frame_i / flow_cpu_s, 2)
        if flow_cpu_s else None,
        'detector_anchor_frames': detector_anchor_frames,
        'anchors_with_two_detections': anchor_pair_frames,
        'estimated_detector_anchor_rate_hz': 60.0 / stride,
        'output_rate_hz': 60.0,
        'output_count_histogram': {str(count): counts.count(count)
                                   for count in sorted(set(counts))},
        'exact_two_output_frames': all_pair,
        'exact_two_output_rate': round(all_pair / frame_i, 5) if frame_i else None,
        'flow_fallback_frames': fallback_frames,
        'center_error_px_by_target': [
            {'n': len(values), 'median': round(median(values), 3),
             'p95': round(sorted(values)[min(len(values) - 1,
                                             math.ceil(.95 * len(values)) - 1)], 3)
             if values else None}
            for values in errors],
        'limitations': 'Targets and camera are stationary in both saved clips. The same cached 1280 TensorRT detections provide anchor decisions; only every stride-th frame is used as a detector update. LK flow uses CPU OpenCV and its 60 Hz output rate is offline simulation, not a full Scout/NetworkTables timing result. Range is not recomputed between anchors. This cannot establish moving-ball, occlusion, crossed-target, or absolute-depth behavior.',
    }


output = {
    'method': 'Replay both 60 fps side-by-side clips frame by frame. At every 2nd, 3rd, or 4th frame, use cached confidence-0.05 1280 TensorRT detections as anchors. Between anchors, propagate each target center with pyramidal Lucas-Kanade sparse optical flow on half-resolution grayscale frames. Count tracks while their last detector anchor is at most six frames old.',
    'anchor_confidence': CONFIDENCE,
    'flow_image_scale': FLOW_SCALE,
    'flow_window_px': FLOW_WINDOW,
    'flow_pyramid_max_level': FLOW_MAX_LEVEL,
    'target_centers_px': TARGETS,
    'max_track_age_frames': MAX_TRACK_AGE,
    'results': [run_scene(scene, stride)
                for scene in ('earlier', 'corrected-settings')
                for stride in STRIDES],
}
path = ROOT / 'media/frc-fuel-flow-between-detector-anchors.json'
path.write_text(json.dumps(output, indent=2) + '\n')
for result in output['results']:
    print(result['scene'], 'stride', result['stride'],
          'exact-two', result['exact_two_output_rate'],
          'center', result['center_error_px_by_target'],
          'fallback', result['flow_fallback_frames'])
print('saved', path)
