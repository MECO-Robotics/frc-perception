#!/usr/bin/env python3
"""Test VPI CUDA KLT box tracking for per-frame stereo range updates."""
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
BOX_DTYPE = np.dtype(vpi.Type.KLT_TRACKED_BOUNDING_BOX)
cv2.setNumThreads(1)


def median(values):
    return statistics.median(values) if values else None


def make_rectifier(stereo):
    c = stereo.calibration
    size = (1600, 1200)
    kl, kr = c['left_camera_matrix'].copy(), c['right_camera_matrix'].copy()
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
    raw = np.asarray([[[raw_half[0] / SCALE, raw_half[1] / SCALE]]], np.float32)
    if camera == 0:
        out = cv2.undistortPoints(raw, kl, stereo.calibration['left_distortion'],
                                  R=r1, P=p1)[0, 0]
    else:
        out = cv2.undistortPoints(raw, kr, stereo.calibration['right_distortion'],
                                  R=r2, P=p2)[0, 0]
    return float(out[0]) * SCALE, float(out[1]) * SCALE


def right_raw_from_rectified(point, rectifier):
    mr = rectifier[-1]
    x = int(np.clip(round(point[0] / SCALE), 0, mr[0].shape[1] - 1))
    y = int(np.clip(round(point[1] / SCALE), 0, mr[0].shape[0] - 1))
    return float(mr[0][y, x]) * SCALE, float(mr[1][y, x]) * SCALE


def tracker_boxes(boxes, stereo, rectifier):
    """Construct left/right cropped box templates in target identity order."""
    assigned = [None, None]
    for box in sorted(boxes, key=lambda item: item.confidence, reverse=True):
        center = ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)
        target_i = min(range(2), key=lambda i: math.dist(center, TARGETS[i]))
        if math.dist(center, TARGETS[target_i]) <= 60 and assigned[target_i] is None:
            assigned[target_i] = box
    if any(box is None or not box.disparity_px for box in assigned):
        return None
    per_view = [[], []]
    target_depth = []
    for box in assigned:
        cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        width = max(10.0, (box.x2 - box.x1) * SCALE * 0.82)
        height = max(10.0, (box.y2 - box.y1) * SCALE * 0.82)
        left_center = (cx * SCALE, cy * SCALE)
        left_rect = rectify_point(left_center, 0, stereo, rectifier)
        right_rect = (left_rect[0] - box.disparity_px * SCALE, left_rect[1])
        right_center = right_raw_from_rectified(right_rect, rectifier)
        per_view[0].append((left_center[0] - width/2, left_center[1] - height/2,
                            width, height))
        per_view[1].append((right_center[0] - width/2, right_center[1] - height/2,
                            width, height))
        target_depth.append({'disparity_px': box.disparity_px,
                             'range_m': box.distance_m})
    return per_view, target_depth


def create_tracker(frame, boxes):
    array = np.zeros(len(boxes), dtype=BOX_DTYPE)
    for i, (x, y, width, height) in enumerate(boxes):
        matrix = array['bbox']['xform']['mat3'][i]
        matrix[0, 0] = matrix[1, 1] = matrix[2, 2] = 1.0
        matrix[0, 2], matrix[1, 2] = x, y
        array['bbox']['width'][i], array['bbox']['height'][i] = width, height
        array['tracking_status'][i] = 0
        array['template_status'][i] = 1
    initial = vpi.asarray(array, type=vpi.Type.KLT_TRACKED_BOUNDING_BOX)
    with vpi.Backend.CUDA:
        return vpi.KLTFeatureTracker(frame, initial, backend=vpi.Backend.CUDA)


def read_box_output(output):
    with output.rlock_cpu() as data:
        boxes = np.array(data, copy=True)
    result = []
    for item in boxes:
        matrix = item['bbox']['xform']['mat3']
        status = int(item['tracking_status'])
        x, y = float(matrix[0, 2]), float(matrix[1, 2])
        width = float(item['bbox']['width'] * matrix[0, 0])
        height = float(item['bbox']['height'] * matrix[1, 1])
        result.append({'center': (x + width/2, y + height/2),
                       'valid': status == 0,
                       'bbox': (x, y, width, height)})
    return result


def run_scene(scene):
    det_rows = next(item for item in DETECTIONS['results']
                    if item['scene'] == scene)['per_frame']
    depth_rows = next(item for item in DENSE_DEPTH['results']
                      if item['scene'] == scene)['frames']
    video = cv2.VideoCapture(str(VIDEOS[scene]))
    stereo = StereoDepth()
    rectifier = make_rectifier(stereo)
    trackers = [None, None]
    outputs = [None, None]
    last_detected = [None, None]
    ranges = [[], []]
    range_deltas = [[], []]
    count_hist = {}
    lost = 0
    frame_i = 0
    processing_s = 0.0
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
            current_frames = (vpi.asimage(left_gray, vpi.Format.U8),
                              vpi.asimage(right_gray, vpi.Format.U8))

        if trackers[0] is not None:
            outputs = [read_box_output(trackers[i](current_frames[i]))
                       for i in range(2)]

        is_detector_anchor = frame_i % DETECTOR_STRIDE == 0
        is_stereo_anchor = frame_i % STEREO_STRIDE == 0
        boxes = []
        if is_detector_anchor:
            for item in det_rows[frame_i]['detections']:
                if item['confidence'] >= CONFIDENCE:
                    boxes.append(Detection(item['x1'], item['y1'], item['x2'], item['y2'],
                                           0, item['confidence']))
            assigned = [None, None]
            for box in sorted(boxes, key=lambda item: item.confidence, reverse=True):
                center = ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)
                target_i = min(range(2), key=lambda i: math.dist(center, TARGETS[i]))
                if math.dist(center, TARGETS[target_i]) <= 60 and assigned[target_i] is None:
                    assigned[target_i] = box
                    last_detected[target_i] = frame_i

        if is_stereo_anchor and boxes:
            stereo.estimate(left, right, boxes)
            seed = tracker_boxes(boxes, stereo, rectifier)
            if seed is not None:
                per_view, _ = seed
                trackers = [create_tracker(current_frames[i], per_view[i]) for i in range(2)]
                # KLT starts with the detector rectangles in this current frame.
                outputs = [[{'center': (x+w/2, y+h/2), 'valid': True,
                             'bbox': (x, y, w, h)} for x, y, w, h in per_view[i]]
                           for i in range(2)]

        valid_ranges = [None, None]
        for target_i in range(2):
            if (last_detected[target_i] is None or
                    frame_i - last_detected[target_i] > MAX_AGE or
                    outputs[0] is None or outputs[1] is None or
                    target_i >= len(outputs[0]) or target_i >= len(outputs[1])):
                continue
            left_box, right_box = outputs[0][target_i], outputs[1][target_i]
            if not (left_box['valid'] and right_box['valid']):
                lost += 1
                continue
            left_rect = rectify_point(left_box['center'], 0, stereo, rectifier)
            right_rect = rectify_point(right_box['center'], 1, stereo, rectifier)
            disparity = (left_rect[0] - right_rect[0]) / SCALE
            if disparity <= 0:
                continue
            valid_ranges[target_i] = stereo.baseline_m * stereo.focal_px / disparity
            ranges[target_i].append(valid_ranges[target_i])
            reference = depth_rows[frame_i]['ranges_m'][target_i]
            if reference is not None:
                range_deltas[target_i].append(abs(valid_ranges[target_i] - reference))
        count = sum(value is not None for value in valid_ranges)
        count_hist[str(count)] = count_hist.get(str(count), 0) + 1
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
        'lost_box_outputs': lost,
        'ranges': [
            {'n': len(values), 'median_m': median(values),
             'mad_m': median([abs(value - median(values)) for value in values])
             if values else None,
             'median_abs_delta_vs_same_frame_sgbm_m': median(range_deltas[i]),
             'p95_abs_delta_vs_same_frame_sgbm_m': sorted(range_deltas[i])[
                 min(len(range_deltas[i]) - 1,
                     math.ceil(.95 * len(range_deltas[i])) - 1)]
             if range_deltas[i] else None}
            for i, values in enumerate(ranges)],
        'limitations': 'Stationary scenes only. Cached 1280 TensorRT detections are used at 20 Hz and SGBM at 6.67 Hz; VPI CUDA KLT tracks box templates in each raw view between anchors. Per-frame depth is compared with cached same-frame SGBM, not measured range truth. Motion, crossings, occlusion, and concurrent Scout runtime are untested.',
    }


output = {
    'method': 'At every ninth frame, use cached 1280 TensorRT detections plus runtime SGBM to initialize two KLT box trackers per camera view. Between depth anchors, run VPI CUDA KLT box tracking each frame and recompute depth from the rectified center disparity. Count/state freshness is checked against detector anchors every third frame.',
    'vpi_version': '2.4.8',
    'anchor_confidence': CONFIDENCE,
    'max_detection_age_frames': MAX_AGE,
    'results': [run_scene(scene) for scene in ('earlier', 'corrected-settings')],
}
path = ROOT / 'media/frc-fuel-vpi-klt-box-stereo-flow-evaluation.json'
path.write_text(json.dumps(output, indent=2) + '\n')
for result in output['results']:
    print(result['scene'], result['count_histogram'], result['ranges'],
          result['processing_rate_excluding_decode_hz'])
print('saved', path)
