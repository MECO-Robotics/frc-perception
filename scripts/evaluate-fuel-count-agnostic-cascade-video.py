#!/usr/bin/env python3
"""Compare two cascade fallback policies on both saved stereo videos."""
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
os.environ.update({
    'SCOUT_CLASS_NAMES': '["ball"]', 'SCOUT_MAX_BOX_ASPECT_RATIO': '2.0',
    'SCOUT_STEREO_ENABLED': '1', 'SCOUT_STEREO_RECTIFY': '1',
    'SCOUT_STEREO_CALIBRATION': 'media/calibration/stereo_calibration_live_order_candidate.yml',
    'SCOUT_STEREO_BASELINE_M': '0.0635', 'SCOUT_STEREO_DISPARITY_SCALE': '1',
    'SCOUT_STEREO_MATCH_SCALE': '0.5', 'SCOUT_STEREO_BLOCK_SIZE': '5',
    'SCOUT_STEREO_NUM_DISPARITIES': '128', 'SCOUT_STEREO_VERTICAL_ROI_PAD': '24',
})
from scout.detectors import Detection, TensorRTDetector
from scout.stereo import StereoDepth

VIDEOS = (
    ('earlier', ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr.mp4'),
    ('corrected-settings', ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'),
)
TARGETS = ((808.0, 589.0), (902.0, 591.0))
CONFIDENCE = 0.05
FAST_CANDIDATE_CONFIDENCE = 0.05
FALLBACK_MIN_CONFIDENCE = float(os.getenv('SCOUT_CASCADE_TEST_MIN_CONFIDENCE', '0.0575'))
MATCH_RADIUS_PX = 60.0
CLUSTER_RADIUS_M = 0.26


def select_policy(fast, full, count_agnostic):
    min_conf = min((b.confidence for b in fast), default=0.0)
    if count_agnostic:
        reason = 'empty_candidate' if not fast else None
    else:
        reason = 'candidate_count' if len(fast) != 2 else None
    if reason is None and min_conf < FALLBACK_MIN_CONFIDENCE:
        reason = 'weak_candidate'
    return (full if reason else fast), reason


def score_frame(boxes):
    selected = [None, None]
    extras = 0
    for box in sorted((b for b in boxes if b.confidence >= CONFIDENCE),
                      key=lambda b: b.confidence, reverse=True):
        cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        distance, index = min((math.hypot(cx - x, cy - y), i)
                              for i, (x, y) in enumerate(TARGETS))
        if distance <= MATCH_RADIUS_PX and selected[index] is None:
            selected[index] = box
        else:
            extras += 1
    return selected, extras


def medmad(values):
    if not values:
        return {'n': 0, 'median_m': None, 'mad_m': None, 'p95_m': None}
    median = statistics.median(values)
    return {'n': len(values), 'median_m': round(median, 5),
            'mad_m': round(statistics.median(abs(v - median) for v in values), 5),
            'p95_m': round(float(np.percentile(values, 95)), 5)}


def adaptive_select(fast, full, tracker, confirm_frames):
    count = len(fast)
    weakest = min((b.confidence for b in fast), default=0.0)
    if tracker['expected'] is None:
        reason = 'uninitialized_count'
    elif count != tracker['expected']:
        reason = 'count_changed'
    elif weakest < FALLBACK_MIN_CONFIDENCE:
        reason = 'weak_candidate'
    else:
        reason = None

    if reason is not None:
        full_count = len(full or [])
        if tracker['expected'] is None or full_count == tracker['expected']:
            tracker['expected'] = full_count
            tracker['pending'] = None
            tracker['pending_streak'] = 0
        elif confirm_frames == 1:
            tracker['expected'] = full_count
            tracker['pending'] = None
            tracker['pending_streak'] = 0
        else:
            if tracker['pending'] == full_count:
                tracker['pending_streak'] += 1
            else:
                tracker['pending'] = full_count
                tracker['pending_streak'] = 1
            if tracker['pending_streak'] >= confirm_frames:
                tracker['expected'] = full_count
                tracker['pending'] = None
                tracker['pending_streak'] = 0
    else:
        tracker['pending'] = None
        tracker['pending_streak'] = 0
    return (full if reason else fast), reason


def main():
    fast = TensorRTDetector(str(ROOT / 'models/rebuilt_fuel_yolo26n_640_fp16.engine'),
                            confidence=FAST_CANDIDATE_CONFIDENCE, iou=0.15)
    full = TensorRTDetector(str(ROOT / 'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'),
                            confidence=FAST_CANDIDATE_CONFIDENCE, iou=0.15)
    output = {
        'method': ('Run 640 candidates at confidence 0.05 on every decoded left-view '
                   'frame. Compare fixed-two, count-agnostic, and temporal expected-count '
                   'fallback policies. Temporal policies initialize/update their expected '
                   'count from 1280 detections, then fall back on fast count changes or '
                   'weak boxes; one policy accepts the first verified count change, and '
                   'the other requires that new high-resolution count twice. Score at '
                   'confidence 0.05 and run current rectified SGBM on policy outputs.'),
        'limitations': ('The videos contain the same two stationary targets and fixed '
                        'camera geometry. The labeled target centers are manual. API or '
                        'stereo ranges have no tape-measured ground truth; this tests '
                        'repeatability and count only, not motion or field diversity.'),
        'configuration': {'fast_engine': 'models/rebuilt_fuel_yolo26n_640_fp16.engine',
                          'fallback_engine': 'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine',
                          'fast_candidate_confidence': FAST_CANDIDATE_CONFIDENCE,
                          'score_confidence': CONFIDENCE, 'nms_iou': 0.15,
                          'fallback_min_confidence': FALLBACK_MIN_CONFIDENCE,
                          'temporal_count_policies': [
                              'update expected count immediately from fallback',
                              'update after two consecutive fallback counts'],
                          'stereo_block_size': 5, 'stereo_disparities': 128,
                          'stereo_vertical_roi_pad': 24,
                          'cluster_radius_m': CLUSTER_RADIUS_M},
        'results': [],
    }
    try:
        for scene, video_path in VIDEOS:
            cap = cv2.VideoCapture(str(video_path))
            frame_count = 0
            names = ('two_ball', 'count_agnostic', 'temporal_count_immediate',
                     'temporal_count_confirm2')
            state = {name: {'tp': 0, 'fp': 0, 'fn': 0, 'paired': 0, 'exact_two': 0,
                            'fallback_reasons': {}, 'fallback_frames': 0,
                            'detector_ms': [], 'stereo_ms': [],
                            'ranges': [[], []], 'pair_spacings': [],
                            'count_histogram': {}}
                     for name in names}
            stereo_by_policy = {name: StereoDepth() for name in names}
            temporal_trackers = {
                'temporal_count_immediate': {'expected': None, 'pending': None,
                                             'pending_streak': 0},
                'temporal_count_confirm2': {'expected': None, 'pending': None,
                                            'pending_streak': 0},
            }
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                left, right = frame[:, :1600], frame[:, 1600:]
                fast_boxes, fast_ms = fast.infer(left)
                orig_fb = len(fast_boxes) != 2 or min(
                    (b.confidence for b in fast_boxes), default=0.0) < FALLBACK_MIN_CONFIDENCE
                agnostic_fb = not fast_boxes or min(
                    (b.confidence for b in fast_boxes), default=0.0) < FALLBACK_MIN_CONFIDENCE
                temporal_fb = {}
                for name, tracker in temporal_trackers.items():
                    weak = min((b.confidence for b in fast_boxes), default=0.0) < FALLBACK_MIN_CONFIDENCE
                    temporal_fb[name] = (tracker['expected'] is None or
                                         len(fast_boxes) != tracker['expected'] or weak)
                full_boxes = None
                full_ms = 0.0
                if orig_fb or agnostic_fb or any(temporal_fb.values()):
                    full_boxes, full_ms = full.infer(left)
                profiles = {
                    'two_ball': select_policy(fast_boxes, full_boxes, False),
                    'count_agnostic': select_policy(fast_boxes, full_boxes, True),
                }
                profiles['temporal_count_immediate'] = adaptive_select(
                    fast_boxes, full_boxes, temporal_trackers['temporal_count_immediate'], 1)
                profiles['temporal_count_confirm2'] = adaptive_select(
                    fast_boxes, full_boxes, temporal_trackers['temporal_count_confirm2'], 2)
                for name, (boxes, reason) in profiles.items():
                    a = state[name]
                    a['fallback_reasons'][reason or 'none'] = (
                        a['fallback_reasons'].get(reason or 'none', 0) + 1)
                    a['fallback_frames'] += reason is not None
                    a['detector_ms'].append(fast_ms + (full_ms if reason else 0.0))
                    selected, extras = score_frame(boxes)
                    detected_count = sum(b is not None for b in selected) + extras
                    a['count_histogram'][str(detected_count)] = (
                        a['count_histogram'].get(str(detected_count), 0) + 1)
                    hits = sum(b is not None for b in selected)
                    a['tp'] += hits
                    a['fn'] += 2 - hits
                    a['fp'] += extras
                    a['paired'] += hits == 2
                    a['exact_two'] += hits == 2 and extras == 0
                    metric_boxes = [Detection(b.x1, b.y1, b.x2, b.y2,
                                              b.class_id, b.confidence)
                                    for b in selected if b is not None]
                    if metric_boxes:
                        started = time.perf_counter()
                        stereo_by_policy[name].estimate(left, right, metric_boxes)
                        a['stereo_ms'].append((time.perf_counter() - started) * 1000)
                        by_target = [None, None]
                        for box in metric_boxes:
                            cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
                            _, index = min((math.hypot(cx - x, cy - y), i)
                                            for i, (x, y) in enumerate(TARGETS))
                            by_target[index] = box
                            if box.distance_m is not None:
                                a['ranges'][index].append(float(box.distance_m))
                        if all(b is not None and b.distance_m is not None for b in by_target):
                            # Pair spacing is approximated from the range difference
                            # and horizontal image-center separation in rectified pixels.
                            z1, z2 = by_target[0].distance_m, by_target[1].distance_m
                            q = np.array([[
                                [(by_target[0].x1 + by_target[0].x2) / 2,
                                 (by_target[0].y1 + by_target[0].y2) / 2],
                                [(by_target[1].x1 + by_target[1].x2) / 2,
                                 (by_target[1].y1 + by_target[1].y2) / 2],
                            ]], dtype=np.float32)
                            s = stereo_by_policy[name]
                            c = s.calibration
                            _, _, _, kl, r1, p1 = s.maps
                            pts = cv2.undistortPoints(
                                q, kl, c['left_distortion'], R=r1, P=p1).reshape(-1, 2)
                            xyz = []
                            for point, z in zip(pts, (z1, z2)):
                                xyz.append(((point[0] - p1[0, 2]) * z / p1[0, 0],
                                            (point[1] - p1[1, 2]) * z / p1[1, 1], z))
                            a['pair_spacings'].append(float(np.linalg.norm(
                                np.asarray(xyz[0]) - np.asarray(xyz[1]))))
                frame_count += 1
            cap.release()
            summarized = {}
            for name, a in state.items():
                p = a['tp'] / (a['tp'] + a['fp']) if a['tp'] + a['fp'] else 0.0
                r = a['tp'] / (a['tp'] + a['fn']) if a['tp'] + a['fn'] else 0.0
                values = a['detector_ms']
                summarized[name] = {
                    'frames': frame_count, 'tp': a['tp'], 'fp': a['fp'], 'fn': a['fn'],
                    'paired_frames': a['paired'], 'exact_two_frames': a['exact_two'],
                    'precision': round(p, 5), 'recall': round(r, 5),
                    'f1': round(2 * p * r / (p + r), 5) if p + r else 0.0,
                    'fallback_frames': a['fallback_frames'],
                    'fallback_fraction': round(a['fallback_frames'] / frame_count, 5),
                    'fallback_reasons': a['fallback_reasons'],
                    'detector_ms_median': round(statistics.median(values), 2) if values else None,
                    'detector_ms_p95': round(float(np.percentile(values, 95)), 2) if values else None,
                    'stereo_ms_median': round(statistics.median(a['stereo_ms']), 2) if a['stereo_ms'] else None,
                    'ranges': [medmad(v) for v in a['ranges']],
                    'pair_spacing': medmad(a['pair_spacings']),
                    'count_histogram': a['count_histogram'],
                }
            output['results'].append({'scene': scene, 'video': str(video_path.relative_to(ROOT)),
                                      'frames': frame_count, 'profiles': summarized})
            print(scene, json.dumps(summarized), flush=True)
    finally:
        fast.close()
        full.close()
    tag = f'{FALLBACK_MIN_CONFIDENCE:g}'.replace('.', 'p')
    path = ROOT / f'media/frc-fuel-count-agnostic-cascade-video-minconf-{tag}.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print('saved', path)


if __name__ == '__main__':
    main()
