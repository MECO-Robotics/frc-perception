#!/usr/bin/env python3
"""Evaluate a 640-only candidate-count confidence cascade with stereo gating."""
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLIPS = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())['results']
TARGETS = ((808.0, 589.0), (902.0, 591.0))
MATCH_RADIUS = 60.0
ENGINE = 'models/rebuilt_fuel_yolo26n_640_fp16.engine'
BASE_CONFIDENCE = .01
TRIGGER = 3
HIGH_CONFIDENCE = .075

cv2.setNumThreads(1)
os.environ.update(
    SCOUT_CLASS_NAMES='["ball"]', SCOUT_MAX_BOX_ASPECT_RATIO='2.0',
    SCOUT_STEREO_ENABLED='1',
    SCOUT_STEREO_CALIBRATION='media/calibration/stereo_calibration_live_order_candidate.yml',
    SCOUT_STEREO_RECTIFY='1', SCOUT_STEREO_BASELINE_M='.0635',
    SCOUT_STEREO_DISPARITY_SCALE='1', SCOUT_STEREO_MATCH_SCALE='.5',
    SCOUT_STEREO_NUM_DISPARITIES='128', SCOUT_STEREO_VERTICAL_ROI_PAD='24',
    SCOUT_STEREO_HORIZONTAL_ROI='1', SCOUT_STEREO_ADAPTIVE_DISPARITY_WIDTH='0',
    SCOUT_FUEL_DIAMETER_M='.15', SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR='.55',
    SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU='.90', SCOUT_MAX_CANDIDATES='0')
from scout.detectors import TensorRTDetector
from scout.stereo import StereoDepth


def med(values):
    return statistics.median(values) if values else None


def mad(values):
    center = med(values)
    return med([abs(value - center) for value in values]) if values else None


def summary(values):
    return {'n': len(values), 'median': round(med(values), 4) if values else None,
            'mad': round(mad(values), 4) if values else None}


def match(boxes):
    result = [None, None]
    extras = 0
    for box in sorted(boxes, key=lambda item: item.confidence, reverse=True):
        cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        distances = [math.hypot(cx - x, cy - y) for x, y in TARGETS]
        index = min(range(2), key=distances.__getitem__)
        if distances[index] <= MATCH_RADIUS and result[index] is None:
            result[index] = box
        else:
            extras += 1
    return result, extras


detector = TensorRTDetector(str(ROOT / ENGINE), confidence=BASE_CONFIDENCE,
                            iou=.10, reuse_preprocess=True)
stereo = StereoDepth()
output = {
    'method': ('Run the FRC9550 640 TensorRT engine at candidate confidence 0.01. '
               'If at least three post-NMS candidates remain, raise that frame to '
               'confidence 0.075. Run rectified stereo only on selected boxes, '
               'then apply the 55% FUEL size/depth gate and duplicate suppression.'),
    'engine': ENGINE,
    'settings': {'candidate_confidence': BASE_CONFIDENCE, 'nms_iou': .10,
                 'adaptive_trigger': TRIGGER, 'adaptive_high_confidence': HIGH_CONFIDENCE,
                 'stereo_disparities': 128, 'vertical_roi_pad_px': 24,
                 'horizontal_roi': True, 'size_gate_relative_tolerance': .55,
                 'duplicate_iou': .90},
    'limitations': ('Only two stationary target windows per clip; the confidence '
                    'trigger is tuned on this pair, and range agreement is not '
                    'absolute accuracy because no tape-measured distance exists.'),
    'results': [],
}

try:
    for clip in CLIPS:
        capture = cv2.VideoCapture(str(ROOT / clip['video']))
        ok, warm = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read {clip['video']}")
        detector.infer(warm[:, :1600])
        capture.release()

        capture = cv2.VideoCapture(str(ROOT / clip['video']))
        counts = {0: 0, 1: 0, 2: 0}
        exact = pairs = tp = fp = fn = 0
        ranges = [[], []]
        fallback_frames = removed = gate_rejected = duplicate_rejected = 0
        input_boxes = selected_boxes_total = 0
        detector_ms = []
        stereo_ms = []
        frames = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            left, right = frame[:, :1600], frame[:, 1600:]
            boxes, latency = detector.infer(left)
            detector_ms.append(latency)
            input_boxes += len(boxes)
            if len(boxes) >= TRIGGER:
                fallback_frames += 1
                before = len(boxes)
                boxes = [box for box in boxes if box.confidence >= HIGH_CONFIDENCE]
                removed += before - len(boxes)
            selected_boxes_total += len(boxes)
            start = time.perf_counter()
            stereo.estimate(left, right, boxes)
            stereo_ms.append((time.perf_counter() - start) * 1000)
            boxes = stereo.filter_fuel_size_inconsistent(boxes)
            gate_rejected += stereo.last_size_gate_rejected
            duplicate_rejected += stereo.last_size_gate_duplicate_rejected
            targets, extras = match(boxes)
            hit_count = sum(box is not None for box in targets)
            counts[hit_count] += 1
            exact += hit_count == 2 and extras == 0
            pairs += hit_count == 2
            tp += hit_count
            fp += extras
            fn += 2 - hit_count
            for index, box in enumerate(targets):
                if box is not None and box.distance_m is not None:
                    ranges[index].append(float(box.distance_m))
        capture.release()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        record = {
            'scene': clip['scene'], 'video': clip['video'], 'frames': frames,
            'exact_two_frames': exact, 'paired_frames': pairs,
            'count_histogram': {str(k): v for k, v in counts.items()},
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(precision, 5), 'recall': round(recall, 5),
            'f1': round(2 * precision * recall / (precision + recall), 5)
            if precision + recall else 0,
            'adaptive_triggered_frames': fallback_frames,
            'adaptive_boxes_removed': removed,
            'size_gate_rejected': gate_rejected,
            'duplicate_suppressed': duplicate_rejected,
            'candidate_boxes_total': input_boxes,
            'boxes_sent_to_stereo': selected_boxes_total,
            'range_by_target_m': [summary(values) for values in ranges],
            'detector_latency_ms_median': round(med(detector_ms), 2),
            'stereo_latency_ms_median': round(med(stereo_ms), 2),
            'detector_plus_stereo_median_ms': round(med(detector_ms) + med(stereo_ms), 2),
        }
        output['results'].append(record)
        print(json.dumps(record))
finally:
    detector.close()

destination = ROOT / 'media/frc-fuel-640-adaptive-stereo-evaluation.json'
destination.write_text(json.dumps(output, indent=2) + '\n')
print('saved', destination)
