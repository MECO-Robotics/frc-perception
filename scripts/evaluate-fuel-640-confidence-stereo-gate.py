#!/usr/bin/env python3
"""Test whether stereo size/depth consistency rescues low-confidence 640 boxes."""
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
THRESHOLDS = (.01, .02, .03, .04, .05, .06, .075, .10, .15, .20)
TARGETS = ((808.0, 589.0), (902.0, 591.0))
MATCH_RADIUS = 60.0
ENGINE = 'models/rebuilt_fuel_yolo26n_640_fp16.engine'

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


def score(boxes, threshold):
    matched = [None, None]
    extras = 0
    for box in sorted((b for b in boxes if b.confidence >= threshold),
                      key=lambda item: item.confidence, reverse=True):
        cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        distances = [math.hypot(cx - x, cy - y) for x, y in TARGETS]
        target = min(range(2), key=distances.__getitem__)
        if distances[target] <= MATCH_RADIUS and matched[target] is None:
            matched[target] = box
        else:
            extras += 1
    hits = sum(box is not None for box in matched)
    return hits, extras, matched


def describe_range(values):
    return {'n': len(values), 'median_m': round(med(values), 4) if values else None,
            'mad_m': round(mad(values), 4) if values else None}


detector = TensorRTDetector(str(ROOT / ENGINE), confidence=.01, iou=.10,
                            reuse_preprocess=True)
stereo = StereoDepth()
output = {
    'method': ('Run the FRC9550 640 TensorRT engine at confidence 0.01 and NMS '
               'IoU 0.10 once per frame. Estimate stereo depth for all surviving '
               'boxes, apply the active 55% nominal-FUEL-size consistency gate '
               'and 0.90 duplicate suppression, then rescore confidence cutoffs.'),
    'engine': ENGINE,
    'settings': {'candidate_confidence': .01, 'nms_iou': .10,
                 'stereo_disparities': 128, 'vertical_roi_pad_px': 24,
                 'horizontal_roi': True, 'size_gate_relative_tolerance': .55,
                 'duplicate_iou': .90},
    'limitations': ('Only two stationary target windows and one camera pose per '
                    'clip. The size gate uses the nominal 15 cm diameter and '
                    'approximate stereo calibration; this is not measured range '
                    'truth or a test of unrelated clusters.'),
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
        frames = 0
        detector_ms, stereo_ms = [], []
        rejected = duplicate_rejected = candidate_total = kept_total = 0
        totals = {threshold: {'hist': {0: 0, 1: 0, 2: 0},
                              'exact': 0, 'paired': 0, 'tp': 0,
                              'fp': 0, 'fn': 0, 'ranges': [[], []]}
                  for threshold in THRESHOLDS}
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            left, right = frame[:, :1600], frame[:, 1600:]
            boxes, elapsed = detector.infer(left)
            detector_ms.append(elapsed)
            candidate_total += len(boxes)
            start = time.perf_counter()
            stereo.estimate(left, right, boxes)
            stereo_ms.append((time.perf_counter() - start) * 1000)
            filtered = stereo.filter_fuel_size_inconsistent(boxes)
            rejected += stereo.last_size_gate_rejected
            duplicate_rejected += stereo.last_size_gate_duplicate_rejected
            kept_total += len(filtered)
            for threshold in THRESHOLDS:
                hits, extras, matched = score(filtered, threshold)
                stats = totals[threshold]
                stats['hist'][hits] += 1
                stats['exact'] += hits == 2 and extras == 0
                stats['paired'] += hits == 2
                stats['tp'] += hits
                stats['fp'] += extras
                stats['fn'] += 2 - hits
                for index, box in enumerate(matched):
                    if box is not None and box.distance_m is not None:
                        stats['ranges'][index].append(float(box.distance_m))
        capture.release()

        summaries = []
        for threshold in THRESHOLDS:
            stats = totals[threshold]
            precision = stats['tp'] / (stats['tp'] + stats['fp']) if stats['tp'] + stats['fp'] else 0
            recall = stats['tp'] / (stats['tp'] + stats['fn']) if stats['tp'] + stats['fn'] else 0
            summaries.append({
                'confidence': threshold,
                'count_histogram': {str(k): v for k, v in stats['hist'].items()},
                'exact_two_frames': stats['exact'], 'paired_frames': stats['paired'],
                'tp': stats['tp'], 'fp': stats['fp'], 'fn': stats['fn'],
                'precision': round(precision, 5), 'recall': round(recall, 5),
                'f1': round(2 * precision * recall / (precision + recall), 5)
                if precision + recall else 0,
                'range_by_target_m': [describe_range(values) for values in stats['ranges']],
            })
        result = {
            'scene': clip['scene'], 'video': clip['video'], 'frames': frames,
            'candidate_boxes_total': candidate_total, 'boxes_after_stereo_size_gate': kept_total,
            'size_gate_rejected_boxes': rejected,
            'duplicate_suppressed_boxes': duplicate_rejected,
            'detector_latency_ms': {'median': round(med(detector_ms), 2),
                                    'mean': round(statistics.mean(detector_ms), 2)},
            'stereo_latency_ms': {'median': round(med(stereo_ms), 2),
                                  'mean': round(statistics.mean(stereo_ms), 2)},
            'threshold_sweep': summaries,
        }
        output['results'].append(result)
        print(clip['scene'], 'gate rejected', rejected, 'duplicates', duplicate_rejected,
              'median ms detector/stereo', result['detector_latency_ms']['median'],
              result['stereo_latency_ms']['median'])
        for row in summaries:
            print(row['confidence'], 'exact', row['exact_two_frames'],
                  'pair', row['paired_frames'], 'fp/fn', row['fp'], row['fn'],
                  'ranges', row['range_by_target_m'])
finally:
    detector.close()

destination = ROOT / 'media/frc-fuel-640-confidence-stereo-gate-evaluation.json'
destination.write_text(json.dumps(output, indent=2) + '\n')
print('saved', destination)
