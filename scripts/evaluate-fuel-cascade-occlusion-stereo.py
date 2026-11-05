#!/usr/bin/env python3
"""Measure detection count and stereo range through cascade occlusion samples."""
import json
import math
import os
import statistics
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
INPUT = ROOT / 'media/frc-fuel-resolution-cascade-occlusion-evaluation.json'
CAL = 'media/calibration/stereo_calibration_live_order_candidate.yml'
os.environ.update({
    'SCOUT_CLASS_NAMES': '["ball"]', 'SCOUT_CONFIDENCE': '0.05',
    'SCOUT_STEREO_ENABLED': '1', 'SCOUT_STEREO_RECTIFY': '1',
    'SCOUT_STEREO_CALIBRATION': str(ROOT / CAL),
    'SCOUT_STEREO_BASELINE_M': '0.0635', 'SCOUT_STEREO_DISPARITY_SCALE': '1',
    'SCOUT_STEREO_MATCH_SCALE': '0.5', 'SCOUT_STEREO_NUM_DISPARITIES': '128',
    'SCOUT_STEREO_VERTICAL_ROI_PAD': '24', 'SCOUT_STEREO_HORIZONTAL_ROI': '0',
    'SCOUT_STEREO_ADAPTIVE_DISPARITY_WIDTH': '0', 'SCOUT_FUEL_DIAMETER_M': '0.15',
    'SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR': '0.55',
    'SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU': '0.90',
})
from scout.detectors import Detection
from scout.stereo import StereoDepth

TARGETS = ((808.0, 589.0), (902.0, 591.0))
MATCH_RADIUS = 60


def match(boxes):
    chosen = [None, None]
    extras = 0
    for box in sorted(boxes, key=lambda item: item.confidence, reverse=True):
        cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        distance, index = min((math.dist((cx, cy), target), i)
                              for i, target in enumerate(TARGETS))
        if distance <= MATCH_RADIUS and chosen[index] is None:
            chosen[index] = box
        else:
            extras += 1
    return chosen, extras


def stats(values):
    if not values:
        return {'n': 0}
    return {'n': len(values), 'median_m': round(statistics.median(values), 5),
            'mad_m': round(statistics.median(abs(x - statistics.median(values))
                                             for x in values), 5),
            'p95_abs_m': round(sorted(abs(x) for x in values)[
                math.ceil(.95 * len(values)) - 1], 5)}


def boxes(items):
    return [Detection(int(x['x1']), int(x['y1']), int(x['x2']), int(x['y2']),
                      0, float(x['confidence'])) for x in items]


data = json.loads(INPUT.read_text())
dense = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
output = {
    'method': ('For the same 15 timestamp/mask cases in each saved clip, use the '
               'saved 640 and 1280 detections (640-first falls back when count is '
               'not two or minimum confidence is below 0.06). Estimate stereo '
               'depth on the original right image and masked left image with the '
               'active rectified 128-disparity, 24 px band, 55% size gate. Compare '
               'each masked estimate with its same-timestamp unmasked estimate.'),
    'limitations': ('Synthetic lower-box masks affect the left view only. Fixed '
                    'target windows and 15 samples per condition; no real '
                    'foreground parallax or metric range ground truth.'),
    'settings': {'baseline_m': 0.0635, 'disparity_scale': 1.0,
                 'match_scale': 0.5, 'num_disparities': 128,
                 'vertical_roi_pad': 24, 'fuel_diameter_m': 0.15,
                 'size_gate_relative_tolerance': 0.55,
                 'cascade_min_confidence': 0.06},
    'results': [],
}

for clip in data['results']:
    dense_clip = next(item for item in dense['results'] if item['scene'] == clip['scene'])
    capture = cv2.VideoCapture(str(ROOT / clip['video']))
    clean_rows = {row['frame']: row for row in clip['fractions'][0]['frames_detail']}
    summaries = []
    for fraction_data in clip['fractions']:
        fraction = fraction_data['occlusion_fraction']
        baseline_ranges = {}
        records = {name: {'target_counts': [], 'range_deltas': [[], []],
                          'valid_depth_count': 0, 'gate_rejected': 0,
                          'duplicate_rejected': 0, 'stereo_ms': []}
                   for name in ('full_1280', 'cascade_640_first')}
        for row in fraction_data['frames_detail']:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(row['frame']))
            ok, frame = capture.read()
            if not ok:
                continue
            left, right = frame[:, :1600].copy(), frame[:, 1600:]
            # Apply the same masks to clean matched target boxes as the detector evaluation.
            if fraction:
                source = dense_clip['per_frame'][int(row['frame'])]['detections']
                clean_targets, _ = match([Detection(int(b['x1']), int(b['y1']),
                    int(b['x2']), int(b['y2']), 0, float(b['confidence']))
                    for b in source if b['confidence'] >= 0.10])
                for box in clean_targets:
                    if box is None:
                        continue
                    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
                    top = max(0, y2 - max(1, round((y2 - y1) * fraction)))
                    left[top:y2, x1:x2] = 0
            detected = {
                'full_1280': boxes(row['full_boxes']),
            }
            fast = boxes(row['fast_boxes'])
            fallback = (len(fast) != 2 or min((b.confidence for b in fast), default=0) < .06)
            detected['cascade_640_first'] = boxes(row['full_boxes']) if fallback else fast
            for name, detections in detected.items():
                estimator = StereoDepth()
                started = __import__('time').perf_counter()
                estimator.estimate(left, right, detections)
                records[name]['stereo_ms'].append((__import__('time').perf_counter() - started) * 1000)
                detections = estimator.filter_fuel_size_inconsistent(detections)
                records[name]['gate_rejected'] += estimator.last_size_gate_rejected
                records[name]['duplicate_rejected'] += estimator.last_size_gate_duplicate_rejected
                targets, _ = match(detections)
                valid = [b for b in targets if b is not None and b.distance_m is not None]
                records[name]['target_counts'].append(len(valid))
                records[name]['valid_depth_count'] += len(valid)
                for index, box in enumerate(targets):
                    if box is None or box.distance_m is None:
                        continue
                    baseline_row = clean_rows[row['frame']]
                    # Zero-mask clip rows are the clean reference; run its stereo
                    # once per frame/name and cache the resulting target ranges.
                    key = (row['frame'], name)
                    if key not in baseline_ranges:
                        clean_frame = frame.copy()
                        if name == 'full_1280':
                            clean_dets = boxes(baseline_row['full_boxes'])
                        else:
                            clean_fast = boxes(baseline_row['fast_boxes'])
                            clean_fallback = (len(clean_fast) != 2 or
                                min((b.confidence for b in clean_fast), default=0) < .06)
                            clean_dets = (boxes(baseline_row['full_boxes']) if clean_fallback
                                          else clean_fast)
                        clean_estimator = StereoDepth()
                        clean_estimator.estimate(clean_frame[:, :1600], clean_frame[:, 1600:], clean_dets)
                        clean_estimator.filter_fuel_size_inconsistent(clean_dets)
                        baseline_ranges[key] = [b.distance_m for b in match(clean_dets)[0]]
                    reference = baseline_ranges[key][index]
                    if reference is not None:
                        records[name]['range_deltas'][index].append(float(box.distance_m - reference))
        policies = {}
        for name, record in records.items():
            counts = record['target_counts']
            policies[name] = {
                'frames': len(counts), 'target_count_histogram_after_gate': {
                    str(n): counts.count(n) for n in range(3)},
                'valid_depth_instances': record['valid_depth_count'],
                'gate_rejected': record['gate_rejected'],
                'duplicate_rejected': record['duplicate_rejected'],
                'range_delta_masked_minus_unmasked_m_by_target': [
                    stats(values) for values in record['range_deltas']],
                'stereo_estimate_ms_median': round(statistics.median(record['stereo_ms']), 2)
                    if record['stereo_ms'] else None,
            }
        summaries.append({'occlusion_fraction': fraction, 'policies': policies})
        print(clip['scene'], fraction, json.dumps(policies))
    capture.release()
    output['results'].append({'scene': clip['scene'], 'video': clip['video'],
                              'conditions': summaries})

path = ROOT / 'media/frc-fuel-cascade-occlusion-stereo-evaluation.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print('saved', path)
