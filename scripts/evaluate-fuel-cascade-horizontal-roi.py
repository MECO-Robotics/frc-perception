#!/usr/bin/env python3
"""A/B stereo ROI on saved 640-first cascade detections and stereo videos."""
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
CASCADE = json.loads((ROOT / 'media/frc-fuel-resolution-cascade-runtime.json').read_text())
CALIBRATION = 'media/calibration/stereo_calibration_live_order_candidate.yml'
CONFIDENCE = .05
TARGETS = ((808, 589), (902, 591))
MATCH_RADIUS = 60

os.environ.update(
    SCOUT_CLASS_NAMES='["ball"]', SCOUT_STEREO_CALIBRATION=CALIBRATION,
    SCOUT_STEREO_RECTIFY='1', SCOUT_STEREO_BASELINE_M='.0635',
    SCOUT_STEREO_DISPARITY_SCALE='1', SCOUT_STEREO_MATCH_SCALE='.5',
    SCOUT_STEREO_NUM_DISPARITIES='128', SCOUT_STEREO_VERTICAL_ROI_PAD='24',
    SCOUT_STEREO_ADAPTIVE_DISPARITY_WIDTH='0', SCOUT_FUEL_DIAMETER_M='.15',
    SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR='.55',
    SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU='.90')
from scout.detectors import Detection
from scout.stereo import StereoDepth


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(value - center) for value in values]) if values else None


def summary(values):
    if not values:
        return {'n': 0, 'median_m': None, 'mad_m': None, 'p95_m': None}
    ordered = sorted(values)
    return {'n': len(ordered), 'median_m': round(median(ordered), 4),
            'mad_m': round(mad(ordered), 4),
            'p95_m': round(ordered[int(.95 * (len(ordered) - 1))], 4)}


def boxes_from(items):
    return [Detection(int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2']),
                      0, float(b['confidence']))
            for b in items if b['confidence'] >= CONFIDENCE]


def target_boxes(boxes):
    targets = [None, None]
    extras = 0
    for box in sorted(boxes, key=lambda item: item.confidence, reverse=True):
        cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        distance, index = min((math.hypot(cx - x, cy - y), i)
                              for i, (x, y) in enumerate(TARGETS))
        if distance <= MATCH_RADIUS and targets[index] is None:
            targets[index] = box
        else:
            extras += 1
    return targets, extras


output = {
    'method': ('Reuse the saved 640-first / 1280-fallback detector boxes, replay '
               'both synchronized videos, and compare the actual runtime StereoDepth '
               'with full-width versus disparity-aware horizontal cropping. Both use '
               'the same 128-disparity, 24 px vertical band and 55% FUEL size gate.'),
    'settings': {'confidence': CONFIDENCE, 'baseline_m': .0635,
                 'disparities': 128, 'vertical_pad_px': 24,
                 'size_gate_relative_tolerance': .55},
    'limitations': ('Two stationary targets at one pose; no absolute range truth. '
                    'Detector latency comes from the prior cascade replay. Combined '
                    'rate estimate excludes decode, queue, UI, and NetworkTables costs.'),
    'results': [],
}

for clip in CASCADE['results']:
    estimators = {}
    records = {}
    for enabled in (False, True):
        os.environ['SCOUT_STEREO_HORIZONTAL_ROI'] = '1' if enabled else '0'
        name = 'cascade_horizontal_crop' if enabled else 'cascade_full_width'
        estimators[name] = StereoDepth()
        records[name] = {'ranges': [[], []], 'counts': {0: 0, 1: 0, 2: 0},
                         'valid_pairs': 0, 'extras': 0, 'gate_rejected': 0,
                         'duplicate_rejected': 0, 'stereo_ms': []}
    deltas = [[], []]
    capture = cv2.VideoCapture(str(ROOT / clip['video']))
    frames = 0
    for row in clip['frames_detail']:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        left, right = frame[:, :1600], frame[:, 1600:]
        outputs = {}
        for name, estimator in estimators.items():
            detections = boxes_from(row['selected_boxes'])
            started = time.perf_counter()
            estimator.estimate(left, right, detections)
            record = records[name]
            record['stereo_ms'].append((time.perf_counter() - started) * 1000)
            filtered = estimator.filter_fuel_size_inconsistent(detections)
            record['gate_rejected'] += estimator.last_size_gate_rejected
            record['duplicate_rejected'] += estimator.last_size_gate_duplicate_rejected
            targets, extras = target_boxes(filtered)
            record['extras'] += extras
            count = sum(box is not None and box.distance_m is not None for box in targets)
            record['counts'][count] += 1
            record['valid_pairs'] += count == 2
            outputs[name] = targets
            for index, box in enumerate(targets):
                if box is not None and box.distance_m is not None:
                    record['ranges'][index].append(float(box.distance_m))
        full = outputs['cascade_full_width']
        cropped = outputs['cascade_horizontal_crop']
        for index in range(2):
            if (full[index] is not None and cropped[index] is not None and
                    full[index].distance_m is not None and cropped[index].distance_m is not None):
                deltas[index].append(float(cropped[index].distance_m - full[index].distance_m))
    capture.release()
    configs = []
    for name, record in records.items():
        detector_ms = clip['cascade_detector_ms']['median_ms']
        stereo_ms = median(record['stereo_ms'])
        configs.append({
            'method': name, 'frames': frames,
            'target_count_histogram_after_size_gate': {str(k): v for k, v in record['counts'].items()},
            'valid_pair_frames': record['valid_pairs'], 'off_target_boxes_after_gate': record['extras'],
            'size_gate_rejected': record['gate_rejected'],
            'duplicate_rejected': record['duplicate_rejected'],
            'range_by_target_m': [summary(items) for items in record['ranges']],
            'stereo_ms_median': round(stereo_ms, 2) if stereo_ms is not None else None,
            'stereo_ms_p95': round(sorted(record['stereo_ms'])[int(.95 * (len(record['stereo_ms']) - 1))], 2),
            'cascade_detector_ms_median': round(detector_ms, 2),
            'detector_plus_stereo_median_ms': round(detector_ms + stereo_ms, 2),
            'median_component_cost_hz_ceiling': round(1000 / (detector_ms + stereo_ms), 2),
        })
    output['results'].append({
        'scene': clip['scene'], 'video': clip['video'], 'frames': frames,
        'same_frame_range_delta_cropped_minus_full_m': [summary(items) for items in deltas],
        'configurations': configs,
    })

destination = ROOT / 'media/frc-fuel-cascade-horizontal-roi-evaluation.json'
destination.write_text(json.dumps(output, indent=2) + '\n')
for result in output['results']:
    print(result['scene'], 'frames', result['frames'],
          'range delta crop-full', result['same_frame_range_delta_cropped_minus_full_m'])
    for config in result['configurations']:
        print(config['method'], 'pairs', config['valid_pair_frames'],
              'counts', config['target_count_histogram_after_size_gate'],
              'ranges', config['range_by_target_m'], 'stereo ms',
              config['stereo_ms_median'], config['stereo_ms_p95'],
              'median component-cost ceiling Hz', config['median_component_cost_hz_ceiling'])
print('saved', destination)
