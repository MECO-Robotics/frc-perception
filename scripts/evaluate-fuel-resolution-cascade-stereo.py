#!/usr/bin/env python3
"""Compare fixed-1280 and cascade boxes through the active stereo estimator."""
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
BASELINE = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
CALIBRATION = 'media/calibration/stereo_calibration_live_order_candidate.yml'
CONFIDENCE = .05
TARGETS = ((808, 589), (902, 591))
MATCH_RADIUS_PX = 60

os.environ.update(
    SCOUT_CLASS_NAMES='["ball"]', SCOUT_STEREO_CALIBRATION=CALIBRATION,
    SCOUT_STEREO_RECTIFY='1', SCOUT_STEREO_BASELINE_M='.0635',
    SCOUT_STEREO_DISPARITY_SCALE='1', SCOUT_STEREO_MATCH_SCALE='.5',
    SCOUT_STEREO_NUM_DISPARITIES='128', SCOUT_STEREO_VERTICAL_ROI_PAD='24',
    SCOUT_STEREO_HORIZONTAL_ROI='0', SCOUT_STEREO_ADAPTIVE_DISPARITY_WIDTH='0',
    SCOUT_FUEL_DIAMETER_M='.15', SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR='.55',
    SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU='.90')
from scout.detectors import Detection
from scout.stereo import StereoDepth


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(value - center) for value in values]) if values else None


def summarize(values):
    if not values:
        return {'n': 0, 'median_m': None, 'mad_m': None, 'p05_m': None, 'p95_m': None}
    ordered = sorted(values)
    return {'n': len(ordered), 'median_m': round(median(ordered), 4),
            'mad_m': round(mad(ordered), 4),
            'p05_m': round(ordered[int(.05 * (len(ordered) - 1))], 4),
            'p95_m': round(ordered[int(.95 * (len(ordered) - 1))], 4)}


def to_boxes(items):
    return [Detection(int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2']),
                      0, float(box['confidence']))
            for box in items if box['confidence'] >= CONFIDENCE]


def match(boxes):
    targets = [None, None]
    extras = 0
    for box in sorted(boxes, key=lambda item: item.confidence, reverse=True):
        cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        distance, index = min((math.hypot(cx - x, cy - y), i)
                              for i, (x, y) in enumerate(TARGETS))
        if distance <= MATCH_RADIUS_PX and targets[index] is None:
            targets[index] = box
        else:
            extras += 1
    return targets, extras


output = {
    'method': ('Decode both videos frame by frame. Use the actual 640-first '
               'cascade outputs and saved 1280x960 detections, run each box set '
               'through a separate runtime StereoDepth instance with identical '
               'rectified 128-disparity / 24 px vertical-band settings, apply '
               'the active 55% size gate, and compare target ranges on the same '
               'frames.'),
    'settings': {'calibration': CALIBRATION, 'baseline_m': .0635,
                 'disparity_scale': 1.0, 'match_scale': .5,
                 'num_disparities': 128, 'vertical_pad': 24,
                 'horizontal_roi': False, 'size_gate_relative_tolerance': .55,
                 'fuel_diameter_m': .15, 'cascade_fallback_min_confidence': .06},
    'limitations': ('Both clips contain stationary targets at a single pose and '
                    'range. The baseline and cascade use different detector '
                    'boxes, so same-frame range deltas measure configuration '
                    'sensitivity rather than absolute accuracy. No tape-measured '
                    'range is available.'),
    'results': [],
}

for baseline_clip, cascade_clip in zip(BASELINE['results'], CASCADE['results']):
    if baseline_clip['scene'] != cascade_clip['scene']:
        raise ValueError('Saved detector outputs are not aligned by scene')
    baseline_rows = baseline_clip['per_frame']
    cascade_rows = cascade_clip['frames_detail']
    if len(baseline_rows) != len(cascade_rows):
        raise ValueError('Saved detector outputs have different frame counts')
    estimators = {'fixed_1280': StereoDepth(), 'cascade_640_then_1280': StereoDepth()}
    records = {name: {'ranges': [[], []], 'target_count_hist': {0: 0, 1: 0, 2: 0},
                      'valid_pair_frames': 0, 'extra_boxes': 0,
                      'gate_rejected': 0, 'duplicate_rejected': 0,
                      'estimate_ms': []}
               for name in estimators}
    deltas = [[], []]
    capture = cv2.VideoCapture(str(ROOT / baseline_clip['video']))
    frames = 0
    for base_row, cascade_row in zip(baseline_rows, cascade_rows):
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        box_sets = {
            'fixed_1280': to_boxes(base_row['detections']),
            'cascade_640_then_1280': to_boxes(cascade_row['selected_boxes']),
        }
        frame_outputs = {}
        for name, estimator in estimators.items():
            boxes = box_sets[name]
            started = time.perf_counter()
            estimator.estimate(frame[:, :1600], frame[:, 1600:], boxes)
            record = records[name]
            record['estimate_ms'].append((time.perf_counter() - started) * 1000)
            filtered = estimator.filter_fuel_size_inconsistent(boxes)
            record['gate_rejected'] += estimator.last_size_gate_rejected
            record['duplicate_rejected'] += estimator.last_size_gate_duplicate_rejected
            targets, extras = match(filtered)
            record['extra_boxes'] += extras
            target_count = sum(box is not None and box.distance_m is not None for box in targets)
            record['target_count_hist'][target_count] += 1
            if target_count == 2:
                record['valid_pair_frames'] += 1
            frame_outputs[name] = targets
            for index, box in enumerate(targets):
                if box is not None and box.distance_m is not None:
                    record['ranges'][index].append(float(box.distance_m))
        for index in range(2):
            first = frame_outputs['fixed_1280'][index]
            second = frame_outputs['cascade_640_then_1280'][index]
            if (first is not None and second is not None and
                    first.distance_m is not None and second.distance_m is not None):
                deltas[index].append(float(second.distance_m - first.distance_m))
    capture.release()

    configurations = []
    for name, record in records.items():
        latencies = record['estimate_ms']
        configurations.append({
            'method': name, 'frames': frames,
            'target_count_histogram_after_gate': {
                str(key): value for key, value in record['target_count_hist'].items()},
            'valid_pair_frames': record['valid_pair_frames'],
            'off_target_boxes_after_gate': record['extra_boxes'],
            'size_gate_rejected': record['gate_rejected'],
            'duplicate_rejected': record['duplicate_rejected'],
            'range_by_target_m': [summarize(values) for values in record['ranges']],
            'stereo_estimate_ms_median': round(median(latencies), 2) if latencies else None,
            'stereo_estimate_ms_p95': round(sorted(latencies)[int(.95 * (len(latencies) - 1))], 2)
            if latencies else None,
        })
    output['results'].append({
        'scene': baseline_clip['scene'], 'video': baseline_clip['video'],
        'frames': frames,
        'same_frame_range_delta_cascade_minus_fixed_m': [summarize(values)
                                                         for values in deltas],
        'configurations': configurations,
    })

path = ROOT / 'media/frc-fuel-resolution-cascade-stereo.json'
path.write_text(json.dumps(output, indent=2) + '\n')
for clip in output['results']:
    print(clip['scene'], 'delta', clip['same_frame_range_delta_cascade_minus_fixed_m'])
    for config in clip['configurations']:
        print(config['method'], 'pairs', config['valid_pair_frames'],
              'counts', config['target_count_histogram_after_gate'],
              'ranges', config['range_by_target_m'],
              'gate/dups', config['size_gate_rejected'], config['duplicate_rejected'],
              'stereo ms', config['stereo_estimate_ms_median'], config['stereo_estimate_ms_p95'])
print('saved', path)
