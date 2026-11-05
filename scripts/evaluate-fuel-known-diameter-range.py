#!/usr/bin/env python3
"""Evaluate Hough-circle known-diameter range on every frame of both clips."""
import json
import math
import statistics
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
STEREO = json.loads((ROOT / 'media/frc-fuel-dense-stereo-evaluation.json').read_text())
CALIBRATION = ROOT / 'media/calibration/stereo_calibration_live_order_candidate.yml'
TARGETS = ((808.0, 589.0), (902.0, 591.0))
MATCH_RADIUS_PX = 60
CONFIDENCE = .05
FUEL_DIAMETER_M = .15
PARAM2_VALUES = (20, 25, 30)

cv2.setNumThreads(1)
storage = cv2.FileStorage(str(CALIBRATION), cv2.FILE_STORAGE_READ)
focal_px = float(storage.getNode('left_camera_matrix').mat()[0, 0])
storage.release()


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(value - center) for value in values]) if values else None


def summary(values):
    if not values:
        return {'n': 0, 'median_m': None, 'mad_m': None,
                'p05_m': None, 'p95_m': None}
    ordered = sorted(values)
    return {'n': len(ordered), 'median_m': round(median(ordered), 4),
            'mad_m': round(mad(ordered), 4),
            'p05_m': round(ordered[int(.05 * (len(ordered) - 1))], 4),
            'p95_m': round(ordered[int(.95 * (len(ordered) - 1))], 4)}


def match(detections):
    targets = [None, None]
    for box in sorted((item for item in detections if item['confidence'] >= CONFIDENCE),
                      key=lambda item: item['confidence'], reverse=True):
        center = ((box['x1'] + box['x2']) / 2,
                  (box['y1'] + box['y2']) / 2)
        distance, index = min((math.dist(center, point), i)
                              for i, point in enumerate(TARGETS))
        if distance <= MATCH_RADIUS_PX and targets[index] is None:
            targets[index] = box
    return targets


output = {
    'method': ('For every decoded frame in both saved videos, match the saved '
               'FRC YOLO26n boxes to the two fixed target windows, then run '
               'Hough circles in a detector-centered crop. Convert circle '
               'diameter to metric range using nominal 0.15 m FUEL diameter and '
               'raw-camera focal length. Compare to saved same-frame SGBM range.'),
    'settings': {'confidence': CONFIDENCE, 'target_centers_px': TARGETS,
                 'match_radius_px': MATCH_RADIUS_PX,
                 'crop_half_extent_multiplier': .8,
                 'radius_box_short_side_multipliers': [.35, .65],
                 'circle_center_offset_short_side_multiplier': .30,
                 'hough_param2_values': PARAM2_VALUES,
                 'fuel_diameter_m': FUEL_DIAMETER_M,
                 'raw_focal_px': focal_px},
    'limitations': ('Nominal ball diameter and calibrated focal length define a '
                    'monocular geometry proxy, not measured range truth. Hough '
                    'fit quality can be affected by clipping, highlight, '
                    'background texture, and occlusion. Both clips contain '
                    'stationary targets at one pose.'),
    'results': [],
}

for detections_clip, stereo_clip in zip(DETECTIONS['results'], STEREO['results']):
    if detections_clip['scene'] != stereo_clip['scene']:
        raise ValueError('Detector and stereo datasets are not aligned by scene')
    capture = cv2.VideoCapture(str(ROOT / detections_clip['video']))
    records = {param2: [{'ranges': [], 'radii': [], 'offsets': [],
                         'range_delta_to_stereo': [], 'abs_range_delta_to_stereo': [],
                         'stereo_ranges': []} for _ in range(2)]
               for param2 in PARAM2_VALUES}
    frame_rows = []
    frames = 0
    for index, detection_row in enumerate(detections_clip['per_frame']):
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        boxes = match(detection_row['detections'])
        left = frame[:, :1600]
        gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        frame_stereo = stereo_clip['frames'][index]['ranges_m']
        item = {'frame': index, 'targets': []}
        for target_index, box in enumerate(boxes):
            target_output = {'detected': box is not None,
                             'stereo_range_m': frame_stereo[target_index],
                             'hough': {}}
            if box is not None:
                bx = (box['x1'] + box['x2']) / 2
                by = (box['y1'] + box['y2']) / 2
                short = min(box['x2'] - box['x1'], box['y2'] - box['y1'])
                x0 = max(0, int(round(bx - short * .8)))
                y0 = max(0, int(round(by - short * .8)))
                x1 = min(1600, int(round(bx + short * .8)))
                y1 = min(1200, int(round(by + short * .8)))
                crop = gray[y0:y1, x0:x1]
                blurred = cv2.GaussianBlur(crop, (9, 9), 2) if crop.size else crop
                radius_min = max(8, int(short * .35))
                radius_max = max(9, int(short * .65))
                for param2 in PARAM2_VALUES:
                    circles = cv2.HoughCircles(
                        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                        minDist=max(10, int(short * .5)), param1=90,
                        param2=param2, minRadius=radius_min,
                        maxRadius=radius_max) if crop.size else None
                    candidates = []
                    if circles is not None:
                        for x, y, radius in circles[0]:
                            center = (float(x + x0), float(y + y0))
                            offset = math.dist(center, (bx, by))
                            if offset <= short * .30:
                                candidates.append((offset, float(radius)))
                    circle_result = None
                    if candidates:
                        offset, radius = min(candidates, key=lambda value: value[0])
                        range_m = focal_px * FUEL_DIAMETER_M / (2 * radius)
                        circle_result = {'radius_px': round(radius, 4),
                                         'center_offset_px': round(offset, 4),
                                         'range_m': round(range_m, 5)}
                        rec = records[param2][target_index]
                        rec['ranges'].append(range_m)
                        rec['radii'].append(radius)
                        rec['offsets'].append(offset)
                        if frame_stereo[target_index] is not None:
                            delta = range_m - frame_stereo[target_index]
                            rec['range_delta_to_stereo'].append(delta)
                            rec['abs_range_delta_to_stereo'].append(abs(delta))
                            rec['stereo_ranges'].append(frame_stereo[target_index])
                    target_output['hough'][str(param2)] = circle_result
            else:
                for param2 in PARAM2_VALUES:
                    target_output['hough'][str(param2)] = None
            item['targets'].append(target_output)
        frame_rows.append(item)
    capture.release()

    param_results = []
    for param2 in PARAM2_VALUES:
        target_results = []
        for target_index, rec in enumerate(records[param2]):
            deltas = rec['range_delta_to_stereo']
            target_results.append({
                'target': target_index + 1,
                'circle_coverage': len(rec['ranges']),
                'circle_range_m': summary(rec['ranges']),
                'radius_px': {'n': len(rec['radii']),
                              'median': round(median(rec['radii']), 3) if rec['radii'] else None,
                              'mad': round(mad(rec['radii']), 3) if rec['radii'] else None},
                'center_offset_px_median': round(median(rec['offsets']), 3)
                if rec['offsets'] else None,
                'same_frame_stereo_range_delta_m': summary(deltas),
                'same_frame_abs_delta_to_stereo_median': round(
                    median(rec['abs_range_delta_to_stereo']), 4)
                if rec['abs_range_delta_to_stereo'] else None,
                'stereo_range_m': summary(rec['stereo_ranges']),
            })
        pair_hough = []
        pair_stereo = []
        pair_delta = []
        for item in frame_rows:
            hough = [target['hough'][str(param2)] for target in item['targets']]
            stereo = [target['stereo_range_m'] for target in item['targets']]
            if all(hough) and all(value is not None for value in stereo):
                pair_hough.append(abs(hough[0]['range_m'] - hough[1]['range_m']))
                pair_stereo.append(abs(stereo[0] - stereo[1]))
                pair_delta.append(pair_hough[-1] - pair_stereo[-1])
        param_results.append({
            'param2': param2, 'targets': target_results,
            'both_circle_and_stereo_pair_frames': len(pair_hough),
            'absolute_inter_target_range_gap_hough_m': summary(pair_hough),
            'absolute_inter_target_range_gap_stereo_m': summary(pair_stereo),
            'same_frame_gap_difference_hough_minus_stereo_m': summary(pair_delta),
        })
    output['results'].append({'scene': detections_clip['scene'],
                              'video': detections_clip['video'],
                              'frames': frames, 'param2_sweep': param_results,
                              'frame_results': frame_rows})

path = ROOT / 'media/frc-fuel-known-diameter-range-evaluation.json'
path.write_text(json.dumps(output, indent=2) + '\n')
for clip in output['results']:
    print(clip['scene'])
    for param in clip['param2_sweep']:
        print('param2', param['param2'], 'pair frames',
              param['both_circle_and_stereo_pair_frames'],
              'circle ranges', [(t['circle_coverage'], t['circle_range_m'],
                                 t['same_frame_abs_delta_to_stereo_median'])
                                for t in param['targets']],
              'Hough gap', param['absolute_inter_target_range_gap_hough_m'],
              'stereo gap', param['absolute_inter_target_range_gap_stereo_m'])
print('saved', path)
