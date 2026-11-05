#!/usr/bin/env python3
"""Stress-test Hough known-diameter range under synthetic partial occlusion."""
import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
STEREO = json.loads((ROOT / 'media/frc-fuel-dense-stereo-evaluation.json').read_text())
CAL = ROOT / 'media/calibration/stereo_calibration_live_order_candidate.yml'
TARGETS = ((808.0, 589.0), (902.0, 591.0))
TIMES = tuple(1.0 + .4 * i for i in range(15))
PARAMS = (20, 25, 30)
MASKS = (('none', 0), ('bottom', .25), ('bottom', .50), ('bottom', .75),
         ('left', .25), ('left', .50), ('left', .75))
CONFIDENCE = .05
FUEL_DIAMETER_M = .15

cv2.setNumThreads(1)
storage = cv2.FileStorage(str(CAL), cv2.FILE_STORAGE_READ)
focal = float(storage.getNode('left_camera_matrix').mat()[0, 0])
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
    for box in sorted((item for item in detections
                       if item['confidence'] >= CONFIDENCE),
                      key=lambda item: item['confidence'], reverse=True):
        center = ((box['x1'] + box['x2']) / 2,
                  (box['y1'] + box['y2']) / 2)
        distance, index = min((math.dist(center, target), i)
                              for i, target in enumerate(TARGETS))
        if distance <= 60 and targets[index] is None:
            targets[index] = box
    return targets


def fit_circle(gray, box, param2):
    x1, y1, x2, y2 = map(int, (box['x1'], box['y1'], box['x2'], box['y2']))
    bx, by = (x1 + x2) / 2, (y1 + y2) / 2
    short = min(x2 - x1, y2 - y1)
    x0, y0 = max(0, int(round(bx - short * .8))), max(0, int(round(by - short * .8)))
    x1c, y1c = min(1600, int(round(bx + short * .8))), min(1200, int(round(by + short * .8)))
    roi = gray[y0:y1c, x0:x1c]
    if not roi.size:
        return None
    blurred = cv2.GaussianBlur(roi, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(10, int(short * .5)), param1=90, param2=param2,
        minRadius=max(8, int(short * .35)), maxRadius=max(9, int(short * .65)))
    if circles is None:
        return None
    candidates = []
    for x, y, radius in circles[0]:
        offset = math.hypot(x + x0 - bx, y + y0 - by)
        if offset <= short * .30:
            candidates.append((offset, float(radius)))
    if not candidates:
        return None
    offset, radius = min(candidates, key=lambda item: item[0])
    return {'radius_px': radius, 'center_offset_px': float(offset),
            'range_m': focal * FUEL_DIAMETER_M / (2 * radius)}


output = {
    'method': ('At 15 timestamps per clip, hold each detector box fixed, black '
               'out the bottom or left side of the ball by 25/50/75%, run '
               'detector-centered Hough range at param2 20/25/30, and compare '
               'coverage/range shift with the unmasked Hough result and same-frame SGBM.'),
    'sample_times_s': TIMES,
    'mask_configs': [{'orientation': name, 'fraction': fraction}
                     for name, fraction in MASKS],
    'param2_values': PARAMS,
    'limitations': ('Black masks on fixed clean boxes do not simulate a real '
                    'foreground occluder, parallax, or detector box changes. '
                    'The known-diameter range is not measured truth.'),
    'settings': {'focal_px': focal, 'fuel_diameter_m': FUEL_DIAMETER_M,
                 'target_centers_px': TARGETS},
    'results': [],
}

for clip_index, clip in enumerate(DATA['results']):
    stereo_clip = STEREO['results'][clip_index]
    cap = cv2.VideoCapture(str(ROOT / clip['video']))
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows = []
    for time_s in TIMES:
        frame_index = round(time_s * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame[:, :1600], cv2.COLOR_BGR2GRAY)
        boxes = match(clip['per_frame'][frame_index]['detections'])
        stereo_ranges = stereo_clip['frames'][frame_index]['ranges_m']
        frame_result = {'frame': frame_index, 'time_s': time_s, 'targets': []}
        for target_index, box in enumerate(boxes):
            target_row = {'box': box, 'stereo_range_m': stereo_ranges[target_index],
                          'configs': {}}
            for orientation, fraction in MASKS:
                masked = gray.copy()
                if box is not None and fraction:
                    x1, y1, x2, y2 = map(int, (box['x1'], box['y1'], box['x2'], box['y2']))
                    if orientation == 'bottom':
                        y_start = max(0, y2 - max(1, round((y2 - y1) * fraction)))
                        masked[y_start:y2, x1:x2] = 0
                    elif orientation == 'left':
                        x_end = max(0, x1 + max(1, round((x2 - x1) * fraction)))
                        masked[y1:y2, x1:x_end] = 0
                for param2 in PARAMS:
                    result = fit_circle(masked, box, param2) if box is not None else None
                    target_row['configs'][f'{orientation}_{fraction:.2f}_p{param2}'] = result
            frame_result['targets'].append(target_row)
        rows.append(frame_result)
    cap.release()

    configs = []
    for orientation, fraction in MASKS:
        for param2 in PARAMS:
            target_results = []
            for target_index in range(2):
                ranges, radii, offsets, stereo_deltas, mask_deltas = [], [], [], [], []
                for row in rows:
                    target = row['targets'][target_index]
                    key = f'{orientation}_{fraction:.2f}_p{param2}'
                    circle = target['configs'][key]
                    if circle:
                        ranges.append(circle['range_m'])
                        radii.append(circle['radius_px'])
                        offsets.append(circle['center_offset_px'])
                        if target['stereo_range_m'] is not None:
                            stereo_deltas.append(circle['range_m'] - target['stereo_range_m'])
                        baseline = target['configs'][f'none_0.00_p{param2}']
                        if baseline:
                            mask_deltas.append(circle['range_m'] - baseline['range_m'])
                target_results.append({
                    'target': target_index + 1,
                    'coverage': len(ranges),
                    'coverage_fraction': round(len(ranges) / len(rows), 4) if rows else None,
                    'range_m': summary(ranges),
                    'radius_px': {'median': round(median(radii), 3) if radii else None,
                                  'mad': round(mad(radii), 3) if radii else None},
                    'center_offset_px_median': round(median(offsets), 3) if offsets else None,
                    'same_frame_delta_to_unmasked_m': summary(mask_deltas),
                    'same_frame_delta_to_stereo_m': summary(stereo_deltas),
                })
            configs.append({'orientation': orientation, 'fraction': fraction,
                            'param2': param2, 'targets': target_results})
    output['results'].append({'scene': clip['scene'], 'video': clip['video'],
                              'frames': len(rows), 'configs': configs})

path = ROOT / 'media/frc-fuel-hough-range-occlusion-evaluation.json'
path.write_text(json.dumps(output, indent=2) + '\n')
for clip in output['results']:
    print(clip['scene'])
    for cfg in clip['configs']:
        if cfg['param2'] == 20:
            print(cfg['orientation'], cfg['fraction'], 'coverage/range',
                  [(target['coverage'], target['range_m'], target['same_frame_delta_to_unmasked_m'])
                   for target in cfg['targets']])
print('saved', path)
