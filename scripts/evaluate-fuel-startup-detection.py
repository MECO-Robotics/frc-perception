#!/usr/bin/env python3
"""Compare early-video FUEL detections and image brightness across saved clips."""
import json
import math
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
TARGETS = ((808, 589), (902, 591))
RADIUS_PX = 60
THRESHOLDS = (0.05, 0.075, 0.10)
BINS = (('0-0.25s', 0, .25), ('0.25-0.5s', .25, .5),
        ('0.5-1s', .5, 1), ('1-2s', 1, 2), ('2-10.1s', 2, 10.1))


def score_frame(detections, threshold):
    targets = [None, None]
    extras = 0
    for box in sorted((item for item in detections
                       if item['confidence'] >= threshold),
                      key=lambda item: item['confidence'], reverse=True):
        cx = (box['x1'] + box['x2']) / 2
        cy = (box['y1'] + box['y2']) / 2
        distance, target_index = min(
            (math.hypot(cx - x, cy - y), index)
            for index, (x, y) in enumerate(TARGETS))
        if distance <= RADIUS_PX and targets[target_index] is None:
            targets[target_index] = box
        else:
            extras += 1
    return targets, extras


output = {
    'method': ('Decode both saved stereo videos. For each frame, match detector '
               'boxes to two fixed labeled image-center windows within 60 px; '
               'rescore at confidence 0.05, 0.075, and 0.10. Also measure mean '
               'left-image luma. Bins are relative to the start of each recording.'),
    'target_centers_px': TARGETS,
    'match_radius_px': RADIUS_PX,
    'confidence_thresholds': THRESHOLDS,
    'time_bins_s': BINS,
    'limitations': ('The fixed target windows assume the same two balls are '
                    'visible at the same image positions throughout each clip. '
                    'They do not establish object visibility or absolute range. '
                    'The two clips may start at different stages of ball '
                    'placement and automatic exposure.'),
    'results': [],
}

for clip in DATA['results']:
    capture = cv2.VideoCapture(str(ROOT / clip['video']))
    records = {threshold: {label: {'frames': 0, 'paired': 0, 'target_hits': 0,
                                   'missed_targets': 0, 'off_target_boxes': 0,
                                   'left_luma': []}
                           for label, _, _ in BINS}
               for threshold in THRESHOLDS}
    frame_count = 0
    for row in clip['per_frame']:
        ok, frame = capture.read()
        if not ok:
            break
        frame_count += 1
        time_s = row['time_s']
        bin_item = next(((label, lo, hi) for label, lo, hi in BINS
                         if lo <= time_s < hi), None)
        if bin_item is None:
            continue
        label, _, _ = bin_item
        luma = float(cv2.cvtColor(frame[:, :1600], cv2.COLOR_BGR2GRAY).mean())
        for threshold in THRESHOLDS:
            record = records[threshold][label]
            targets, extras = score_frame(row['detections'], threshold)
            hits = sum(target is not None for target in targets)
            record['frames'] += 1
            record['paired'] += int(hits == 2)
            record['target_hits'] += hits
            record['missed_targets'] += 2 - hits
            record['off_target_boxes'] += extras
            record['left_luma'].append(luma)
    capture.release()

    bins = []
    for label, lo, hi in BINS:
        per_threshold = {}
        for threshold in THRESHOLDS:
            record = records[threshold][label]
            frames = record['frames']
            per_threshold[str(threshold)] = {
                'paired_frames': record['paired'],
                'pair_fraction': round(record['paired'] / frames, 4) if frames else None,
                'target_recall': round(record['target_hits'] / (2 * frames), 4) if frames else None,
                'missed_target_instances': record['missed_targets'],
                'off_target_boxes': record['off_target_boxes'],
                'mean_left_luma': round(sum(record['left_luma']) / frames, 2) if frames else None,
            }
        bins.append({'interval_s': [lo, hi], 'frame_count': records[THRESHOLDS[0]][label]['frames'],
                     'thresholds': per_threshold})
    output['results'].append({'scene': clip['scene'], 'video': clip['video'],
                              'decoded_frames': frame_count, 'bins': bins})

path = ROOT / 'media/frc-fuel-startup-detection-evaluation.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output, indent=2))
print('saved', path)
