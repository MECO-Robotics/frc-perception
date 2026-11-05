#!/usr/bin/env python3
"""Dense per-frame FUEL count/confidence evaluation on the two saved clips."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import cv2
from scout.detectors import TensorRTDetector

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--engine', default='models/rebuilt_fuel_yolo26n_1280x960_fp16.engine')
parser.add_argument('--output', default='media/frc-fuel-dense-count-evaluation.json')
args = parser.parse_args()
ENGINE = ROOT / args.engine
VIDEOS = (
    ('earlier', ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr.mp4'),
    ('corrected-settings', ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'),
)
TARGETS = ((808, 589), (902, 591))
MATCH_RADIUS_PX = 60.0
CONFIDENCES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
               0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


def summarize(rows, threshold):
    tp = fp = fn = paired = exact_two = 0
    target_scores = [[], []]
    for row in rows:
        detections = [item for item in row['detections'] if item['confidence'] >= threshold]
        hits = [None, None]
        extras = 0
        for item in sorted(detections, key=lambda d: d['confidence'], reverse=True):
            cx = (item['x1'] + item['x2']) / 2
            cy = (item['y1'] + item['y2']) / 2
            distances = [((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 for x, y in TARGETS]
            distance, target_i = min((value, i) for i, value in enumerate(distances))
            if distance <= MATCH_RADIUS_PX and hits[target_i] is None:
                hits[target_i] = item
            else:
                extras += 1
        count = sum(hit is not None for hit in hits)
        tp += count
        fn += 2 - count
        fp += extras
        paired += count == 2
        exact_two += count == 2 and extras == 0
        for i, hit in enumerate(hits):
            if hit is not None:
                target_scores[i].append(hit['confidence'])
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {
        'confidence': threshold,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'paired_frames': paired,
        'exact_two_frames': exact_two,
        'precision': round(precision, 5),
        'recall': round(recall, 5),
        'f1': round(2 * precision * recall / (precision + recall), 5)
        if precision + recall else 0,
        'target_confidence_min_median': [
            {'minimum': round(min(values), 4),
             'median': round(statistics.median(values), 4)} if values else None
            for values in target_scores
        ],
    }


os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '2.0'
detector = TensorRTDetector(str(ENGINE), confidence=0.05)
result = {
    'engine': str(ENGINE.relative_to(ROOT)),
    'method': 'Run the 1280x960 FP16 TensorRT detector on every decoded left-view frame in each saved synchronized stereo video. Match detections to the two manually labeled stationary target centers within 60 px; sweep confidence after inference.',
    'limitations': 'The target centers are manually labeled and fixed for these clips. This does not test moving balls, varied distances, unrecorded occlusions, or metric range accuracy.',
    'match_radius_px': MATCH_RADIUS_PX,
    'results': [],
}
try:
    for scene, video_path in VIDEOS:
        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        rows = []
        latencies = []
        started = time.perf_counter()
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            boxes, latency = detector.infer(frame[:, :1600])
            latencies.append(latency)
            rows.append({
                'frame': index,
                'time_s': round(index / fps, 4) if fps else None,
                'detections': [
                    {'x1': box.x1, 'y1': box.y1, 'x2': box.x2, 'y2': box.y2,
                     'confidence': box.confidence}
                    for box in boxes
                ],
            })
            index += 1
        capture.release()
        result['results'].append({
            'scene': scene,
            'video': str(video_path.relative_to(ROOT)),
            'fps': fps,
            'reported_frames': total,
            'evaluated_frames': len(rows),
            'decode_and_inference_seconds': round(time.perf_counter() - started, 2),
            'single_view_inference_ms_median': round(statistics.median(latencies), 2)
            if latencies else None,
            'threshold_sweep': [summarize(rows, threshold) for threshold in CONFIDENCES],
            'per_frame': rows,
        })
finally:
    detector.close()

output_path = ROOT / args.output
output_path.write_text(json.dumps(result, indent=2) + '\n')
for scene_result in result['results']:
    print(scene_result['scene'], 'frames', scene_result['evaluated_frames'],
          'fps', scene_result['fps'], 'median_ms', scene_result['single_view_inference_ms_median'])
    for summary in scene_result['threshold_sweep']:
        print(summary['confidence'], 'exact', summary['exact_two_frames'],
              '/', scene_result['evaluated_frames'], 'pair', summary['paired_frames'],
              'fp/fn', summary['fp'], summary['fn'], 'P/R/F1', summary['precision'],
              summary['recall'], summary['f1'])
print('saved', output_path)
