#!/usr/bin/env python3
"""Evaluate the FRC TensorRT detector on Project516's labeled public images."""
import concurrent.futures
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '2.0'
from scout.detectors import TensorRTDetector

REPO = 'https://huggingface.co/datasets/project516/rebuilt-fuel-dataset/resolve/main'
ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
THRESHOLDS = (0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
IOU_THRESHOLD = 0.5


def get_bytes(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Scout-FRC-evaluation/1.0'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def labels_for(image_id, width, height):
    text = get_bytes(f'{REPO}/{image_id}.txt').decode('utf-8').strip()
    boxes = []
    for line in text.splitlines():
        values = [float(value) for value in line.split()]
        if len(values) != 5:
            raise ValueError(f'Unexpected label row for {image_id}: {line}')
        class_id, cx, cy, bw, bh = values
        boxes.append({'class_id': int(class_id),
                      'x1': (cx - bw / 2) * width, 'y1': (cy - bh / 2) * height,
                      'x2': (cx + bw / 2) * width, 'y2': (cy + bh / 2) * height})
    return boxes


def iou(a, b):
    x1, y1 = max(a['x1'], b['x1']), max(a['y1'], b['y1'])
    x2, y2 = min(a['x2'], b['x2']), min(a['y2'], b['y2'])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a['x2'] - a['x1']) * max(0, a['y2'] - a['y1'])
    area_b = max(0, b['x2'] - b['x1']) * max(0, b['y2'] - b['y1'])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match(predictions, ground_truth, threshold):
    candidates = sorted(((iou(pred, gt), p, g)
                         for p, pred in enumerate(predictions)
                         for g, gt in enumerate(ground_truth)), reverse=True)
    used_predictions, used_ground_truth = set(), set()
    for overlap, prediction, target in candidates:
        if overlap < threshold:
            break
        if prediction not in used_predictions and target not in used_ground_truth:
            used_predictions.add(prediction)
            used_ground_truth.add(target)
    return len(used_predictions), len(predictions) - len(used_predictions), \
        len(ground_truth) - len(used_ground_truth)


def summarize(rows):
    tp, fp, fn = (sum(row[key] for row in rows) for key in ('tp', 'fp', 'fn'))
    exact_count = sum(row['predicted_count'] == row['ground_truth_count'] for row in rows)
    zero_error = sum(row['predicted_count'] == row['ground_truth_count'] for row in rows)
    absolute_count_errors = [abs(row['predicted_count'] - row['ground_truth_count'])
                             for row in rows]
    return {'images': len(rows), 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(tp / (tp + fp), 5) if tp + fp else None,
            'recall': round(tp / (tp + fn), 5) if tp + fn else None,
            'exact_cluster_count_images': exact_count,
            'exact_cluster_count_fraction': round(exact_count / len(rows), 5) if rows else None,
            'mean_absolute_count_error': round(statistics.mean(absolute_count_errors), 4)
                if absolute_count_errors else None,
            'cluster_count_error_zero_images': zero_error}


def main():
    detector = TensorRTDetector(str(ENGINE), confidence=0.02)
    results = []
    download_seconds = []
    inference_ms = []
    missing_ids = []
    warmup_done = False
    try:
        for image_id in range(1, 55):
            try:
                started = time.perf_counter()
                encoded = get_bytes(f'{REPO}/{image_id}.png')
                download_seconds.append(time.perf_counter() - started)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    missing_ids.append(image_id)
                    continue
                raise
            image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Unable to decode public dataset image {image_id}.png')
            height, width = image.shape[:2]
            ground_truth = labels_for(image_id, width, height)
            if not warmup_done:
                detector.infer(image)
                warmup_done = True
            detections, latency = detector.infer(image)
            inference_ms.append(latency)
            per_threshold = {}
            for threshold in THRESHOLDS:
                predictions = [{'x1': float(box.x1), 'y1': float(box.y1),
                                'x2': float(box.x2), 'y2': float(box.y2),
                                'confidence': float(box.confidence)}
                               for box in detections if box.confidence >= threshold]
                tp, fp, fn = match(predictions, ground_truth, IOU_THRESHOLD)
                per_threshold[str(threshold)] = {
                    'predicted_count': len(predictions), 'tp': tp, 'fp': fp, 'fn': fn,
                }
            results.append({'image_id': image_id, 'width': width, 'height': height,
                            'ground_truth_count': len(ground_truth),
                            'inference_ms': round(latency, 2),
                            'thresholds': per_threshold})
            print(image_id, len(ground_truth),
                  {key: row['predicted_count'] for key, row in per_threshold.items()},
                  round(latency, 1), 'ms')
    finally:
        detector.close()

    summaries = {}
    for threshold in THRESHOLDS:
        rows = []
        for image in results:
            row = dict(image['thresholds'][str(threshold)])
            row['ground_truth_count'] = image['ground_truth_count']
            rows.append(row)
        summaries[str(threshold)] = summarize(rows)
    output = {
        'method': ('Stream public Project516 images from Hugging Face, run the '
                   'current FRC 9550 1280x960 FP16 TensorRT engine at candidate '
                   'cutoff 0.02, and rescore at confidence 0.02/0.05/0.075/0.10/0.125/0.15/0.20/0.25/0.30/0.40/0.50. Match '
                   'predictions to YOLO box labels by greedy IoU >= 0.50; compare '
                   'per-image predicted piece count with labeled count.'),
        'dataset': {
            'source': 'https://huggingface.co/datasets/project516/rebuilt-fuel-dataset',
            'license': 'AGPL-3.0',
            'images_evaluated': len(results), 'missing_image_ids': missing_ids,
            'ground_truth_piece_count': sum(row['ground_truth_count'] for row in results),
            'image_object_count_histogram': {
                str(count): sum(row['ground_truth_count'] == count for row in results)
                for count in sorted({row['ground_truth_count'] for row in results})},
            'image_sizes': sorted({f"{row['width']}x{row['height']}" for row in results}),
            'dataset_split_limitation': ('The public files do not identify train versus '
                "validation membership. Project516's model was trained on this dataset, "
                'but the evaluated detector is the separate FRC 9550 checkpoint.'),
        },
        'settings': {'engine': str(ENGINE.relative_to(ROOT)),
                     'engine_input': '1280x960', 'base_candidate_confidence': 0.02,
                     'rescore_thresholds': THRESHOLDS, 'iou_threshold': IOU_THRESHOLD,
                     'max_box_aspect_ratio': 2.0},
        'limitations': ('These are monocular still images, so they evaluate FUEL count '
                        'and box localization only, not stereo range or 3D cluster '
                        'position. Labels are from Project516; there is no train/test '
                        'membership manifest. Images were streamed and are not stored.'),
        'timing': {'download_seconds_median': round(statistics.median(download_seconds), 3)
                       if download_seconds else None,
                   'inference_ms_median': round(statistics.median(inference_ms), 2)
                       if inference_ms else None,
                   'inference_ms_p95': round(float(np.percentile(inference_ms, 95)), 2)
                       if inference_ms else None},
        'summary_by_confidence': summaries,
        'per_image': results,
    }
    path = ROOT / 'media/frc-fuel-project516-threshold-sweep.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print('summary', json.dumps(summaries, indent=2))
    print('saved', path)


if __name__ == '__main__':
    main()
