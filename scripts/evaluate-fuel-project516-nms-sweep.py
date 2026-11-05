#!/usr/bin/env python3
"""Sweep post-decode NMS IoU against labeled FUEL object counts."""
import importlib.util
import json
import os
import statistics
import sys
import time
import urllib.error
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '2.0'
from scout.detectors import TensorRTDetector

spec = importlib.util.spec_from_file_location(
    'project516_eval', ROOT / 'scripts/evaluate-fuel-project516-public-dataset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
IOU_VALUES = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.45)
CONFIDENCE_VALUES = (0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def summarize(rows):
    tp = sum(row['tp'] for row in rows)
    fp = sum(row['fp'] for row in rows)
    fn = sum(row['fn'] for row in rows)
    errors = [abs(row['predicted_count'] - row['ground_truth_count']) for row in rows]
    exact = sum(error == 0 for error in errors)
    return {
        'images': len(rows), 'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(tp / (tp + fp), 5) if tp + fp else None,
        'recall': round(tp / (tp + fn), 5) if tp + fn else None,
        'exact_count_images': exact,
        'exact_count_fraction': round(exact / len(rows), 5) if rows else None,
        'mean_absolute_count_error': round(statistics.mean(errors), 4) if errors else None,
    }


detector = TensorRTDetector(str(ENGINE), confidence=min(CONFIDENCE_VALUES), iou=0.45)
results = []
missing_ids = []
image_transfer_s = []
inference_latency_ms = {str(iou): [] for iou in IOU_VALUES}
try:
    warmed = False
    for image_id in range(1, 55):
        started = time.perf_counter()
        try:
            encoded = module.get_bytes(f'{module.REPO}/{image_id}.png')
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                missing_ids.append(image_id)
                continue
            raise
        image_transfer_s.append(time.perf_counter() - started)
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Unable to decode {image_id}.png')
        height, width = image.shape[:2]
        ground_truth = module.labels_for(image_id, width, height)
        if not warmed:
            detector.infer(image)
            warmed = True
        result = {'image_id': image_id, 'width': width, 'height': height,
                  'ground_truth_count': len(ground_truth), 'by_iou': {}}
        for iou in IOU_VALUES:
            detector.iou = iou
            detections, latency = detector.infer(image)
            inference_latency_ms[str(iou)].append(latency)
            per_threshold = {}
            for threshold in CONFIDENCE_VALUES:
                predictions = [{'x1': float(box.x1), 'y1': float(box.y1),
                                'x2': float(box.x2), 'y2': float(box.y2),
                                'confidence': float(box.confidence)}
                               for box in detections if box.confidence >= threshold]
                tp, fp, fn = module.match(predictions, ground_truth, 0.5)
                per_threshold[str(threshold)] = {
                    'predicted_count': len(predictions), 'tp': tp, 'fp': fp, 'fn': fn,
                }
            result['by_iou'][str(iou)] = {
                'inference_ms': round(latency, 2), 'thresholds': per_threshold}
        results.append(result)
        print('image', image_id, 'GT', len(ground_truth), 'done', flush=True)
finally:
    detector.close()

summaries = {}
for iou in IOU_VALUES:
    summaries[str(iou)] = {}
    for threshold in CONFIDENCE_VALUES:
        rows = []
        for image in results:
            row = dict(image['by_iou'][str(iou)]['thresholds'][str(threshold)])
            row['ground_truth_count'] = image['ground_truth_count']
            rows.append(row)
        summaries[str(iou)][str(threshold)] = summarize(rows)

output = {
    'method': ('Stream Project516 labeled images from Hugging Face, run the FRC 9550 '
               '1280x960 FP16 TensorRT detector with candidate confidence 0.02, vary '
               'Scout post-decode class-wise NMS IoU, and rescore confidence thresholds. '
               'Match by greedy IoU >= 0.50 and compare per-image piece counts.'),
    'dataset': {'source': 'https://huggingface.co/datasets/project516/rebuilt-fuel-dataset',
                'license': 'AGPL-3.0', 'images_evaluated': len(results),
                'missing_image_ids': missing_ids,
                'ground_truth_piece_count': sum(row['ground_truth_count'] for row in results),
                'image_sizes': sorted({f"{row['width']}x{row['height']}" for row in results}),
                'split_limitation': 'No train/validation membership manifest is published. The tested FRC 9550 checkpoint is separate from Project516 weights.'},
    'engine': str(ENGINE.relative_to(ROOT)),
    'candidate_confidence': min(CONFIDENCE_VALUES),
    'iou_values': IOU_VALUES,
    'confidence_values': CONFIDENCE_VALUES,
    'label_iou_threshold': 0.5,
    'timing': {'image_download_s_median': round(statistics.median(image_transfer_s), 3) if image_transfer_s else None,
               'inference_ms_median_by_iou': {k: round(statistics.median(v), 2) for k, v in inference_latency_ms.items()}},
    'summary_by_iou_and_confidence': summaries,
    'per_image': results,
    'limitations': ('Monocular images evaluate 2D count only. They do not validate stereo range, '
                    'and labels do not identify a held-out split.'),
}
path = ROOT / 'media/frc-fuel-project516-nms-sweep.json'
path.write_text(json.dumps(output, indent=2) + '\n')
for iou in IOU_VALUES:
    print('NMS', iou, json.dumps({threshold: {
        'exact': summaries[str(iou)][str(threshold)]['exact_count_images'],
        'mae': summaries[str(iou)][str(threshold)]['mean_absolute_count_error'],
        'p': summaries[str(iou)][str(threshold)]['precision'],
        'r': summaries[str(iou)][str(threshold)]['recall']}
        for threshold in ('0.05', '0.075', '0.1', '0.2', '0.3', '0.4', '0.5')}), flush=True)
print('saved', path)
