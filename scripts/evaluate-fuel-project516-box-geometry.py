#!/usr/bin/env python3
"""Evaluate size and aspect filters on FRC 9550 predictions for labeled FUEL."""
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
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '0'
from scout.detectors import TensorRTDetector

spec = importlib.util.spec_from_file_location(
    'project516_eval', ROOT / 'scripts/evaluate-fuel-project516-public-dataset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
NMS_IOU = 0.10
BASE_CONFIDENCE = 0.02
CONFIDENCES = (0.05, 0.10, 0.20, 0.40)
MIN_SHORT_SIDE_PX = (0, 8, 12, 16, 20, 24, 28, 32)
MAX_ASPECT_RATIOS = (1.2, 1.4, 1.6, 1.8, 2.0)


def metric(rows):
    tp = sum(r['tp'] for r in rows)
    fp = sum(r['fp'] for r in rows)
    fn = sum(r['fn'] for r in rows)
    errors = [abs(r['predicted_count'] - r['ground_truth_count']) for r in rows]
    exact = sum(error == 0 for error in errors)
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(tp / (tp + fp), 5) if tp + fp else None,
            'recall': round(tp / (tp + fn), 5) if tp + fn else None,
            'exact_images': exact,
            'exact_fraction': round(exact / len(rows), 5) if rows else None,
            'mae': round(statistics.mean(errors), 4) if errors else None}


detector = TensorRTDetector(str(ENGINE), confidence=BASE_CONFIDENCE, iou=NMS_IOU)
per_image = []
image_downloads = []
try:
    warmed = False
    for image_id in range(1, 55):
        start = time.perf_counter()
        try:
            encoded = module.get_bytes(f'{module.REPO}/{image_id}.png')
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        image_downloads.append(time.perf_counter() - start)
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Unable to decode image {image_id}.png')
        height, width = image.shape[:2]
        labels = module.labels_for(image_id, width, height)
        if not warmed:
            detector.infer(image)
            warmed = True
        boxes, latency = detector.infer(image)
        per_image.append({'image_id': image_id, 'width': width, 'height': height,
            'ground_truth_count': len(labels), 'inference_ms': round(latency, 2),
            'labels': labels,
            'predictions': [{'x1': float(b.x1), 'y1': float(b.y1),
                             'x2': float(b.x2), 'y2': float(b.y2),
                             'confidence': float(b.confidence)} for b in boxes]})
        print('image', image_id, 'gt', len(labels), 'pred', len(boxes), flush=True)
finally:
    detector.close()

results = []
for min_side in MIN_SHORT_SIDE_PX:
    for max_aspect in MAX_ASPECT_RATIOS:
        for confidence in CONFIDENCES:
            rows = []
            for image in per_image:
                predictions = []
                for box in image['predictions']:
                    w, h = box['x2'] - box['x1'], box['y2'] - box['y1']
                    if box['confidence'] < confidence or min(w, h) < min_side:
                        continue
                    aspect = max(w / max(h, 1e-9), h / max(w, 1e-9))
                    if aspect > max_aspect:
                        continue
                    predictions.append(box)
                tp, fp, fn = module.match(predictions, image['labels'], 0.5)
                rows.append({'predicted_count': len(predictions),
                             'ground_truth_count': image['ground_truth_count'],
                             'tp': tp, 'fp': fp, 'fn': fn})
            results.append({'min_short_side_px': min_side,
                            'max_aspect_ratio': max_aspect,
                            'confidence': confidence, **metric(rows)})

output = {
    'method': ('Stream Project516 labeled images, run the FRC 9550 engine with NMS IoU 0.10 '
               'and candidate confidence 0.02, then sweep confidence, minimum box short side, '
               'and maximum box aspect ratio. Match by greedy IoU >= 0.50.'),
    'dataset': 'https://huggingface.co/datasets/project516/rebuilt-fuel-dataset',
    'engine': str(ENGINE.relative_to(ROOT)), 'images': len(per_image),
    'ground_truth_objects': sum(row['ground_truth_count'] for row in per_image),
    'nms_iou': NMS_IOU, 'base_candidate_confidence': BASE_CONFIDENCE,
    'min_short_side_px_values': MIN_SHORT_SIDE_PX,
    'max_aspect_ratio_values': MAX_ASPECT_RATIOS,
    'confidence_values': CONFIDENCES,
    'download_seconds_median': round(statistics.median(image_downloads), 3) if image_downloads else None,
    'sweep': results, 'per_image_predictions': per_image,
    'limitations': ('Monocular set with no official train/validation manifest; size filters may '
                    'confound range with confidence or reject distant balls. No stereo range is evaluated.')}
path = ROOT / 'media/frc-fuel-project516-box-geometry-evaluation.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print('saved', path)
for confidence in CONFIDENCES:
    options = [r for r in results if r['confidence'] == confidence]
    best = sorted(options, key=lambda r:(r['mae'], -r['recall'], -r['exact_images']))[:3]
    print('confidence', confidence, 'best', json.dumps(best), flush=True)
