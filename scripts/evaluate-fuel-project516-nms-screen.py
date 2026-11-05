#!/usr/bin/env python3
"""Screen post-NMS IoU settings on the three worst dense-count frames."""
import importlib.util
import json
import os
import sys
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
IMAGE_IDS = (29, 32, 34)
IOU_VALUES = (0.15, 0.25, 0.35, 0.45, 0.60, 0.80)
SCORE_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50)

inputs = {}
for image_id in IMAGE_IDS:
    data = module.get_bytes(f'{module.REPO}/{image_id}.png')
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    height, width = frame.shape[:2]
    labels = module.labels_for(image_id, width, height)
    inputs[image_id] = (frame, labels)

output = {
    'method': 'Screen post-decode class-wise NMS IoU settings on images 29, 32, and 34, selected because the default 0.45 run had the largest count overestimates. The model candidate cutoff is 0.02; detector output counts and IoU>=0.5 label matches are rescored at confidence 0.10 through 0.50.',
    'dataset': 'https://huggingface.co/datasets/project516/rebuilt-fuel-dataset',
    'iou_values': IOU_VALUES,
    'confidence_thresholds': SCORE_THRESHOLDS,
    'images': [],
}
for image_id, (frame, labels) in inputs.items():
    output['images'].append({'image_id': image_id, 'ground_truth_count': len(labels), 'by_iou': {}})

for iou_value in IOU_VALUES:
    detector = TensorRTDetector(str(ENGINE), confidence=0.02, iou=iou_value)
    try:
        for image_id, (frame, labels) in inputs.items():
            detections, inference_ms = detector.infer(frame)
            predictions_all = [dict(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                                    confidence=d.confidence) for d in detections]
            per_conf = {}
            for threshold in SCORE_THRESHOLDS:
                predictions = [d for d in predictions_all if d['confidence'] >= threshold]
                tp, fp, fn = module.match(predictions, labels, 0.5)
                per_conf[str(threshold)] = {'predicted_count': len(predictions),
                    'tp': tp, 'fp': fp, 'fn': fn}
            row = next(row for row in output['images'] if row['image_id'] == image_id)
            row['by_iou'][str(iou_value)] = {
                'inference_ms': round(inference_ms, 2), 'thresholds': per_conf}
    finally:
        detector.close()
    print('completed NMS IoU', iou_value, flush=True)

path = ROOT / 'media/frc-fuel-project516-nms-screen.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print('saved', path)
for row in output['images']:
    print('image', row['image_id'], 'GT', row['ground_truth_count'])
    for threshold in SCORE_THRESHOLDS:
        print(' confidence', threshold, {iou: row['by_iou'][str(iou)]['thresholds'][str(threshold)]['predicted_count'] for iou in IOU_VALUES})
