#!/usr/bin/env python3
"""Sweep confidence-based fallback policies on multi-count labeled FUEL images."""
import json
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
REPO = 'https://huggingface.co/datasets/project516/rebuilt-fuel-dataset/resolve/main'
FAST_ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_640_fp16.engine'
FULL_ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
SCORE_THRESHOLDS = (0.05, 0.075, 0.10)
FALLBACK_THRESHOLDS = (0.0575, 0.075, 0.10, 0.15, 0.20)
CANDIDATE_CONFIDENCE = 0.05
NMS_IOU = 0.15
MATCH_IOU = 0.50

os.environ.update({'SCOUT_CLASS_NAMES': '["ball"]',
                   'SCOUT_MAX_BOX_ASPECT_RATIO': '2.0',
                   'SCOUT_NMS_IOU': str(NMS_IOU),
                   'SCOUT_REUSE_PREPROCESS': '1'})
from scout.detectors import TensorRTDetector


def get_bytes(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Scout-FRC-evaluation/1.0'})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()


def load_dataset():
    samples, missing = [], []
    for image_id in range(1, 55):
        try:
            encoded = get_bytes(f'{REPO}/{image_id}.png')
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                missing.append(image_id)
                continue
            raise
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Unable to decode image {image_id}')
        height, width = image.shape[:2]
        labels = []
        for line in get_bytes(f'{REPO}/{image_id}.txt').decode().strip().splitlines():
            class_id, cx, cy, bw, bh = map(float, line.split())
            labels.append({'class_id': int(class_id),
                           'x1': (cx - bw / 2) * width, 'y1': (cy - bh / 2) * height,
                           'x2': (cx + bw / 2) * width, 'y2': (cy + bh / 2) * height})
        samples.append({'image_id': image_id, 'image': image, 'labels': labels})
    return samples, missing


def pack(detections):
    return [{'x1': float(b.x1), 'y1': float(b.y1), 'x2': float(b.x2),
             'y2': float(b.y2), 'confidence': float(b.confidence)} for b in detections]


def iou(a, b):
    x1, y1 = max(a['x1'], b['x1']), max(a['y1'], b['y1'])
    x2, y2 = min(a['x2'], b['x2']), min(a['y2'], b['y2'])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a['x2'] - a['x1']) * max(0, a['y2'] - a['y1'])
    ab = max(0, b['x2'] - b['x1']) * max(0, b['y2'] - b['y1'])
    return intersection / (aa + ab - intersection) if aa + ab > intersection else 0.0


def count_metrics(predictions, labels):
    pairs = sorted(((iou(p, g), pi, gi) for pi, p in enumerate(predictions)
                    for gi, g in enumerate(labels)), reverse=True)
    used_p, used_g = set(), set()
    for overlap, pi, gi in pairs:
        if overlap < MATCH_IOU:
            break
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    tp = len(used_p)
    fp, fn = len(predictions) - tp, len(labels) - tp
    return tp, fp, fn, abs(len(predictions) - len(labels)) == 0, abs(len(predictions) - len(labels))


def summarize(rows, fallback_type, fallback_threshold, score_threshold):
    tp = fp = fn = exact = 0
    errors, detector_cost = [], []
    fallback_frames = 0
    per_image = []
    for row in rows:
        fast = [b for b in row['fast'] if b['confidence'] >= CANDIDATE_CONFIDENCE]
        min_conf = min((b['confidence'] for b in fast), default=0.0)
        if fallback_type == 'two_ball':
            fallback = len(fast) != 2 or min_conf < fallback_threshold
        else:
            fallback = not fast or min_conf < fallback_threshold
        fallback_frames += fallback
        detections = row['full'] if fallback else row['fast']
        predictions = [b for b in detections if b['confidence'] >= score_threshold]
        a, b, c, is_exact, error = count_metrics(predictions, row['labels'])
        tp += a; fp += b; fn += c; exact += is_exact; errors.append(error)
        detector_cost.append(row['fast_ms'] + (row['full_ms'] if fallback else 0.0))
        per_image.append({'image_id': row['image_id'],
                          'ground_truth_count': len(row['labels']),
                          'predicted_count': len(predictions), 'fallback': fallback,
                          'tp': a, 'fp': b, 'fn': c})
    return {
        'images': len(rows), 'fallback_frames': fallback_frames,
        'fallback_fraction': round(fallback_frames / len(rows), 5) if rows else None,
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(tp / (tp + fp), 5) if tp + fp else None,
        'recall': round(tp / (tp + fn), 5) if tp + fn else None,
        'exact_count_images': exact,
        'exact_count_fraction': round(exact / len(rows), 5) if rows else None,
        'count_mae': round(statistics.mean(errors), 5) if errors else None,
        'estimated_detector_ms_median': round(statistics.median(detector_cost), 2) if detector_cost else None,
        'estimated_detector_ms_p95': round(float(np.percentile(detector_cost, 95)), 2) if detector_cost else None,
        'per_image': per_image,
    }


def run_model(engine, samples):
    detector = TensorRTDetector(str(engine), confidence=CANDIDATE_CONFIDENCE, iou=NMS_IOU)
    rows, latencies = [], []
    try:
        for sample in samples:
            detections, latency = detector.infer(sample['image'])
            rows.append(pack(detections))
            latencies.append(latency)
    finally:
        detector.close()
    return rows, latencies


def main():
    samples, missing = load_dataset()
    full_outputs, full_latencies = run_model(FULL_ENGINE, samples)
    fast_outputs, fast_latencies = run_model(FAST_ENGINE, samples)
    rows = [{'image_id': s['image_id'], 'labels': s['labels'],
             'full': full_outputs[i], 'fast': fast_outputs[i],
             'full_ms': full_latencies[i], 'fast_ms': fast_latencies[i]}
            for i, s in enumerate(samples)]
    policies = {}
    per_image = {}
    for fallback_type in ('two_ball', 'count_agnostic'):
        per_image[fallback_type] = {}
        for fallback_threshold in FALLBACK_THRESHOLDS:
            key = f'{fallback_type}/fallback-{fallback_threshold:g}'
            per_image[fallback_type][str(fallback_threshold)] = {}
            for score_threshold in SCORE_THRESHOLDS:
                result = summarize(rows, fallback_type, fallback_threshold, score_threshold)
                result.pop('per_image')
                policies[f'{key}/score-{score_threshold:g}'] = result
            per_image[fallback_type][str(fallback_threshold)] = summarize(
                rows, fallback_type, fallback_threshold, 0.05)['per_image']
    output = {
        'method': ('Run the 1280x960 and 640 FP16 engines at confidence 0.05 once each '
                   'on every available labeled image, then post-score two fallback '
                   'policies across candidate '
                   'minimum confidences 0.0575/0.075/0.10/0.15/0.20. Two-ball falls back '
                   'unless fast count is exactly two; count-agnostic falls back only for '
                   'empty output. Both fall back if the weakest fast box is below the '
                   'swept threshold. Count metrics use greedy IoU >= 0.50.'),
        'limitations': ('Monocular still images only; no stereo/range labels. There is '
                        'no train/validation membership manifest, and this is one small '
                        'dataset rather than a robot field sequence. Latency is a per-image '
                        'component-cost estimate and excludes pipeline/stereo scheduling.'),
        'dataset': {'source': 'https://huggingface.co/datasets/project516/rebuilt-fuel-dataset',
                    'license': 'AGPL-3.0', 'images': len(samples),
                    'missing_image_ids': missing,
                    'ground_truth_count_histogram': {
                        str(n): sum(len(s['labels']) == n for s in samples)
                        for n in sorted({len(s['labels']) for s in samples})}},
        'configuration': {'full_engine': str(FULL_ENGINE.relative_to(ROOT)),
                          'fast_engine': str(FAST_ENGINE.relative_to(ROOT)),
                          'candidate_confidence': CANDIDATE_CONFIDENCE, 'nms_iou': NMS_IOU,
                          'max_aspect_ratio': 2.0, 'fallback_thresholds': FALLBACK_THRESHOLDS,
                          'score_thresholds': SCORE_THRESHOLDS,
                          'match_iou': MATCH_IOU},
        'fallback_policy_sweep': policies,
        'per_image_at_score_confidence_0_05': per_image,
        'engine_latency_ms': {
            'full': {'median': round(statistics.median(full_latencies), 2),
                     'p95': round(float(np.percentile(full_latencies, 95)), 2)},
            'fast': {'median': round(statistics.median(fast_latencies), 2),
                     'p95': round(float(np.percentile(fast_latencies, 95)), 2)}},
    }
    path = ROOT / 'media/frc-fuel-cascade-fallback-threshold-public-sweep.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'images': len(samples),
                      'count_histogram': output['dataset']['ground_truth_count_histogram'],
                      'engine_latency_ms': output['engine_latency_ms'],
                      'fallback_policy_sweep': policies}, indent=2))
    print('saved', path)


if __name__ == '__main__':
    main()
