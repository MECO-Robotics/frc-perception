#!/usr/bin/env python3
"""Compare the 1280-only detector and two-ball cascade on labeled FUEL images."""
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
BASE_ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
FAST_ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_640_fp16.engine'
CONFIDENCE = 0.05
THRESHOLDS = (0.05, 0.075, 0.10)
NMS_IOU = 0.15
MATCH_IOU = 0.50

os.environ.update({
    'SCOUT_CLASS_NAMES': '["ball"]',
    'SCOUT_MAX_BOX_ASPECT_RATIO': '2.0',
    'SCOUT_NMS_IOU': str(NMS_IOU),
    'SCOUT_REUSE_PREPROCESS': '1',
})
from scout.detectors import TensorRTDetector, create_detector


def get_bytes(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Scout-FRC-evaluation/1.0'})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()


def parse_labels(image_id, width, height):
    text = get_bytes(f'{REPO}/{image_id}.txt').decode('utf-8').strip()
    out = []
    for line in text.splitlines():
        class_id, cx, cy, bw, bh = map(float, line.split())
        out.append({'class_id': int(class_id),
                    'x1': (cx - bw / 2) * width, 'y1': (cy - bh / 2) * height,
                    'x2': (cx + bw / 2) * width, 'y2': (cy + bh / 2) * height})
    return out


def as_boxes(detections):
    return [{'x1': float(d.x1), 'y1': float(d.y1), 'x2': float(d.x2),
             'y2': float(d.y2), 'confidence': float(d.confidence)}
            for d in detections]


def iou(a, b):
    x1, y1 = max(a['x1'], b['x1']), max(a['y1'], b['y1'])
    x2, y2 = min(a['x2'], b['x2']), min(a['y2'], b['y2'])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a['x2'] - a['x1']) * max(0, a['y2'] - a['y1'])
    area_b = max(0, b['x2'] - b['x1']) * max(0, b['y2'] - b['y1'])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match(predictions, ground_truth):
    pairs = sorted(((iou(p, g), pi, gi)
                    for pi, p in enumerate(predictions)
                    for gi, g in enumerate(ground_truth)), reverse=True)
    used_p, used_g = set(), set()
    for overlap, pi, gi in pairs:
        if overlap < MATCH_IOU:
            break
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    return len(used_p), len(predictions) - len(used_p), len(ground_truth) - len(used_g)


def summarize(rows, threshold):
    tp = fp = fn = exact = 0
    errors = []
    for row in rows:
        predictions = [b for b in row['predictions'] if b['confidence'] >= threshold]
        a, b, c = match(predictions, row['labels'])
        tp += a
        fp += b
        fn += c
        error = abs(len(predictions) - len(row['labels']))
        exact += error == 0
        errors.append(error)
    return {'images': len(rows), 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(tp / (tp + fp), 5) if tp + fp else None,
            'recall': round(tp / (tp + fn), 5) if tp + fn else None,
            'exact_count_images': exact,
            'exact_count_fraction': round(exact / len(rows), 5) if rows else None,
            'count_mae': round(statistics.mean(errors), 5) if errors else None}


def main():
    samples, missing, download_s = [], [], []
    for image_id in range(1, 55):
        try:
            start = time.perf_counter()
            encoded = get_bytes(f'{REPO}/{image_id}.png')
            download_s.append(time.perf_counter() - start)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                missing.append(image_id)
                continue
            raise
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Unable to decode image {image_id}')
        height, width = image.shape[:2]
        samples.append({'image_id': image_id, 'image': image,
                        'labels': parse_labels(image_id, width, height)})

    results = {'baseline_1280': [], 'two_ball_cascade': [],
               'count_agnostic_cascade': []}
    timing = {key: [] for key in results}
    fallback_reasons = {'two_ball_cascade': {}, 'count_agnostic_cascade': {}}
    detector = TensorRTDetector(str(BASE_ENGINE), confidence=CONFIDENCE, iou=NMS_IOU)
    try:
        for sample in samples:
            detections, latency = detector.infer(sample['image'])
            results['baseline_1280'].append({'image_id': sample['image_id'],
                                             'labels': sample['labels'],
                                             'predictions': as_boxes(detections)})
            timing['baseline_1280'].append(latency)
    finally:
        detector.close()

    os.environ['SCOUT_CASCADE_FAST_ENGINE'] = str(FAST_ENGINE)
    os.environ['SCOUT_CASCADE_FALLBACK_ENGINE'] = str(BASE_ENGINE)
    os.environ['SCOUT_CASCADE_MIN_CONFIDENCE'] = '0.0575'
    os.environ['SCOUT_CASCADE_COUNT_AGNOSTIC'] = '0'
    detector = create_detector('tensorrt', engine_path=str(BASE_ENGINE),
                               confidence=CONFIDENCE, iou=NMS_IOU)
    try:
        for sample in samples:
            detections, latency = detector.infer(sample['image'])
            results['two_ball_cascade'].append({'image_id': sample['image_id'],
                                                'labels': sample['labels'],
                                                'predictions': as_boxes(detections)})
            timing['two_ball_cascade'].append(latency)
            reason = detector.last_profile.get('cascade_fallback_reason', 'unknown')
            fallback_reasons['two_ball_cascade'][reason] = (
                fallback_reasons['two_ball_cascade'].get(reason, 0) + 1)
    finally:
        detector.close()

    os.environ['SCOUT_CASCADE_COUNT_AGNOSTIC'] = '1'
    detector = create_detector('tensorrt', engine_path=str(BASE_ENGINE),
                               confidence=CONFIDENCE, iou=NMS_IOU)
    try:
        for sample in samples:
            detections, latency = detector.infer(sample['image'])
            results['count_agnostic_cascade'].append({
                'image_id': sample['image_id'], 'labels': sample['labels'],
                'predictions': as_boxes(detections)})
            timing['count_agnostic_cascade'].append(latency)
            reason = detector.last_profile.get('cascade_fallback_reason', 'unknown')
            fallback_reasons['count_agnostic_cascade'][reason] = (
                fallback_reasons['count_agnostic_cascade'].get(reason, 0) + 1)
    finally:
        detector.close()

    summary = {}
    for profile, rows in results.items():
        summary[profile] = {
            str(threshold): summarize(rows, threshold) for threshold in THRESHOLDS}
    output = {
        'method': ('Compare 1280-only FP16 TensorRT with the current 640-first, '
                   '1280-fallback detector cascade on all available Project516 '
                   'labeled still images. Both use confidence candidate 0.05, NMS '
                   '0.15, max aspect ratio 2.0, and score at confidence 0.05/0.075/0.10. '
                   'Greedy box matching uses IoU >= 0.50. The two-ball cascade falls '
                   'back unless exactly two fast boxes remain and both have confidence '
                   '>= 0.0575. The count-agnostic variant falls back for no fast boxes '
                   'or any fast box below 0.0575, and accepts any other predicted count.'),
        'limitations': ('Monocular images provide count/box labels only; no range or '
                        '3D cluster truth is available. Dataset split membership is '
                        'not identified, and images are from Project516 rather than '
                        'a field set collected by this robot.'),
        'dataset': {'source': 'https://huggingface.co/datasets/project516/rebuilt-fuel-dataset',
                    'license': 'AGPL-3.0', 'images': len(samples), 'missing_image_ids': missing,
                    'ground_truth_count_histogram': {
                        str(n): sum(len(s['labels']) == n for s in samples)
                        for n in sorted({len(s['labels']) for s in samples})}},
        'configuration': {'baseline_engine': str(BASE_ENGINE.relative_to(ROOT)),
                          'fast_engine': str(FAST_ENGINE.relative_to(ROOT)),
                          'confidence_candidate': CONFIDENCE, 'nms_iou': NMS_IOU,
                          'score_thresholds': THRESHOLDS, 'max_aspect_ratio': 2.0,
                          'cascade_fallback_min_confidence': 0.0575,
                          'cascade_expected_fast_box_count': 2,
                          'count_agnostic_policy': 'fallback for empty or weak fast outputs',
                          'match_iou': MATCH_IOU},
        'fallback_reasons': fallback_reasons,
        'timing_ms': {profile: {
            'median': round(statistics.median(values), 2) if values else None,
            'p95': round(float(np.percentile(values, 95)), 2) if values else None}
            for profile, values in timing.items()},
        'median_download_s': round(statistics.median(download_s), 3) if download_s else None,
        'summary_by_profile_and_threshold': summary,
        'per_image': [{
            'image_id': sample['image_id'], 'ground_truth_count': len(sample['labels']),
            'baseline_counts': {str(t): sum(b['confidence'] >= t for b in results['baseline_1280'][i]['predictions']) for t in THRESHOLDS},
            'two_ball_cascade_counts': {str(t): sum(b['confidence'] >= t for b in results['two_ball_cascade'][i]['predictions']) for t in THRESHOLDS},
            'count_agnostic_cascade_counts': {str(t): sum(b['confidence'] >= t for b in results['count_agnostic_cascade'][i]['predictions']) for t in THRESHOLDS},
        } for i, sample in enumerate(samples)],
    }
    path = ROOT / 'media/frc-fuel-cascade-public-dataset-evaluation.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'images': len(samples), 'missing': missing,
                      'ground_truth_count_histogram': output['dataset']['ground_truth_count_histogram'],
                      'fallback_reasons': fallback_reasons,
                      'timing_ms': output['timing_ms'],
                      'summary_by_profile_and_threshold': summary}, indent=2))
    print('saved', path)


if __name__ == '__main__':
    main()
