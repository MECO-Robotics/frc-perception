#!/usr/bin/env python3
"""Compare Project516 v1 detector sizes on synchronized saved FUEL videos."""
import json
import statistics
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'models/project516_rebuilt_fuel_v1.pt'
VIDEOS = (
    ('earlier', ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr.mp4'),
    ('corrected-settings', ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'),
)
TARGETS = ((808, 589), (902, 591))
CONFIDENCES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65)
STRIDE = 10
IMGSZ = (960, 1280)


def evaluate(rows, confidence):
    tp = fp = fn = exact = paired = 0
    for row in rows:
        candidates = [box for box in row['boxes'] if box['confidence'] >= confidence]
        matched = [None, None]
        frame_fp = 0
        for box in sorted(candidates, key=lambda item: item['confidence'], reverse=True):
            cx = (box['x1'] + box['x2']) / 2
            cy = (box['y1'] + box['y2']) / 2
            distance, target = min((((cx - x) ** 2 + (cy - y) ** 2) ** 0.5, i)
                                   for i, (x, y) in enumerate(TARGETS))
            if distance <= 60 and matched[target] is None:
                matched[target] = box
            else:
                frame_fp += 1
        count = sum(item is not None for item in matched)
        tp += count
        fn += 2 - count
        fp += frame_fp
        paired += count == 2
        exact += count == 2 and frame_fp == 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {
        'confidence': confidence,
        'tp': tp, 'fp': fp, 'fn': fn,
        'paired_frames': paired, 'exact_two_frames': exact,
        'precision': round(precision, 5), 'recall': round(recall, 5),
        'f1': round(2 * precision * recall / (precision + recall), 5)
        if precision + recall else 0,
    }


torch.set_num_threads(4)
model = YOLO(str(WEIGHTS))
output = {
    'weights': str(WEIGHTS.relative_to(ROOT)),
    'task': model.task,
    'names': model.names,
    'method': f'Run Project516 v1 at square input sizes {list(IMGSZ)} on every {STRIDE}th decoded frame from each saved video. Apply the same 60 px target windows and max box aspect ratio 2.0 as the active FRC9550 detector; sweep confidence after inference.',
    'limitations': 'Sparse 10 fps sampling; PyTorch CPU latency is not comparable to TensorRT. Fixed target windows do not test arbitrary FUEL positions. Stereo range is omitted because previous Project516 v1 boxes disagreed substantially with the stereo and size-based estimates.',
    'stride_frames': STRIDE,
    'results': [],
}

for imgsz in IMGSZ:
    for scene, video_path in VIDEOS:
        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        rows, latencies = [], []
        started = time.perf_counter()
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % STRIDE == 0:
                left = frame[:, :1600]
                t0 = time.perf_counter()
                result = model.predict(left, imgsz=imgsz, conf=0.05,
                                       device='cpu', verbose=False, max_det=100)[0]
                latencies.append((time.perf_counter() - t0) * 1000)
                boxes = []
                if result.boxes is not None:
                    for xyxy, confidence in zip(result.boxes.xyxy.cpu().tolist(),
                                                result.boxes.conf.cpu().tolist()):
                        x1, y1, x2, y2 = map(float, xyxy)
                        width, height = x2 - x1, y2 - y1
                        if height <= 0 or width / height > 2.0 or height / width > 2.0:
                            continue
                        boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                      'confidence': float(confidence)})
                rows.append({'frame': index, 'time_s': round(index / fps, 4) if fps else None,
                             'boxes': boxes})
            index += 1
        capture.release()
        output['results'].append({
            'scene': scene,
            'video': str(video_path.relative_to(ROOT)),
            'imgsz': imgsz,
            'fps': fps,
            'reported_frames': total,
            'sampled_frames': len(rows),
            'decode_and_inference_seconds': round(time.perf_counter() - started, 2),
            'single_view_inference_ms_median': round(statistics.median(latencies), 2)
            if latencies else None,
            'threshold_sweep': [evaluate(rows, confidence) for confidence in CONFIDENCES],
            'frames': rows,
        })

out_path = ROOT / 'media/frc-project516-v1-dense-sampled-evaluation.json'
out_path.write_text(json.dumps(output, indent=2) + '\n')
for result in output['results']:
    print(result['scene'], 'imgsz', result['imgsz'], 'frames', result['sampled_frames'],
          'median_ms', result['single_view_inference_ms_median'])
    for summary in result['threshold_sweep']:
        print(summary['confidence'], 'exact', summary['exact_two_frames'],
              '/', result['sampled_frames'], 'pair', summary['paired_frames'],
              'fp/fn', summary['fp'], summary['fn'], 'F1', summary['f1'])
print('saved', out_path)
