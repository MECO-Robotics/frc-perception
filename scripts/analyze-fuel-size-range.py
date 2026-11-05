#!/usr/bin/env python3
"""Cross-check stereo ranges against the nominal 2026 FUEL apparent-size range."""
import json
import math
import statistics
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / 'media/calibration/stereo_calibration_live_order_candidate.yml'
_calibration = cv2.FileStorage(str(CALIBRATION), cv2.FILE_STORAGE_READ)
FOCAL_PX = float(_calibration.getNode('left_camera_matrix').mat()[0, 0])
_calibration.release()
FUEL_DIAMETER_M = 0.15
TARGETS = ((808, 589), (902, 591))


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(v - center) for v in values]) if values else None


def summarize(rows, model, video):
    targets = []
    for target_index in range(2):
        samples = [row for row in rows if row['target'] == target_index]
        size = [row['size_range_m'] for row in samples]
        stereo = [row['stereo_range_m'] for row in samples if row['stereo_range_m'] is not None]
        paired_delta = [abs(row['size_range_m'] - row['stereo_range_m'])
                        for row in samples if row['stereo_range_m'] is not None]
        targets.append({
            'target': target_index + 1,
            'samples': len(samples),
            'median_box_width_px': round(median([s['width_px'] for s in samples]), 2),
            'median_box_height_px': round(median([s['height_px'] for s in samples]), 2),
            'size_range': {'median_m': round(median(size), 3), 'mad_m': round(mad(size), 3)},
            'stereo_range': {'median_m': round(median(stereo), 3), 'mad_m': round(mad(stereo), 3)},
            'median_absolute_size_stereo_difference_m': round(median(paired_delta), 3),
        })
    return {'model': model, 'video': video, 'targets': targets}


def raw_detector(path, model, choose):
    data = json.loads(path.read_text())
    results = []
    for result in data['results']:
        if not choose(result):
            continue
        rows = []
        for frame in result['frames']:
            for target_index, (tx, ty) in enumerate(TARGETS):
                candidates = []
                for box in frame:
                    if box.get('confidence', 0) < 0.10:
                        continue
                    x1, y1, x2, y2 = box['xyxy']
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    d2 = (cx - tx) ** 2 + (cy - ty) ** 2
                    if d2 <= 60 ** 2:
                        candidates.append((d2, box))
                if not candidates:
                    continue
                box = min(candidates, key=lambda item: item[0])[1]
                x1, y1, x2, y2 = box['xyxy']
                width, height = x2 - x1, y2 - y1
                size_range = FOCAL_PX * FUEL_DIAMETER_M / math.sqrt(width * height)
                rows.append({'target': target_index, 'width_px': width, 'height_px': height,
                             'size_range_m': size_range, 'stereo_range_m': box.get('distance_m')})
        results.append(summarize(rows, model, result['video']))
    return results


def replay(path, video):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        boxes = [b for b in event.get('boxes', [])
                 if b.get('confidence', 0) >= 0.25 and b.get('w', 0) > 0 and b.get('h', 0) > 0]
        # Match the two persistent targets near their known image centers. Coordinates
        # are normalized to the 1600x1200 left-camera view.
        for target_index, (tx, ty) in enumerate(((808 / 1600, 589 / 1200), (902 / 1600, 591 / 1200))):
            candidates = []
            for box in boxes:
                cx, cy = box['x'] + box['w'] / 2, box['y'] + box['h'] / 2
                d2 = ((cx - tx) * 1600) ** 2 + ((cy - ty) * 1200) ** 2
                if d2 <= 60 ** 2:
                    candidates.append((d2, box))
            if not candidates:
                continue
            box = min(candidates, key=lambda item: item[0])[1]
            width, height = box['w'] * 1600, box['h'] * 1200
            size_range = FOCAL_PX * FUEL_DIAMETER_M / math.sqrt(width * height)
            rows.append({'target': target_index, 'width_px': width, 'height_px': height,
                         'size_range_m': size_range, 'stereo_range_m': box.get('distance_m')})
    return summarize(rows, 'FRC 9550 YOLO26n TensorRT 1280 replay', video)


results = []
results += raw_detector(ROOT / 'media/frc-model-checkpoint-sweep-640-tensorrt.json',
                        'FRC 9550 YOLO26n TensorRT 640', lambda _: True)
results += raw_detector(ROOT / 'media/frc-model-checkpoint-sweep-project516.json',
                        'Project516 v1', lambda r: r.get('model') == 'Project516 v1')
results.append(replay(ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr-replay-detections.jsonl',
                      'media/frc-2026-live-two-fuel-60fps-cfr.mp4'))
results.append(replay(ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps-replay-detections.jsonl',
                      'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'))
output = {
    'method': 'Nominal size-derived distance = raw left-camera focal length * official FUEL diameter / sqrt(unrectified detector box width * height); compare to stereo distance on matched boxes. This is an apparent-size consistency proxy, not measured ground truth.',
    'reference': {'fuel_diameter_m': FUEL_DIAMETER_M,
                  'source': '2026 FIRST Robotics Competition Game Manual, section 5.10.1',
                  'url': 'https://firstfrc.blob.core.windows.net/frc2026/Manual/HTML/2026GameManual.htm',
                  'left_camera_focal_px': FOCAL_PX,
                  'calibration': 'media/calibration/stereo_calibration_live_order_candidate.yml'},
    'results': results,
}
(ROOT / 'media/frc-ball-size-range-crosscheck.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output, indent=2))
