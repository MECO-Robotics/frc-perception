#!/usr/bin/env python3
"""Evaluate whether Project516 v1 consensus improves FRC9550 FUEL counts.

Consumes existing replay outputs only; it does not open the camera or rerun
inference. The matched replay is every tenth frame at 960 px from each clip.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ((808, 589), (902, 591))
FRC_CONFIDENCES = (0.05, 0.10)
P516_CONFIDENCES = (0.05, 0.10, 0.20, 0.25)
CONSENSUS_IOUS = (0.25, 0.50)


def box_iou(a, b):
    x1, y1 = max(a['x1'], b['x1']), max(a['y1'], b['y1'])
    x2, y2 = min(a['x2'], b['x2']), min(a['y2'], b['y2'])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a['x2'] - a['x1']) * max(0.0, a['y2'] - a['y1'])
    area_b = max(0.0, b['x2'] - b['x1']) * max(0.0, b['y2'] - b['y1'])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def score(rows):
    tp = fp = fn = exact = paired = 0
    for boxes in rows:
        matched = [False, False]
        extras = 0
        for box in boxes:
            cx = (box['x1'] + box['x2']) / 2
            cy = (box['y1'] + box['y2']) / 2
            errors = [math.hypot(cx - x, cy - y) for x, y in TARGETS]
            target = min(range(2), key=errors.__getitem__)
            if errors[target] <= 60 and not matched[target]:
                matched[target] = True
            else:
                extras += 1
        count = sum(matched)
        tp += count
        fn += 2 - count
        fp += extras
        paired += count == 2
        exact += count == 2 and extras == 0
    return {
        'frames': len(rows), 'exact_two_frames': exact, 'paired_frames': paired,
        'tp': tp, 'fp': fp, 'fn': fn,
        'recall': round(tp / (tp + fn), 5) if tp + fn else 0.0,
    }


frc_data = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
p516_data = json.loads((ROOT / 'media/frc-project516-v1-dense-sampled-evaluation.json').read_text())
output = {
    'method': 'Post-process matched frames from existing FRC9550 full-clip replay and Project516 v1 960 px replay. Retain an FRC9550 box only when it overlaps a Project516 box above the configured IoU. Fixed target centers are used solely to score these two stationary clips.',
    'inputs': ['media/frc-fuel-dense-count-evaluation.json', 'media/frc-project516-v1-dense-sampled-evaluation.json'],
    'limitations': 'Only 60 sampled frames per clip at one pose; no broad dense-cluster ground truth. Project516 inference is CPU-heavy and previously showed much worse stereo range agreement, so this checks count filtering only and does not support its use for range.',
    'results': [],
}

for scene in ('earlier', 'corrected-settings'):
    frc_scene = next(item for item in frc_data['results'] if item['scene'] == scene)
    p516_scene = next(item for item in p516_data['results']
                      if item['scene'] == scene and item['imgsz'] == 960)
    frc_frames = {row['frame']: row['detections'] for row in frc_scene['per_frame']}
    p516_frames = {row['frame']: row['boxes'] for row in p516_scene['frames']}
    frames = sorted(frc_frames.keys() & p516_frames.keys())
    scene_result = {'scene': scene, 'matched_frames': len(frames), 'comparisons': []}
    for frc_conf in FRC_CONFIDENCES:
        frc_rows = [[box for box in frc_frames[frame]
                     if box['confidence'] >= frc_conf] for frame in frames]
        comparison = {'frc9550_confidence': frc_conf, 'frc9550_baseline': score(frc_rows),
                      'project516_gates': []}
        for p516_conf in P516_CONFIDENCES:
            p516_rows = [[box for box in p516_frames[frame]
                          if box['confidence'] >= p516_conf] for frame in frames]
            gate = {'project516_confidence': p516_conf, 'project516_alone': score(p516_rows),
                    'consensus': []}
            for threshold in CONSENSUS_IOUS:
                filtered = [
                    [box for box in frc_boxes if any(box_iou(box, other) >= threshold
                                                     for other in p516_boxes)]
                    for frc_boxes, p516_boxes in zip(frc_rows, p516_rows)
                ]
                gate['consensus'].append({'minimum_iou': threshold, 'metrics': score(filtered)})
            comparison['project516_gates'].append(gate)
        scene_result['comparisons'].append(comparison)
    output['results'].append(scene_result)

destination = ROOT / 'media/frc-fuel-detector-consensus-evaluation.json'
destination.write_text(json.dumps(output, indent=2) + '\n')
for scene in output['results']:
    print(scene['scene'], 'matched frames', scene['matched_frames'])
    for comparison in scene['comparisons']:
        print(' FRC conf', comparison['frc9550_confidence'], comparison['frc9550_baseline'])
        for gate in comparison['project516_gates']:
            print('  Project516 conf', gate['project516_confidence'],
                  'alone', gate['project516_alone'],
                  'consensus', [(item['minimum_iou'], item['metrics'])
                                for item in gate['consensus']])
print('saved', destination)
