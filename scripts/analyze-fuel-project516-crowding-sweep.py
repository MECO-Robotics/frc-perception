#!/usr/bin/env python3
"""Stratify cached detector/NMS sweeps by GT box overlap and object count."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    'project516_eval', ROOT / 'scripts/evaluate-fuel-project516-public-dataset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

source = json.loads((ROOT / 'media/frc-fuel-project516-nms-sweep.json').read_text())
thresholds = ('0.05', '0.075', '0.1', '0.15', '0.2', '0.3')
nms_values = ('0.05', '0.1', '0.15', '0.2', '0.25', '0.45')
rows = []
missing = []
for image in source['per_image']:
    image_id = image['image_id']
    try:
        labels = module.labels_for(image_id, image['width'], image['height'])
    except Exception as exc:
        if getattr(exc, 'code', None) == 404:
            missing.append(image_id)
            continue
        raise
    overlaps = [module.iou(labels[a], labels[b])
                for a in range(len(labels)) for b in range(a + 1, len(labels))]
    max_overlap = max(overlaps, default=0.0)
    row = dict(image_id=image_id, object_count=len(labels),
               gt_pair_count=len(overlaps), gt_max_pair_iou=round(max_overlap, 5),
               overlap_proxy=max_overlap >= 0.05,
               configs={})
    for nms in nms_values:
        if nms not in image['by_iou']:
            continue
        for confidence in thresholds:
            metrics = image['by_iou'][nms]['thresholds'][confidence]
            row['configs'][f'nms{nms}_conf{confidence}'] = metrics
    rows.append(row)


def summarize_values(values):
    if not values:
        return {'images': 0}
    tp = sum(metrics['tp'] for metrics, _ in values)
    fp = sum(metrics['fp'] for metrics, _ in values)
    fn = sum(metrics['fn'] for metrics, _ in values)
    errors = [abs(metrics['predicted_count'] - count) for metrics, count in values]
    exact = sum(error == 0 for error in errors)
    return {
        'images': len(values),
        'exact_count_images': exact,
        'exact_count_fraction': round(exact / len(values), 5),
        'mean_absolute_count_error': round(sum(errors) / len(errors), 4),
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(tp / (tp + fp), 5) if tp + fp else None,
        'recall': round(tp / (tp + fn), 5) if tp + fn else None,
    }


def summarize(selected, key):
    return summarize_values([(row['configs'][key], row['object_count'])
                             for row in selected if key in row['configs']])


groups = {
    'all': rows,
    'overlap_proxy_iou_ge_0p05': [row for row in rows if row['overlap_proxy']],
    'no_box_overlap_proxy': [row for row in rows if not row['overlap_proxy']],
    'three_or_more_objects': [row for row in rows if row['object_count'] >= 3],
    'one_or_two_objects': [row for row in rows if row['object_count'] <= 2],
}
summary = {
    name: {key: summarize(group, key)
           for nms in nms_values for confidence in thresholds
           for key in (f'nms{nms}_conf{confidence}',)}
    for name, group in groups.items()
}
adaptive = {}
for name, group in groups.items():
    adaptive[name] = {}
    for nms in nms_values:
        for trigger in (2, 3, 4, 5):
            for high in ('0.1', '0.15', '0.2', '0.3'):
                chosen = []
                low_key = f'nms{nms}_conf0.05'
                high_key = f'nms{nms}_conf{high}'
                for row in group:
                    if low_key not in row['configs'] or high_key not in row['configs']:
                        continue
                    low = row['configs'][low_key]
                    selected_metrics = (row['configs'][high_key]
                                        if low['predicted_count'] >= trigger else low)
                    chosen.append((selected_metrics, row['object_count']))
                adaptive[name][f'nms{nms}_trigger{trigger}_high{high}'] = summarize_values(chosen)
output = {
    'method': ('Use cached per-image FRC 9550 YOLO26n predictions from the complete '
               'Project516 NMS/confidence sweep. Fetch only YOLO label text to mark '
               'images whose labeled FUEL boxes overlap at IoU >= 0.05; stratify exact '
               'cluster-count and detection metrics by overlap and target count. Also '
               'simulate a count-adaptive confidence cutoff: if at least N boxes survive '
               'at 0.05, rescore that image using the higher cutoff.'),
    'source': 'https://huggingface.co/datasets/project516/rebuilt-fuel-dataset',
    'labelled_images': len(rows),
    'missing_label_ids': missing,
    'overlap_proxy_definition': 'At least one pair of labeled FUEL boxes has IoU >= 0.05. This is a box-overlap/crowding proxy, not a verified occlusion label.',
    'image_count_and_overlap': {
        'object_count_histogram': {str(count): sum(row['object_count'] == count for row in rows)
                                   for count in sorted({row['object_count'] for row in rows})},
        'overlap_proxy_images': sum(row['overlap_proxy'] for row in rows),
        'non_overlap_images': sum(not row['overlap_proxy'] for row in rows),
    },
    'groups': {name: {'n': len(group), 'metrics': summary[name]}
               for name, group in groups.items()},
    'adaptive_confidence_policies': adaptive,
    'per_image': rows,
    'limitations': ('This is a stratified analysis of the existing unmanifested Project516 '
                    'image set, not a held-out benchmark. Box overlap is only a crowding '
                    'proxy. Monocular labels cannot validate range or 3D cluster positions.'),
}
path = ROOT / 'media/frc-fuel-project516-crowding-stratification.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps({'labelled_images': len(rows),
                  'overlap_proxy_images': output['image_count_and_overlap']['overlap_proxy_images'],
                  'count_histogram': output['image_count_and_overlap']['object_count_histogram'],
                  'current_nms015_conf005': {
                      name: metrics.get('nms0.15_conf0.05')
                      for name, metrics in summary.items()},
                  'adaptive_nms015_trigger3_high020': {
                      name: metrics.get('nms0.15_trigger3_high0.2')
                      for name, metrics in adaptive.items()},
                  'saved': str(path)}, indent=2))
