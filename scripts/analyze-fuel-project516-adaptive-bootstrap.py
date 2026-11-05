#!/usr/bin/env python3
"""Paired image bootstrap for adaptive-confidence policy comparisons."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = json.loads((ROOT / 'media/frc-fuel-project516-crowding-stratification.json').read_text())
rows = source['per_image']
POLICIES = {
    'fixed_nms015_conf005': ('nms0.15_conf0.05', None, None),
    'current_nms015_trigger3_high020': ('nms0.15_conf0.05', 3, 'nms0.15_conf0.2'),
    'aggressive_nms010_trigger2_high030': ('nms0.1_conf0.05', 2, 'nms0.1_conf0.3'),
}


def selected_metrics(row, policy):
    low_key, trigger, high_key = POLICIES[policy]
    low = row['configs'][low_key]
    if trigger is not None and low['predicted_count'] >= trigger:
        return row['configs'][high_key]
    return low


def vector(row, policy):
    metrics = selected_metrics(row, policy)
    return {
        'exact': int(metrics['predicted_count'] == row['object_count']),
        'count_error': abs(metrics['predicted_count'] - row['object_count']),
        'tp': metrics['tp'], 'fp': metrics['fp'], 'fn': metrics['fn'],
    }


def summarize(indices, policy):
    vals = [vector(rows[index], policy) for index in indices]
    tp = sum(item['tp'] for item in vals)
    fp = sum(item['fp'] for item in vals)
    fn = sum(item['fn'] for item in vals)
    n = len(vals)
    return {
        'exact_count_fraction': sum(item['exact'] for item in vals) / n,
        'mean_absolute_count_error': sum(item['count_error'] for item in vals) / n,
        'precision': tp / (tp + fp) if tp + fp else 0.0,
        'recall': tp / (tp + fn) if tp + fn else 0.0,
    }


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


comparisons = (
    ('current_nms015_trigger3_high020', 'fixed_nms015_conf005'),
    ('aggressive_nms010_trigger2_high030', 'current_nms015_trigger3_high020'),
)
rng = random.Random(20261102)
replicates = 20000
indices = list(range(len(rows)))
observed = {}
bootstrap = {}
for candidate, reference in comparisons:
    c = summarize(indices, candidate)
    r = summarize(indices, reference)
    differences = {metric: [] for metric in c}
    for _ in range(replicates):
        sample = [rng.choice(indices) for _ in indices]
        cs = summarize(sample, candidate)
        rs = summarize(sample, reference)
        for metric in differences:
            differences[metric].append(cs[metric] - rs[metric])
    observed[f'{candidate}_minus_{reference}'] = {
        metric: round(c[metric] - r[metric], 5) for metric in c}
    bootstrap[f'{candidate}_minus_{reference}'] = {
        metric: {
            'ci95': [round(percentile(values, .025), 5),
                     round(percentile(values, .975), 5)],
            'probability_positive': round(sum(value > 0 for value in values) / replicates, 5),
        }
        for metric, values in differences.items()
    }

output = {
    'method': 'Paired nonparametric bootstrap over Project516 images; resample images with replacement and compare each confidence policy on identical samples.',
    'images': len(rows), 'replicates': replicates, 'seed': 20261102,
    'policies': POLICIES,
    'point_estimates': {name: summarize(indices, name) for name in POLICIES},
    'observed_paired_differences': observed,
    'bootstrap_differences': bootstrap,
    'limitations': 'The dataset is a small unmanifested image set with only three box-overlap proxies. Bootstrap intervals quantify sampling uncertainty within these images; they do not establish field generalization, temporal count, or stereo accuracy.',
}
path = ROOT / 'media/frc-fuel-project516-adaptive-bootstrap.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps({'point_estimates': output['point_estimates'],
                  'observed_paired_differences': observed,
                  'bootstrap_differences': bootstrap,
                  'saved': str(path)}, indent=2))
