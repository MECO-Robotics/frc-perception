#!/usr/bin/env python3
"""Check stereo range-quality gates with one saved clip held out."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'media/frc-fuel-stereo-quality-sweep.json'
OUTPUT = ROOT / 'media/frc-fuel-stereo-quality-holdout.json'
FEATURES = ('mad_px', 'size_stereo_error', 'lr_consistency_fraction', 'valid_fraction')
BAD_LIMIT_M = 0.05
MAX_FALSE_REJECT = 0.05


def classify(row, rule):
    return ((row['mad_px'] is not None and row['mad_px'] > rule['mad_px']) or
            (row['size_stereo_error'] is not None and
             row['size_stereo_error'] > rule['size_stereo_error']))


def confusion(rows, reject):
    good = [row for row in rows if row['abs_error_m'] is not None and row['abs_error_m'] <= BAD_LIMIT_M]
    bad = [row for row in rows if row['abs_error_m'] is not None and row['abs_error_m'] > BAD_LIMIT_M]
    good_rejected = sum(reject(row) for row in good)
    bad_rejected = sum(reject(row) for row in bad)
    return {
        'good_ranges': len(good), 'bad_ranges': len(bad),
        'good_rejected': good_rejected, 'bad_rejected': bad_rejected,
        'false_reject_rate': round(good_rejected / len(good), 4) if good else None,
        'bad_range_recall': round(bad_rejected / len(bad), 4) if bad else None,
    }


def threshold_values(rows, feature):
    values = sorted({float(row[feature]) for row in rows if row.get(feature) is not None})
    if not values:
        return [math.inf]
    return values


def select_rule(train):
    best = None
    best_rank = None
    for mad_threshold in threshold_values(train, 'mad_px'):
        for size_threshold in threshold_values(train, 'size_stereo_error'):
            rule = {'mad_px': mad_threshold, 'size_stereo_error': size_threshold}
            stats = confusion(train, lambda row: classify(row, rule))
            fpr = stats['false_reject_rate']
            recall = stats['bad_range_recall']
            if fpr is None or fpr > MAX_FALSE_REJECT:
                continue
            rank = (recall, -fpr, -mad_threshold, -size_threshold)
            if best_rank is None or rank > best_rank:
                best, best_rank = (rule, stats), rank
    return {'rule': best[0], 'train_metrics': best[1]} if best else None


source = json.loads(SOURCE.read_text())
rows = [row for row in source['results']
        if row['method'] == 'vertical_band_24' and row['occlusion_fraction'] > 0]
results = []
for held_out in ('earlier', 'corrected-settings'):
    test = [row for row in rows if row['scene'] == held_out]
    train = [row for row in rows if row['scene'] != held_out]
    selected = select_rule(train)
    previous_rule = {'mad_px': 1.25, 'size_stereo_error': math.inf}
    combined_rule = {'mad_px': 1.25, 'size_stereo_error': 0.06}
    fixed_lr_rule = None
    baseline_test = confusion(test, lambda row: False)
    selected_test = confusion(test, lambda row: classify(row, selected['rule'])) if selected else None
    rule_test = {
        'mad_gt_1.25_or_size_error_gt_0.06': confusion(
            test, lambda row: classify(row, combined_rule)),
        'mad_gt_1.25_only': confusion(test, lambda row: classify(row, previous_rule)),
        'size_error_gt_0.06_only': confusion(
            test, lambda row: row['size_stereo_error'] is not None and row['size_stereo_error'] > 0.06),
        'size_error_gt_0.04_only': confusion(
            test, lambda row: row['size_stereo_error'] is not None and row['size_stereo_error'] > 0.04),
        'lr_below_0.58_only': confusion(
            test, lambda row: row['lr_consistency_fraction'] is not None and row['lr_consistency_fraction'] < 0.58),
        'mad_gt_1.25_or_lr_below_0.58': confusion(
            test, lambda row: row['mad_px'] > 1.25 or
            (row['lr_consistency_fraction'] is not None and row['lr_consistency_fraction'] < 0.58)),
    }
    by_mask = {}
    for fraction in (0.25, 0.5, 0.75):
        subset = [row for row in test if row['occlusion_fraction'] == fraction]
        by_mask[str(fraction)] = {
            'n': len(subset),
            'bad_ranges_gt_5cm': sum(row['abs_error_m'] > BAD_LIMIT_M for row in subset),
            'previous_rule_rejects': sum(classify(row, previous_rule) for row in subset),
            'selected_rule_rejects': sum(classify(row, selected['rule']) for row in subset) if selected else None,
        }
    results.append({
        'held_out_scene': held_out,
        'train_scene': train[0]['scene'] if train else None,
        'train_selected_rule': selected,
        'held_out_no_gate': baseline_test,
        'held_out_selected_rule': selected_test,
        'held_out_fixed_rules': rule_test,
        'held_out_by_mask_fraction': by_mask,
    })

report = {
    'method': ('Use the saved active vertical-band-24 stereo-quality measurements. '
               'A range is labeled bad when its synthetic-mask estimate differs by '
               'more than 5 cm from that same frame’s clean estimate. Select an '
               'OR gate using one clip with at most 5% false rejection, then score '
               'the other clip without retuning.'),
    'limitations': ('Same-scene clean stereo is only a repeatability reference, '
                    'not measured distance truth. Synthetic masks do not reproduce '
                    'physical occlusion, motion, or detector misses. The threshold '
                    'search is exploratory and based on only 15 timestamps per clip.'),
    'results': results,
}
OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
for row in results:
    print(row['held_out_scene'], 'train', row['train_selected_rule'],
          'test', row['held_out_selected_rule'],
          'fixed', row['held_out_fixed_rules'])
print('saved', OUTPUT)
