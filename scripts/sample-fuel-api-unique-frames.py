#!/usr/bin/env python3
"""Record unique inference outputs from Scout's running detection API."""
import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


def fetch(base, path):
    with urlopen(base.rstrip('/') + path, timeout=3) as response:
        return json.load(response)


def median(values):
    return statistics.median(values) if values else None


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8080')
parser.add_argument('--duration', type=float, default=30.0)
parser.add_argument('--interval', type=float, default=0.04)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.duration <= 0 or args.interval <= 0:
    parser.error('--duration and --interval must be positive')

before = fetch(args.url, '/api/stats')
started = time.monotonic()
deadline = started + args.duration
seen = set()
records = []
errors = 0
polls = 0
while time.monotonic() < deadline:
    try:
        item = fetch(args.url, '/api/detections')
        polls += 1
        timestamp = item.get('rtp_timestamp')
        if item.get('status') == 'ready' and timestamp not in seen:
            seen.add(timestamp)
            current_stats = fetch(args.url, '/api/stats')
            boxes = sorted(item.get('boxes') or [],
                           key=lambda box: box.get('distance_m') or -1,
                           reverse=True)
            records.append({
                'rtp_timestamp': timestamp,
                'source_loop': current_stats.get('loops'),
                'age_s': item.get('age_s'),
                'count': len(boxes),
                'ranges_m': [box.get('distance_m') for box in boxes],
                'disparities_px': [box.get('disparity_px') for box in boxes],
                'centers_normalized': [
                    [box['x'] + box['w'] / 2, box['y'] + box['h'] / 2]
                    for box in boxes],
                'confidences': [box.get('confidence') for box in boxes],
            })
    except Exception:
        errors += 1
    time.sleep(args.interval)
elapsed = time.monotonic() - started
after = fetch(args.url, '/api/stats')
count_hist = {}
for record in records:
    key = str(record['count'])
    count_hist[key] = count_hist.get(key, 0) + 1
range_summaries = []
for index in range(2):
    exact_values = [record['ranges_m'][index] for record in records
              if record['count'] == 2 and len(record['ranges_m']) == 2
              and record['ranges_m'][index] is not None]
    disparity_values = [record['disparities_px'][index] for record in records
                        if record['count'] == 2 and len(record['disparities_px']) == 2
                        and record['disparities_px'][index] is not None]
    baseline = before.get('stereo_baseline_m')
    focal = before.get('stereo_focal_px')
    disparity_scale = before.get('stereo_disparity_scale', 1.0)
    reconstructed = [baseline * focal / (value * disparity_scale)
                     for value in disparity_values if baseline and focal and value]
    center = median(exact_values)
    reconstructed_center = median(reconstructed)
    range_summaries.append({
        'n_unique_two_box_frames': len(exact_values),
        'median_m': center,
        'mad_m': median([abs(value - center) for value in exact_values]) if exact_values else None,
        'p95_abs_deviation_m': (sorted(abs(value - center) for value in exact_values)
                                [int(.95 * (len(exact_values) - 1))] if exact_values else None),
        'disparity_reconstructed_range_m': {
            'n': len(reconstructed),
            'median_m': reconstructed_center,
            'mad_m': median([abs(value - reconstructed_center) for value in reconstructed])
                     if reconstructed else None,
            'p95_abs_deviation_m': (sorted(abs(value - reconstructed_center) for value in reconstructed)
                                    [int(.95 * (len(reconstructed) - 1))]
                                    if reconstructed else None),
        },
    })
output = {
    'sampled_at_utc': datetime.now(timezone.utc).isoformat(),
    'method': 'Unique ready inference outputs deduplicated by RTP timestamp; range summaries include rounded API distance and a Z reconstruction from API disparity rounded to 0.01 px.',
    'elapsed_s': round(elapsed, 2),
    'polls': polls,
    'unique_ready_frames': len(records),
    'poll_errors': errors,
    'detection_count_histogram_unique_frames': count_hist,
    'range_by_near_to_far_box_order_m': range_summaries,
    'stats_before': before,
    'stats_after': after,
    'unique_frames': records,
    'limitations': 'This is one fixed, stationary two-ball MP4 replay. It measures repeatability/count on unique inference outputs, not absolute range accuracy, motion response, or clutter false positives. API distances are rounded to centimeters and disparities to 0.01 px; disparity-derived range reduces distance quantization but is not full precision.',
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps({key: output[key] for key in (
    'elapsed_s', 'polls', 'unique_ready_frames', 'poll_errors',
    'detection_count_histogram_unique_frames',
    'range_by_near_to_far_box_order_m')}, indent=2))
print('saved', args.output)
