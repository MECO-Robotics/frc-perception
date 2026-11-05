#!/usr/bin/env python3
"""Sample the running Scout detections/stats API without changing the pipeline."""
import argparse
import json
import statistics
import time
from collections import Counter
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
parser.add_argument('--interval', type=float, default=0.1)
parser.add_argument('--output', type=Path,
                    default=Path('media/frc-fuel-api-runtime-sample.json'))
args = parser.parse_args()
if args.duration <= 0 or args.interval <= 0:
    parser.error('--duration and --interval must be positive')

before = fetch(args.url, '/api/stats')
started = time.monotonic()
deadline = started + args.duration
counts = Counter()
ranges = [[], []]
ages = []
ready = 0
failed = 0
polls = 0
while time.monotonic() < deadline:
    try:
        item = fetch(args.url, '/api/detections')
        polls += 1
        ages.append(float(item.get('age_s') or 0))
        if item.get('status') == 'ready':
            ready += 1
            boxes = item.get('boxes') or []
            counts[len(boxes)] += 1
            for i, box in enumerate(sorted(boxes,
                                            key=lambda b: b.get('distance_m') or -1,
                                            reverse=True)[:2]):
                distance = box.get('distance_m')
                if distance is not None:
                    ranges[i].append(float(distance))
        else:
            failed += 1
    except Exception:
        failed += 1
    time.sleep(args.interval)
elapsed = time.monotonic() - started
after = fetch(args.url, '/api/stats')
age_p95 = sorted(ages)[int(.95 * (len(ages) - 1))] if ages else None
output = {
    'sampled_at_utc': datetime.now(timezone.utc).isoformat(),
    'url': args.url,
    'elapsed_s': round(elapsed, 2),
    'polls': polls,
    'ready_responses': ready,
    'failed_or_not_ready_responses': failed,
    'detection_count_histogram': {str(k): v for k, v in sorted(counts.items())},
    'range_by_box_order_m': [
        {'n': len(values), 'median': round(median(values), 3) if values else None,
         'mad': round(median([abs(v - median(values)) for v in values]), 3) if values else None}
        for values in ranges],
    'detection_age_p95_s': round(age_p95, 3) if age_p95 is not None else None,
    'stats_before': before,
    'stats_after': after,
    'capture_frames_delta': after.get('capture_frames', 0) - before.get('capture_frames', 0),
    'inference_frames_delta': after.get('inference_frames', 0) - before.get('inference_frames', 0),
    'limitations': 'API responses are polls, not independent inference frames. Distances are rounded by the API. Interpret this only for the source and configuration reported by /api/stats; it is not an absolute range-accuracy measurement.',
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps({key: output[key] for key in (
    'elapsed_s', 'polls', 'ready_responses', 'failed_or_not_ready_responses',
    'detection_count_histogram', 'range_by_box_order_m', 'detection_age_p95_s',
    'capture_frames_delta', 'inference_frames_delta')}, indent=2))
print('saved', args.output)
