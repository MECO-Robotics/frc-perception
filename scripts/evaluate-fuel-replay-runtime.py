#!/usr/bin/env python3
"""Sample an already-running local Scout replay without touching a camera."""
import argparse
import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8080')
parser.add_argument('--duration', type=float, default=20.0)
parser.add_argument('--interval', type=float, default=0.1)
parser.add_argument('--output', default='media/frc-fuel-replay-runtime-check.json')
args = parser.parse_args()


def get(path):
    with urllib.request.urlopen(args.url + path, timeout=2) as response:
        return json.load(response)


def summarize(values):
    if not values:
        return {'n': 0}
    median = statistics.median(values)
    return {'n': len(values), 'median': round(median, 4),
            'mad': round(statistics.median(abs(v-median) for v in values), 4),
            'min': round(min(values), 4), 'max': round(max(values), 4)}


started_at = datetime.now(timezone.utc).isoformat()
stats_before = get('/api/stats')
samples = []
start = time.monotonic()
while time.monotonic() - start < args.duration:
    request_started = time.monotonic()
    detection = get('/api/detections')
    boxes = detection.get('boxes', [])
    samples.append({'elapsed_s': round(request_started-start, 3),
                    'status': detection.get('status'),
                    'age_s': detection.get('age_s'),
                    'boxes': boxes})
    remaining = args.interval - (time.monotonic()-request_started)
    if remaining > 0:
        time.sleep(remaining)
stats_after = get('/api/stats')
counts = [len(row['boxes']) for row in samples]
per_box = []
for i in range(max(counts, default=0)):
    selected = [sorted(row['boxes'], key=lambda b:b['x'])[i]
                for row in samples if len(row['boxes']) > i]
    per_box.append({'detections': len(selected),
                    'confidence': summarize([b['confidence'] for b in selected if b.get('confidence') is not None]),
                    'distance_m': summarize([b['distance_m'] for b in selected if b.get('distance_m') is not None]),
                    'disparity_px': summarize([b['disparity_px'] for b in selected if b.get('disparity_px') is not None])})
result = {'started_at_utc': started_at, 'duration_s': round(time.monotonic()-start, 2),
          'interval_s': args.interval, 'sample_count': len(samples),
          'method': 'Poll /api/detections from the already-running replay; sort boxes by horizontal image position for per-target summaries.',
          'stats_before': stats_before, 'stats_after': stats_after,
          'exactly_two_fraction': round(sum(n == 2 for n in counts)/len(counts), 5) if counts else None,
          'count_histogram': {str(n): counts.count(n) for n in sorted(set(counts))},
          'per_box_left_to_right': per_box, 'samples': samples}
path = ROOT / args.output
path.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'samples'}, indent=2))
print('saved', path)
