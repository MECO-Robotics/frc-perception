#!/usr/bin/env python3
import json, sys, time, urllib.request

url = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8080/api/stats'
seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 60
samples = []
deadline = time.time() + seconds
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            sample = json.load(response)
        samples.append(sample)
        print(json.dumps(sample, sort_keys=True), flush=True)
    except Exception as exc:
        print(json.dumps({'error': str(exc)}), flush=True)
    time.sleep(1)
valid = [x for x in samples if 'capture_fps' in x and x.get('status') == 'running']
if not valid:
    raise SystemExit('no running samples collected')
for key in ('capture_fps','inference_fps','display_fps','latency_ms','inference_latency_ms'):
    values = [float(x[key]) for x in valid if key in x]
    if values: print(f'{key}: avg={sum(values)/len(values):.2f} min={min(values):.2f} max={max(values):.2f}')
print(f'valid_samples: {len(valid)}')
