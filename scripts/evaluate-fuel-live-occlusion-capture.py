#!/usr/bin/env python3
"""Replay the 2026-11-03 live USB occlusion capture through TensorRT YOLO."""
import json, math, os, statistics, sys, time
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '2.0'
from scout.detectors import TensorRTDetector
VIDEO = ROOT / 'media/frc-fuel-live-occlusion-auto-20261103.mp4'
ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
OUT = ROOT / 'media/frc-fuel-live-occlusion-auto-20261103-replay.json'
TARGETS = ((808, 589), (902, 591))  # approximate centers on left stereo image, from captured preview
STEP = 3
SCORE_THRESHOLDS = (.05, .10, .20)

def summarize(boxes, threshold):
    chosen = [b for b in boxes if b['confidence'] >= threshold]
    hits = [None, None]
    extras = []
    for b in sorted(chosen, key=lambda b: b['confidence'], reverse=True):
        cx, cy = (b['x1'] + b['x2']) / 2, (b['y1'] + b['y2']) / 2
        d, i = min((math.hypot(cx-x, cy-y), i) for i, (x,y) in enumerate(TARGETS))
        if d <= 65 and hits[i] is None:
            hits[i] = {'distance_px': round(d, 1), **b}
        else:
            extras.append(b)
    return {'target_hits': hits, 'count_near_targets': sum(x is not None for x in hits),
            'extra_detections': len(extras)}

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened(): raise RuntimeError(f'cannot open {VIDEO}')
fps = cap.get(cv2.CAP_PROP_FPS); frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
detector = TensorRTDetector(str(ENGINE), confidence=.005, iou=.15)
rows=[]; latencies=[]; started=time.perf_counter()
try:
    # Decode sequentially; run one warm-up inference before timed samples.
    ok, first = cap.read()
    if ok: detector.infer(first[:, :width//2])
    idx = 1
    ok, frame = cap.read()
    while ok:
        if idx % STEP == 0:
            left = frame[:, :width//2]
            boxes, elapsed = detector.infer(left); latencies.append(elapsed)
            serial = [{'x1':b.x1,'y1':b.y1,'x2':b.x2,'y2':b.y2,'class_id':b.class_id,'confidence':round(b.confidence,5)} for b in boxes]
            row={'frame':idx,'time_s':round(idx/fps,4),'inference_ms':round(elapsed,2),
                 'profile':getattr(detector,'last_profile',{}),'detections':serial,
                 'at_thresholds':{str(t):summarize(serial,t) for t in SCORE_THRESHOLDS}}
            rows.append(row)
        ok, frame = cap.read()
        idx += 1
finally:
    detector.close(); cap.release()
valid=[r for r in rows if r['detections']]
summary={str(t):{'samples':len(rows),'frames_with_two_targets':sum(r['at_thresholds'][str(t)]['count_near_targets']==2 for r in rows),
                 'frames_with_one_target':sum(r['at_thresholds'][str(t)]['count_near_targets']==1 for r in rows),
                 'frames_with_zero_targets':sum(r['at_thresholds'][str(t)]['count_near_targets']==0 for r in rows),
                 'two_target_fraction':round(sum(r['at_thresholds'][str(t)]['count_near_targets']==2 for r in rows)/len(rows),4) if rows else None}
         for t in SCORE_THRESHOLDS}
result={'video':str(VIDEO.relative_to(ROOT)),'engine':str(ENGINE.relative_to(ROOT)),
        'video_fps':fps,'video_frames':frame_count,'resolution':[width,height],
        'sample_step_frames':STEP,'sample_rate_hz':fps/STEP,'samples':len(rows),
        'target_centers_left_view_px':TARGETS,'target_match_radius_px':65,
        'threshold_summaries':summary,
        'latency_ms':{'median':round(statistics.median(latencies),2) if latencies else None,
                      'p95':round(sorted(latencies)[int(.95*(len(latencies)-1))],2) if latencies else None,
                      'min':round(min(latencies),2) if latencies else None,
                      'max':round(max(latencies),2) if latencies else None,
                      'throughput_hz':round(len(latencies)/(time.perf_counter()-started),2) if latencies else None},
        'per_sample':rows}
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('per_sample',)},indent=2))
print('saved',OUT)
