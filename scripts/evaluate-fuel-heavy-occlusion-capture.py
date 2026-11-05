#!/usr/bin/env python3
"""Replay the 2026-11-03 heavy physical-occlusion capture through TensorRT."""
import json, os, statistics, sys, time
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['SCOUT_CLASS_NAMES']='["ball"]';os.environ['SCOUT_MAX_BOX_ASPECT_RATIO']='2.0'
from scout.detectors import TensorRTDetector
VIDEO=ROOT/'media/frc-fuel-live-heavy-occlusion-20261103.mp4'
ENGINE=ROOT/'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
OUT=ROOT/'media/frc-fuel-live-heavy-occlusion-20261103-replay.json'
STEP=3;THRESHOLDS=(.05,.10,.20); ROI=(700,480,1000,700)

def summarize(boxes,threshold):
 x1,y1,x2,y2=ROI
 hits=[]
 for b in boxes:
  cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
  if b['confidence']>=threshold and x1<=cx<=x2 and y1<=cy<=y2:hits.append(b)
 hits.sort(key=lambda b:b['confidence'],reverse=True)
 return {'roi_detection_count':len(hits),'detections':hits}
cap=cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():raise RuntimeError(f'cannot open {VIDEO}')
fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
det=TensorRTDetector(str(ENGINE),confidence=.005,iou=.15);rows=[];lat=[];start=time.perf_counter()
try:
 ok,first=cap.read()
 if ok:det.infer(first[:,:w//2]) # warm-up excluded
 idx=1;ok,frame=cap.read()
 while ok:
  if idx%STEP==0:
   boxes,ms=det.infer(frame[:,:w//2]);lat.append(ms)
   bs=[{'x1':b.x1,'y1':b.y1,'x2':b.x2,'y2':b.y2,'confidence':round(b.confidence,5)} for b in boxes]
   rows.append({'frame':idx,'time_s':round(idx/fps,4),'inference_ms':round(ms,2),'boxes':bs,
                'at_thresholds':{str(t):summarize(bs,t) for t in THRESHOLDS}})
  ok,frame=cap.read();idx+=1
finally:det.close();cap.release()
summary={}
for t in THRESHOLDS:
 key=str(t);counts=[r['at_thresholds'][key]['roi_detection_count'] for r in rows]
 ball=[(r['at_thresholds'][key]['detections'] or [None])[0] for r in rows]
 conf=[b['confidence'] for b in ball if b]
 widths=[b['x2']-b['x1'] for b in ball if b]
 summary[key]={'samples':len(counts),'roi_zero':counts.count(0),'roi_one':counts.count(1),'roi_two_or_more':sum(x>=2 for x in counts),
  'roi_count_histogram':{str(x):counts.count(x) for x in sorted(set(counts))},
  'top_detection_confidence_median_min_p05':[round(statistics.median(conf),3),round(min(conf),3),round(sorted(conf)[int(.05*(len(conf)-1))],3)] if conf else None,
  'top_detection_box_width_px_median':[round(statistics.median(widths),1)] if widths else None}
result={'video':str(VIDEO.relative_to(ROOT)),'engine':str(ENGINE.relative_to(ROOT)),'inference':'TensorRT FP16 GPU','video_fps':fps,'frames':n,'resolution':[w,h],
 'sample_step_frames':STEP,'sample_rate_hz':fps/STEP,'samples':len(rows),'expected_physical_ball_count':2,'left_view_ball_roi_xyxy_px':ROI,
 'threshold_summary':summary,'latency_ms':{'median':round(statistics.median(lat),2),'p95':round(sorted(lat)[int(.95*(len(lat)-1))],2),'min':round(min(lat),2),'max':round(max(lat),2),'wall_throughput_hz':round(len(lat)/(time.perf_counter()-start),2)},
 'limitations':['The ROI was manually set around the stationary pair in this one camera view.','The two physical balls appear heavily merged in the left image; counting predicted boxes in the ROI evaluates detector instance separation, not tracker persistence.','Replay samples every third recorded frame, about 20 Hz.'],'per_sample':rows}
OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='per_sample'},indent=2));print('saved',OUT)
