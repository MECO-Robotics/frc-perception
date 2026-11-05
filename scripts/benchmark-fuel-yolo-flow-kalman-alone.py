#!/usr/bin/env python3
"""Isolated GPU replay rate for 20 Hz YOLO anchors plus live-scale LK/Kalman."""
import json,statistics,sys,time,os
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ['SCOUT_CLASS_NAMES']='["ball"]';os.environ['SCOUT_MAX_BOX_ASPECT_RATIO']='2.0'
from scout.detectors import TensorRTDetector
from scout.tracking import FuelTracker
VIDEO=ROOT/'media/frc-fuel-live-heavy-occlusion-20261103.mp4';OUT=ROOT/'media/frc-fuel-yolo-flow-kalman-alone-20261103.json';FPS=59.966685;STEP=3;REF=json.load(open(ROOT/'media/frc-fuel-live-heavy-occlusion-20261103-replay.json'));anchor_frames={r['frame'] for r in REF['per_sample']}
cap=cv2.VideoCapture(str(VIDEO));det=TensorRTDetector('models/rebuilt_fuel_yolo26n_1280x960_fp16.engine',confidence=.05,iou=.15);tracker=FuelTracker();flow=[];updates=[];detlat=[];hist={};counts=[];i=0;started=time.perf_counter()
try:
 ok,frame=cap.read()
 if ok:det.infer(cv2.resize(frame[:,:1600],(320,240),interpolation=cv2.INTER_AREA))
 i=0
 while ok:
  low=cv2.resize(frame[:,:1600],(320,240),interpolation=cv2.INTER_AREA);now=i/FPS;t=time.perf_counter();tracker.advance(low,now,i);flow.append((time.perf_counter()-t)*1000)
  if i in anchor_frames:
   boxes,ms=det.infer(low);detlat.append(ms);sel=[]
   for b in boxes:
    cx=(b.x1+b.x2)/2;cy=(b.y1+b.y2)/2
    if b.confidence>=.05 and 130<=cx<=210 and 95<=cy<=145:sel.append(b)
   sel=sorted(sel,key=lambda x:x.confidence,reverse=True)[:2];hist[len(sel)]=hist.get(len(sel),0)+1;t=time.perf_counter();tracker.update(sel,now,i);updates.append((time.perf_counter()-t)*1000)
  counts.append(len(tracker.snapshot(now)));i+=1;ok,frame=cap.read()
finally:det.close();cap.release()
frames=len(counts);wall=time.perf_counter()-started;fm=statistics.median(flow);um=statistics.median(updates);dm=statistics.median(detlat);budget=fm+um/STEP+dm/STEP
out={'video':str(VIDEO.relative_to(ROOT)),'frames':frames,'fps':FPS,'tracker_input_resolution':[320,240],'engine_input_resolution':'1280x960','anchors':len(detlat),'anchor_hz':round(len(detlat)/(frames/FPS),2),'detection_count_histogram':{str(k):v for k,v in sorted(hist.items())},'tracked_frame_histogram':{str(k):counts.count(k) for k in sorted(set(counts))},'reported_position_rate_hz':FPS,'tensorrt_anchor_latency_ms':{'median':round(dm,3),'p95':round(float(np.percentile(detlat,95)),3)},'lk_advance_latency_ms':{'median':round(fm,3),'p95':round(float(np.percentile(flow,95)),3)},'kalman_update_latency_ms':{'median':round(um,3),'p95':round(float(np.percentile(updates,95)),3)},'serial_stage_budget_ms_per_frame':round(budget,3),'serial_stage_budget_hz':round(1000/budget,1),'replay_wall_s_including_mp4_decode':round(wall,2),'replay_wall_fps_including_mp4_decode':round(frames/wall,2),'limitations':['Static two-ball clip; no stereo, NetworkTables, GStreamer camera callbacks, or live scheduling contention.','Position labels are the detector boxes and tracker outputs; no independent object-center labels are available.']}
OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));print('saved',OUT)
