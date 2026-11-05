#!/usr/bin/env python3
"""Replay both saved FUEL clips with configured NMS on YOLO26 E2E boxes."""
import json,math,os,statistics,sys,time
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scout.detectors import TensorRTDetector
VIDEOS=(('earlier','media/frc-2026-live-two-fuel-60fps-cfr.mp4'),('corrected-settings','media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'))
TARGETS=((808,589),(902,591));MATCH=60.0;SWEEP=(.02,.03,.05,.075,.10)
def score(frames,confidence):
 tp=fp=fn=exact=0;out=[]
 for row in frames:
  hits=[None,None];extras=0
  for b in sorted((b for b in row if b['confidence']>=confidence),key=lambda b:b['confidence'],reverse=True):
   cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
   d,i=min((math.hypot(cx-x,cy-y),i) for i,(x,y) in enumerate(TARGETS))
   if d<=MATCH and hits[i] is None:hits[i]=b
   else:extras+=1
  n=sum(b is not None for b in hits);tp+=n;fn+=2-n;fp+=extras;exact+=n==2 and extras==0
  out.append({'target_count':n,'extra_boxes':extras})
 return {'confidence':confidence,'tp':tp,'fp':fp,'fn':fn,'exact_two_frames':exact,'precision':round(tp/(tp+fp),6) if tp+fp else 0,'recall':round(tp/(tp+fn),6) if tp+fn else 0,'per_frame':out}

os.environ['SCOUT_CLASS_NAMES']='["ball"]';os.environ['SCOUT_MAX_BOX_ASPECT_RATIO']='2.0'
detector=TensorRTDetector(str(ROOT/'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'),confidence=.005,iou=.45)
result={'method':'Run FRC YOLO26n 1280x960 FP16 on every left-view frame from each saved 3200x1200 stereo clip. Apply class-wise OpenCV NMS at configured IoU 0.45 to the end-to-end xyxy/score/class output, then rescore confidence .02-.10 against the two fixed target centers.','configured_nms_iou':.45,'limitations':'Stationary two-ball indoor recordings only. The fixed target windows do not establish performance on close overlapping balls, field background, motion, or real occlusion.','results':[]}
try:
 for scene,video in VIDEOS:
  cap=cv2.VideoCapture(str(ROOT/video));rows=[];lat=[];started=time.perf_counter()
  while True:
   ok,frame=cap.read()
   if not ok:break
   boxes,ms=detector.infer(frame[:,:1600]);lat.append(ms)
   rows.append([{'x1':b.x1,'y1':b.y1,'x2':b.x2,'y2':b.y2,'confidence':b.confidence} for b in boxes])
  cap.release();entry={'scene':scene,'video':video,'frames':len(rows),'elapsed_s':round(time.perf_counter()-started,2),'median_detector_ms':round(statistics.median(lat),2),'p95_detector_ms':round(sorted(lat)[int(.95*(len(lat)-1))],2),'confidence_sweep':[],'detections':rows}
  for conf in SWEEP:
   metrics=score(rows,conf);entry['confidence_sweep'].append({k:v for k,v in metrics.items() if k!='per_frame'});print(scene,entry['confidence_sweep'][-1])
  result['results'].append(entry)
finally:detector.close()
p=ROOT/'media/frc-fuel-endtoend-nms-replay.json';p.write_text(json.dumps(result,indent=2)+'\n');print('saved',p)
