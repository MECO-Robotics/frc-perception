#!/usr/bin/env python3
"""Score a luma-triggered gamma correction across both complete saved FUEL clips."""
import json, math, os, statistics, time, sys
from pathlib import Path
import cv2, numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scout.detectors import TensorRTDetector
VIDEOS=(('earlier','media/frc-2026-live-two-fuel-60fps-cfr.mp4'),('corrected-settings','media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'))
TARGETS=((808.,589.),(902.,591.));STRIDE=5;NMS=.15;THRESHOLDS=(15.,20.,25.,30.,35.);CONFIDENCES=(.02,.05,.075,.10)
GAMMA=.65;LUT=np.clip((np.arange(256,dtype=np.float32)/255.)**GAMMA*255,0,255).astype(np.uint8)
def score(rawboxes,threshold):
 hits=[None,None];extras=0
 for b in sorted((b for b in rawboxes if b['confidence']>=threshold),key=lambda b:-b['confidence']):
  x=(b['x1']+b['x2'])/2;y=(b['y1']+b['y2'])/2;err,i=min((math.hypot(x-a,y-c),i) for i,(a,c) in enumerate(TARGETS))
  if err<=60 and hits[i] is None:hits[i]=(b,err)
  else:extras+=1
 return hits,extras
def metrics(records,policy,threshold):
 tp=fp=fn=paired=exact=0;errors=[[],[]];transformed=0
 for row in records:
  key='gamma' if policy=='gamma' or (policy.startswith('adaptive_') and row['mean_luma']<float(policy.split('_')[1])) else 'raw'
  transformed+=key=='gamma';hits,extra=score(row[key],threshold);n=sum(x is not None for x in hits)
  tp+=n;fn+=2-n;fp+=extra;paired+=n==2;exact+=n==2 and extra==0
  for i,item in enumerate(hits):
   if item:errors[i].append(item[1])
 return {'sampled_frames':len(records),'gamma_frames':transformed,'paired_frames':paired,'exact_two_frames':exact,'tp':tp,'fp':fp,'fn':fn,'precision':round(tp/(tp+fp),5) if tp+fp else None,'recall':round(tp/(tp+fn),5) if tp+fn else None,'target_median_center_error_px':[round(statistics.median(a),2) if a else None for a in errors]}
os.environ.update(SCOUT_CLASS_NAMES='["ball"]',SCOUT_MAX_BOX_ASPECT_RATIO='2.0',SCOUT_NMS_IOU=str(NMS))
det=TensorRTDetector(str(ROOT/'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'),confidence=.02,iou=NMS)
out={'method':'On every fifth frame in both full saved clips, run the same FRC YOLO26n 1280x960 FP16 engine on raw and gamma .65 left images. Compare fixed raw, fixed gamma, and luma-mean-triggered gamma policies at confidence .02/.05/.075/.10; sweep luma thresholds using original-frame whole-left-image mean Y. NMS is fixed at .15.','limitations':'One pair of fixed indoor scenes only; luma threshold may not transfer to other cameras, exposure, or FRC field backgrounds. Gamma does not model sensor gain/noise or motion blur. No range truth is available.','gamma':GAMMA,'nms_iou':NMS,'frame_stride':STRIDE,'luma_gate':'apply gamma when mean grayscale luma of original left image is below threshold','results':[]}
try:
 for scene,video in VIDEOS:
  cap=cv2.VideoCapture(str(ROOT/video));rows=[];idx=0;started=time.perf_counter()
  while True:
   ok,frame=cap.read()
   if not ok:break
   if idx%STRIDE==0:
    left=frame[:,:1600];gray=cv2.cvtColor(left,cv2.COLOR_BGR2GRAY);mean_luma=float(gray.mean())
    raw,raw_ms=det.infer(left);bright=cv2.LUT(left,LUT);gamma,gamma_ms=det.infer(bright)
    def pack(bs):return [{'x1':b.x1,'y1':b.y1,'x2':b.x2,'y2':b.y2,'confidence':float(b.confidence)} for b in bs]
    rows.append({'frame':idx,'time_s':idx/59.900166,'mean_luma':mean_luma,'raw':pack(raw),'gamma':pack(gamma),'raw_ms':raw_ms,'gamma_ms':gamma_ms})
   idx+=1
  cap.release();policies=['raw','gamma']+[f'adaptive_{x:g}' for x in THRESHOLDS];config=[]
  for policy in policies:
   for conf in CONFIDENCES:config.append({'policy':policy,'confidence':conf,**metrics(rows,policy,conf)})
  out['results'].append({'scene':scene,'video':video,'decoded_frames':idx,'elapsed_s':round(time.perf_counter()-started,2),'luma_mean_median':round(statistics.median(r['mean_luma'] for r in rows),2),'luma_p05_p95':[round(float(np.percentile([r['mean_luma'] for r in rows],5)),2),round(float(np.percentile([r['mean_luma'] for r in rows],95)),2)],'median_detector_ms_raw':round(statistics.median(r['raw_ms'] for r in rows),2),'median_detector_ms_gamma':round(statistics.median(r['gamma_ms'] for r in rows),2),'configuration_metrics':config,'per_frame':rows})
  print(scene,[(r['policy'],r['confidence'],r['paired_frames'],r['exact_two_frames'],r['fp'],r['fn'],r['gamma_frames']) for r in config if r['confidence'] in (.05,.075)])
finally:det.close()
p=ROOT/'media/frc-fuel-adaptive-luma-preprocessing.json';p.write_text(json.dumps(out,indent=2)+'\n');print('saved',p)
