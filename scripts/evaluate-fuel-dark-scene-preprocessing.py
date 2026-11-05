#!/usr/bin/env python3
"""Compare detector preprocessing on the dark opening frames of the saved FUEL clip."""
import json, math, os, statistics, time
from pathlib import Path
import cv2, numpy as np
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scout.detectors import TensorRTDetector
VIDEO=ROOT/'media/frc-2026-live-two-fuel-60fps-cfr.mp4'
ENGINE=ROOT/'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
TARGETS=((808,589),(902,591));FRAME_COUNT=32;NMS=.15
CONFIGS=('raw','gamma_0.5','gamma_0.55','gamma_0.6','gamma_0.65','gamma_0.7','gamma_0.75','gamma_0.8','clahe_1','clahe_2','clahe_4','equalize_y')
LUTS={name:np.clip((np.arange(256,dtype=np.float32)/255.)**gamma*255,0,255).astype(np.uint8) for name,gamma in [('gamma_0.5',.5),('gamma_0.55',.55),('gamma_0.6',.6),('gamma_0.65',.65),('gamma_0.7',.7),('gamma_0.75',.75),('gamma_0.8',.8)]}
def enhance(frame,name):
 if name=='raw':return frame
 if name in LUTS:return cv2.LUT(frame,LUTS[name])
 ycc=cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb); y=ycc[:,:,0]
 if name=='equalize_y':ycc[:,:,0]=cv2.equalizeHist(y)
 else:
  clip=float(name.split('_')[1]);ycc[:,:,0]=cv2.createCLAHE(clipLimit=clip,tileGridSize=(8,8)).apply(y)
 return cv2.cvtColor(ycc,cv2.COLOR_YCrCb2BGR)
def score(boxes,threshold):
 hits=[None,None];extras=0
 for b in sorted((b for b in boxes if b.confidence>=threshold),key=lambda b:-b.confidence):
  x=(b.x1+b.x2)/2;y=(b.y1+b.y2)/2;err,i=min((math.hypot(x-tx,y-ty),i) for i,(tx,ty) in enumerate(TARGETS))
  if err<=60 and hits[i] is None:hits[i]=(b,err)
  else:extras+=1
 return hits,extras
os.environ.update(SCOUT_CLASS_NAMES='["ball"]',SCOUT_MAX_BOX_ASPECT_RATIO='2.0',SCOUT_NMS_IOU=str(NMS))
cap=cv2.VideoCapture(str(VIDEO));samples=[]
for idx in range(FRAME_COUNT):
 cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read()
 if not ok:continue
 y=cv2.cvtColor(frame[:,:1600],cv2.COLOR_BGR2GRAY)
 samples.append({'frame':idx,'image':frame[:,:1600],'luma_mean':float(y.mean()),'luma_p90':float(np.percentile(y,90))})
cap.release()
# Engine emits candidates down to 0.02; score postprocessing cutoffs without rerunning inference.
detector=TensorRTDetector(str(ENGINE),confidence=.02,iou=NMS)
records=[];start=time.perf_counter()
try:
 for sample in samples:
  row={'frame':sample['frame'],'luma_mean':round(sample['luma_mean'],2),'luma_p90':round(sample['luma_p90'],2),'configs':{}}
  for config in CONFIGS:
   image=enhance(sample['image'],config);boxes,latency=detector.infer(image)
   row['configs'][config]={'detections':[{'x1':b.x1,'y1':b.y1,'x2':b.x2,'y2':b.y2,'confidence':b.confidence} for b in boxes],'inference_ms':latency}
  records.append(row)
finally:detector.close()
results=[]
for config in CONFIGS:
 cfg={}
 for threshold in (.02,.05,.075,.10):
  tp=fp=fn=exact=paired=0;center_errors=[[],[]];target_confs=[[],[]]
  for row in records:
   # Convert row dicts to simple namespace-like items used by score.
   boxes=[type('Box',(),{'x1':b['x1'],'y1':b['y1'],'x2':b['x2'],'y2':b['y2'],'confidence':b['confidence']}) for b in row['configs'][config]['detections']]
   hits,extras=score(boxes,threshold);n=sum(h is not None for h in hits);tp+=n;fn+=2-n;fp+=extras;paired+=n==2;exact+=n==2 and extras==0
   for i,item in enumerate(hits):
    if item:center_errors[i].append(item[1]);target_confs[i].append(item[0].confidence)
  cfg[str(threshold)]={'paired_frames':paired,'exact_two_frames':exact,'tp':tp,'fp':fp,'fn':fn,'target_median_center_error_px':[round(statistics.median(x),2) if x else None for x in center_errors],'target_min_confidence':[round(min(x),4) if x else None for x in target_confs]}
 results.append({'config':config,'thresholds':cfg,'median_inference_ms':round(statistics.median(r['configs'][config]['inference_ms'] for r in records),2)})
out={'method':'Run the current FRC YOLO26n 1280x960 TensorRT engine on frames 0-31 of the darker saved stereo clip after raw, gamma, CLAHE, or equalized-Y preprocessing. Score confidence thresholds on the same detections at fixed NMS IoU 0.15.','limitations':'One dark scene and 32 frames; synthetic preprocessing cannot recreate sensor exposure, noise, or motion blur. Targets are stationary and marked by fixed center windows. Range is not measured here.','video':'media/frc-2026-live-two-fuel-60fps-cfr.mp4','frames':len(records),'nms_iou':NMS,'brightness_by_frame':[{'frame':r['frame'],'mean':r['luma_mean'],'p90':r['luma_p90']} for r in records],'results':results,'per_frame':records}
path=ROOT/'media/frc-fuel-dark-scene-preprocessing.json';path.write_text(json.dumps(out,indent=2)+'\n')
for r in results:print(r['config'],json.dumps(r['thresholds']), 'median_ms',r['median_inference_ms'])
print('elapsed_s',round(time.perf_counter()-start,2),'saved',path)
