#!/usr/bin/env python3
"""Re-evaluate a luma-triggered gamma rule against every saved-video frame."""
import json,math,os,statistics,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scout.detectors import TensorRTDetector
VIDEOS=(('earlier','media/frc-2026-live-two-fuel-60fps-cfr.mp4'),('corrected-settings','media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'))
BASE=json.loads((ROOT/'media/frc-fuel-dense-count-nms015.json').read_text())
TARGETS=((808.,589.),(902.,591.));NMS=.15;BASE_CONF=.02;SCORE_CONF=.05;GATES=(15.,20.,25.,30.)
GAMMA=.6;LUT=np.clip((np.arange(256,dtype=np.float32)/255.)**GAMMA*255,0,255).astype(np.uint8)
def score(dets):
 hits=[None,None];extras=0
 for b in sorted((b for b in dets if b['confidence']>=SCORE_CONF),key=lambda b:-b['confidence']):
  x=(b['x1']+b['x2'])/2;y=(b['y1']+b['y2'])/2;err,i=min((math.hypot(x-a,y-c),i) for i,(a,c) in enumerate(TARGETS))
  if err<=60 and hits[i] is None:hits[i]=(b,err)
  else:extras+=1
 return hits,extras
def summarize(rows,gate=None):
 tp=fp=fn=paired=exact=gamma_n=0;errors=[[],[]];changed=[]
 for row in rows:
  use=gate is not None and row['mean_luma']<gate
  dets=row['gamma'] if use else row['raw'];gamma_n+=use
  h,e=score(dets);n=sum(x is not None for x in h);tp+=n;fp+=e;fn+=2-n;paired+=n==2;exact+=n==2 and e==0
  for i,x in enumerate(h):
   if x:errors[i].append(x[1])
  if use:
   hr,_=score(row['raw']);
   if any((hr[i] is None)!=(h[i] is None) or (hr[i] is not None and h[i] is not None and abs(hr[i][0]['x1']-h[i][0]['x1'])+abs(hr[i][0]['y1']-h[i][0]['y1'])+abs(hr[i][0]['x2']-h[i][0]['x2'])+abs(hr[i][0]['y2']-h[i][0]['y2'])>0) for i in range(2)):changed.append(row['frame'])
 return {'frames':len(rows),'gamma_frames':gamma_n,'paired_frames':paired,'exact_two_frames':exact,'tp':tp,'fp':fp,'fn':fn,'center_error_median_px':[round(statistics.median(a),2) if a else None for a in errors],'changed_gamma_frames':changed}
os.environ.update(SCOUT_CLASS_NAMES='["ball"]',SCOUT_MAX_BOX_ASPECT_RATIO='2.0',SCOUT_NMS_IOU=str(NMS))
det=TensorRTDetector(str(ROOT/'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'),confidence=BASE_CONF,iou=NMS)
out={'method':'Score raw current NMS 0.15 detections on every frame, and conditionally replace them with gamma .60 inference only when original left-image mean grayscale luma falls below gates 15/20/25/30. Existing raw detections are reused; gamma inference runs only on frames below 30. Operating confidence is .05.','limitations':'Two stationary indoor clips; the dim clip has a short dark opening and the corrected clip is brighter. This measures only this exposure transition, not field lighting variation, sensor gain/noise, motion blur, or measured distance.','gamma':GAMMA,'luma_gates':GATES,'confidence':SCORE_CONF,'nms_iou':NMS,'results':[]}
try:
 for ci,(scene,video) in enumerate(VIDEOS):
  base=BASE['results'][ci];cap=cv2.VideoCapture(str(ROOT/video));rows=[];idx=0;started=time.perf_counter()
  while True:
   ok,frame=cap.read()
   if not ok:break
   left=frame[:,:1600];luma=float(cv2.cvtColor(left,cv2.COLOR_BGR2GRAY).mean());raw=base['per_frame'][idx]['detections'];gamma=None
   if luma<max(GATES):
    bs,_=det.infer(cv2.LUT(left,LUT));gamma=[{'x1':b.x1,'y1':b.y1,'x2':b.x2,'y2':b.y2,'confidence':float(b.confidence)} for b in bs]
   rows.append({'frame':idx,'mean_luma':luma,'raw':raw,'gamma':gamma if gamma is not None else raw})
   idx+=1
  cap.release();config=[{'policy':'raw','metrics':summarize(rows)}]
  for gate in GATES:config.append({'policy':f'gamma_if_luma_below_{gate:g}','metrics':summarize(rows,gate)})
  out['results'].append({'scene':scene,'video':video,'decoded_frames':idx,'elapsed_s':round(time.perf_counter()-started,2),'luma_percentiles':[round(float(np.percentile([r['mean_luma'] for r in rows],x)),2) for x in (5,50,95)],'configs':config})
  print(scene,[(r['policy'],r['metrics']['gamma_frames'],r['metrics']['paired_frames'],r['metrics']['exact_two_frames'],r['metrics']['tp'],r['metrics']['fp'],r['metrics']['fn']) for r in config])
finally:det.close()
p=ROOT/'media/frc-fuel-adaptive-luma-gamma06-full-frames.json';p.write_text(json.dumps(out,indent=2)+'\n');print('saved',p)
