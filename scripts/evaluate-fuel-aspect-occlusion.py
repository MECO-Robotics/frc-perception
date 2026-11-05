#!/usr/bin/env python3
"""Check a symmetric aspect-ratio postfilter under synthetic FUEL occlusion."""
import json, math, os, statistics, time
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0,str(ROOT))
from scout.detectors import TensorRTDetector
TARGETS=((808,589),(902,591)); TIMES=tuple(1.0+.4*i for i in range(15)); FRACS=(0,.25,.5,.75)
CONF=.05; LIMITS=(1.3,2.0)
def eval_rows(rows,limit):
 tp=fp=fn=exact=0
 for row in rows:
  hits=[None,None]; extras=0
  for b in sorted((b for b in row if b['confidence']>=CONF and max((b['x2']-b['x1'])/(b['y2']-b['y1']),(b['y2']-b['y1'])/(b['x2']-b['x1']))<=limit),key=lambda b:b['confidence'],reverse=True):
   cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
   distance,i=min((math.hypot(cx-x,cy-y),i) for i,(x,y) in enumerate(TARGETS))
   if distance<=60 and hits[i] is None: hits[i]=b
   else: extras+=1
  count=sum(b is not None for b in hits);tp+=count;fn+=2-count;fp+=extras;exact+=count==2 and extras==0
 return {'tp':tp,'fp':fp,'fn':fn,'exact_two_frames':exact,'precision':round(tp/(tp+fp),5) if tp+fp else 0,'recall':round(tp/(tp+fn),5) if tp+fn else 0}
source=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
os.environ.update(SCOUT_CLASS_NAMES='["ball"]',SCOUT_MAX_BOX_ASPECT_RATIO='2.0')
detector=TensorRTDetector(str(ROOT/'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'),confidence=.005)
out={'method':'On 15 stable frames per clip (1.0–6.6 s), mask the lower 0/25/50/75% of each manually matched clean-frame ball box, rerun the 1280x960 TensorRT detector at 0.005, then score confidence 0.05 with symmetric max(w/h,h/w) limits 1.3 and 2.0. Masks are synthetic black rectangles, not real foreground occlusions.','limitations':'Fixed target locations and synthetic lower masks only; does not model arbitrary occluder shapes, motion blur, crossings, or different field backgrounds. Detector postprocessing still runs with the broad 2.0 cap before the evaluated tighter gate.','sample_times_s':TIMES,'confidence':CONF,'aspect_limits':LIMITS,'results':[]}
try:
 for ci,clip in enumerate(source['results']):
  cap=cv2.VideoCapture(str(ROOT/clip['video']));fps=cap.get(cv2.CAP_PROP_FPS); per={f:[] for f in FRACS};lat=[]
  for t in TIMES:
   idx=round(t*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read()
   if not ok:continue
   left=frame[:,:1600].copy(); targets=[None,None]
   for b in sorted((b for b in clip['per_frame'][idx]['detections'] if b['confidence']>=.1),key=lambda b:b['confidence'],reverse=True):
    cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2;ds=[math.hypot(cx-x,cy-y) for x,y in TARGETS];d,i=min((v,i) for i,v in enumerate(ds))
    if d<=60 and targets[i] is None:targets[i]=b
   for frac in FRACS:
    img=left.copy()
    if frac:
     for b in targets:
      if b is None:continue
      y1=round(b['y1']+(b['y2']-b['y1'])*(1-frac));img[y1:b['y2'],b['x1']:b['x2']]=0
    boxes,ms=detector.infer(img);lat.append(ms)
    per[frac].append([{'x1':b.x1,'y1':b.y1,'x2':b.x2,'y2':b.y2,'confidence':b.confidence} for b in boxes])
  cap.release();entry={'scene':clip['scene'],'video':clip['video'],'fractions':[]}
  for frac in FRACS:
   entry['fractions'].append({'occlusion_fraction':frac,'frames':len(per[frac]),'configs':[{'max_symmetric_aspect':lim,**eval_rows(per[frac],lim)} for lim in LIMITS]})
  out['results'].append(entry)
  print(clip['scene'],[(r['occlusion_fraction'],[(c['max_symmetric_aspect'],c['exact_two_frames'],c['tp'],c['fp'],c['fn']) for c in r['configs']]) for r in entry['fractions']])
finally:detector.close()
out['median_detector_inference_ms']=round(statistics.median(lat),2) if lat else None
p=ROOT/'media/frc-fuel-aspect-occlusion-sweep.json';p.write_text(json.dumps(out,indent=2)+'\n');print('saved',p)
