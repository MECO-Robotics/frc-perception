#!/usr/bin/env python3
"""Evaluate post-NMS for YOLO26 end-to-end detections under synthetic masks."""
import json,math,os,statistics,sys
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scout.detectors import TensorRTDetector
TARGETS=((808,589),(902,591));TIMES=tuple(1.0+.4*i for i in range(15));FRACS=(0,.25,.5,.75);IOUS=(None,.05,.10,.15,.20)

def overlap(a,b):
 x1=max(a['x1'],b['x1']);y1=max(a['y1'],b['y1']);x2=min(a['x2'],b['x2']);y2=min(a['y2'],b['y2']);inter=max(0,x2-x1)*max(0,y2-y1);aa=(a['x2']-a['x1'])*(a['y2']-a['y1']);ab=(b['x2']-b['x1'])*(b['y2']-b['y1']);return inter/(aa+ab-inter) if aa+ab>inter else 0

def nms(boxes,threshold):
 if threshold is None:return boxes
 kept=[]
 for b in sorted(boxes,key=lambda z:z['confidence'],reverse=True):
  if all(overlap(b,k)<=threshold for k in kept):kept.append(b)
 return kept

def metric(rows,threshold):
 tp=fp=fn=exact=pair_iou=0
 for row in rows:
  hits=[None,None];extra=0
  for b in sorted(nms([b for b in row if b['confidence']>=.05],threshold),key=lambda z:z['confidence'],reverse=True):
   cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2;d,i=min((math.hypot(cx-x,cy-y),i) for i,(x,y) in enumerate(TARGETS))
   if d<=60 and hits[i] is None:hits[i]=b
   else:extra+=1
  n=sum(x is not None for x in hits);tp+=n;fn+=2-n;fp+=extra;exact+=n==2 and extra==0
  if all(hits):pair_iou=max(pair_iou,overlap(*hits))
 return {'tp':tp,'fp':fp,'fn':fn,'exact_two_frames':exact,'max_target_pair_iou':round(pair_iou,4)}

os.environ.update(SCOUT_CLASS_NAMES='["ball"]',SCOUT_MAX_BOX_ASPECT_RATIO='2.0')
source=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
detector=TensorRTDetector(str(ROOT/'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'),confidence=.005)
out={'method':'At 15 stable timestamps per recording, mask the lower 0/25/50/75% of each manually matched clean-frame target box, rerun YOLO26 at 0.005, then compare no extra NMS against same-class greedy post-NMS at IoU .05/.10/.15/.20, scored at confidence .05.','limitations':'Black rectangles are synthetic occlusions, not real foreground objects; target windows are fixed. Very low IoU thresholds may suppress two distinct, closely overlapping FUEL boxes in other poses.','confidence':.05,'sample_times_s':TIMES,'occlusion_fractions':FRACS,'iou_thresholds':IOUS,'results':[]}
try:
 for clip in source['results']:
  cap=cv2.VideoCapture(str(ROOT/clip['video']));fps=cap.get(cv2.CAP_PROP_FPS);rows={frac:[] for frac in FRACS};lat=[]
  for t in TIMES:
   idx=round(t*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read()
   if not ok:continue
   left=frame[:,:1600].copy();targets=[None,None]
   for b in sorted((b for b in clip['per_frame'][idx]['detections'] if b['confidence']>=.1),key=lambda b:b['confidence'],reverse=True):
    cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2;d,i=min((math.hypot(cx-x,cy-y),i) for i,(x,y) in enumerate(TARGETS))
    if d<=60 and targets[i] is None:targets[i]=b
   for frac in FRACS:
    img=left.copy()
    if frac:
     for b in targets:
      if b is None:continue
      y1=round(b['y1']+(b['y2']-b['y1'])*(1-frac));img[y1:b['y2'],b['x1']:b['x2']]=0
    boxes,ms=detector.infer(img);lat.append(ms);rows[frac].append([{'x1':b.x1,'y1':b.y1,'x2':b.x2,'y2':b.y2,'confidence':b.confidence} for b in boxes])
  cap.release();entry={'scene':clip['scene'],'video':clip['video'],'fractions':[]}
  for frac in FRACS:
   entry['fractions'].append({'occlusion_fraction':frac,'frames':len(rows[frac]),'configs':[{'iou_threshold':iou,**metric(rows[frac],iou)} for iou in IOUS]})
  out['results'].append(entry)
  print(clip['scene'],[(r['occlusion_fraction'],[(c['iou_threshold'],c['exact_two_frames'],c['tp'],c['fp'],c['fn'],c['max_target_pair_iou']) for c in r['configs']]) for r in entry['fractions']])
finally:detector.close()
out['median_detector_inference_ms']=round(statistics.median(lat),2) if lat else None
p=ROOT/'media/frc-fuel-nms-occlusion-sweep.json';p.write_text(json.dumps(out,indent=2)+'\n');print('saved',p)
