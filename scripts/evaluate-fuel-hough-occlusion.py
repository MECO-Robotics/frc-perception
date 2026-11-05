#!/usr/bin/env python3
"""Test whether Hough sphere fits preserve apparent diameter under synthetic occlusion."""
import json,math,statistics
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
TARGETS=((808,589),(902,591));TIMES=tuple(1.0+.4*i for i in range(15));FRACS=(0,.25,.5,.75);P2S=(20,25,30)
fs=cv2.FileStorage(str(ROOT/'media/calibration/stereo_calibration_live_order_candidate.yml'),cv2.FILE_STORAGE_READ);focal=float(fs.getNode('left_camera_matrix').mat()[0,0]);fs.release()
def median(x):return statistics.median(x) if x else None
def mad(x):
 m=median(x);return median([abs(v-m) for v in x]) if x else None
def match(row):
 out=[None,None]
 for b in sorted((b for b in row['detections'] if b['confidence']>=.1),key=lambda x:x['confidence'],reverse=True):
  cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2;ds=[math.hypot(cx-x,cy-y) for x,y in TARGETS];i=min(range(2),key=lambda j:ds[j])
  if ds[i]<=60 and out[i] is None:out[i]=b
 return out
out={'method':'On stable saved frames, apply a lower black mask to the target detector box in the left view, then Hough-fit a circle in the fixed 160x160 target patch. Compare radii and nominal-size range across occlusion fractions. This synthetic mask tests arc fitting only; it is not real foreground occlusion or range truth.','hough_param2':list(P2S),'occlusion_fractions':list(FRACS),'fuel_diameter_m':.15,'focal_px':focal,'results':[]}
for ci,clip in enumerate(D['results']):
 cap=cv2.VideoCapture(str(ROOT/clip['video']));fps=cap.get(cv2.CAP_PROP_FPS);rows={f:{p:[[],[]] for p in P2S} for f in FRACS}
 for t in TIMES:
  idx=round(t*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read()
  if not ok:continue
  left=frame[:,:1600].copy();boxes=match(clip['per_frame'][idx])
  for frac in FRACS:
   img=left.copy()
   if frac:
    for b in boxes:
     if b is None:continue
     y1=round(b['y1']+(b['y2']-b['y1'])*(1-frac));img[y1:b['y2'],b['x1']:b['x2']]=0
   gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   for ti,(cx,cy) in enumerate(TARGETS):
    x0,y0=cx-80,cy-80;crop=gray[y0:cy+80,x0:cx+80];blur=cv2.GaussianBlur(crop,(9,9),2)
    for p2 in P2S:
     cs=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,dp=1.2,minDist=30,param1=90,param2=p2,minRadius=35,maxRadius=65)
     candidates=[]
     if cs is not None:
      for x,y,r in cs[0]:
       off=math.hypot(x+x0-cx,y+y0-cy)
       if off<=35:candidates.append((off,float(r)))
     if candidates:
      off,r=min(candidates);rows[frac][p2][ti].append({'radius_px':r,'offset_px':off,'time_s':t,'range_m':focal*.15/(2*r)})
 cap.release();result={'scene':clip['scene'],'configs':[]}
 for frac in FRACS:
  for p2 in P2S:
   tr=[]
   for ti in range(2):
    vals=rows[frac][p2][ti];r=[x['radius_px'] for x in vals];z=[x['range_m'] for x in vals]
    tr.append({'target':ti+1,'circle_frames':len(vals),'radius_median_px':round(median(r),2) if r else None,'radius_mad_px':round(mad(r),2) if r else None,'size_range_median':round(median(z),3) if z else None,'size_range_mad_m':round(mad(z),3) if z else None})
   result['configs'].append({'occlusion_fraction':frac,'param2':p2,'targets':tr});print(clip['scene'],frac,p2,tr)
 out['results'].append(result)
path=ROOT/'media/frc-fuel-hough-occlusion-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
