#!/usr/bin/env python3
"""Compare dense detector-box stereo, box-size, and Hough-circle FUEL ranges."""
import cv2,json,math,statistics,time
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
S=json.load(open(ROOT/'media/frc-fuel-dense-stereo-evaluation.json'))
fs=cv2.FileStorage(str(ROOT/'media/calibration/stereo_calibration_live_order_candidate.yml'),cv2.FILE_STORAGE_READ);focal=float(fs.getNode('left_camera_matrix').mat()[0,0]);fs.release()
TARGETS=((808,589),(902,591));PARAM2=(20,25,30);DIAM=.15
def med(a):return statistics.median(a) if a else None
def stats(a):
 if not a:return None
 m=med(a);return {'n':len(a),'median_m':round(m,4),'mad_m':round(med([abs(x-m) for x in a]),4),'p05_m':round(sorted(a)[int(.05*(len(a)-1))],4),'p95_m':round(sorted(a)[int(.95*(len(a)-1))],4)}
def match(row):
 out=[None,None]
 for b in sorted((x for x in row['detections'] if x['confidence']>=.05),key=lambda x:x['confidence'],reverse=True):
  cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2;ds=[math.hypot(cx-x,cy-y) for x,y in TARGETS];i=min(range(2),key=lambda j:ds[j])
  if ds[i]<=60 and out[i] is None:out[i]=b
 return out
out={'method':'Dense FRC9550 detector boxes at confidence >=0.05. For each matched target, run Hough on a detector-centered crop; search radii 0.35-0.65 times the box short side and require circle center within 0.30 short side. Compare nominal 15 cm circle-size range with production SGBM Z and box-size range. Hough size range is a proxy, not measured truth.','fuel_diameter_m':DIAM,'focal_px':focal,'param2':list(PARAM2),'results':[]}
for ci,clip in enumerate(D['results']):
 cap=cv2.VideoCapture(str(ROOT/clip['video']));records={p:{'hough':[[],[]],'box_size':[[],[]],'stereo':[[],[]],'agree':[]} for p in PARAM2};latency={p:[] for p in PARAM2};started=time.perf_counter()
 for row,srow in zip(clip['per_frame'],S['results'][ci]['frames']):
  ok,frame=cap.read()
  if not ok:break
  gray=cv2.cvtColor(frame[:,:1600],cv2.COLOR_BGR2GRAY)
  for ti,b in enumerate(match(row)):
   if b is None:continue
   cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2;short=min(b['x2']-b['x1'],b['y2']-b['y1']);size_z=focal*DIAM/math.sqrt((b['x2']-b['x1'])*(b['y2']-b['y1']))
   z=srow['ranges_m'][ti]
   for p2 in PARAM2:
    if size_z:records[p2]['box_size'][ti].append(size_z)
    if z is not None:records[p2]['stereo'][ti].append(z);records[p2]['agree'].append(abs(size_z-z)/z)
   rad_lo=max(8,int(short*.35));rad_hi=max(rad_lo+1,int(short*.65));x0=int(round(cx-short*.8));y0=int(round(cy-short*.8));x1=x0+int(short*1.6);y1=y0+int(short*1.6);x0=max(0,x0);y0=max(0,y0);x1=min(1600,x1);y1=min(1200,y1)
   crop=gray[y0:y1,x0:x1]
   if crop.size==0:continue
   blur=cv2.GaussianBlur(crop,(9,9),2)
   for p2 in PARAM2:
    ts=time.perf_counter();cs=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,dp=1.2,minDist=max(10,int(short*.5)),param1=90,param2=p2,minRadius=rad_lo,maxRadius=rad_hi);latency[p2].append((time.perf_counter()-ts)*1000)
    cand=[]
    if cs is not None:
     for x,y,r in cs[0]:
      off=math.hypot(x+x0-cx,y+y0-cy)
      if off<=short*.30:cand.append((off,float(r)))
    if cand:
     _,r=min(cand);z_h=focal*DIAM/(2*r);records[p2]['hough'][ti].append(z_h)
 cap.release();entry={'scene':clip['scene'],'frames_processed':600,'elapsed_s':round(time.perf_counter()-started,1),'configs':[]}
 for p2 in PARAM2:
  r=records[p2];entry['configs'].append({'param2':p2,'hough_range_m':[stats(x) for x in r['hough']],'box_size_range_m':[stats(x) for x in r['box_size']],'stereo_range_m':[stats(x) for x in r['stereo']],'size_stereo_relative_error_median':round(med(r['agree']),3) if r['agree'] else None,'hough_ms_median':round(med(latency[p2]),3),'hough_ms_p95':round(sorted(latency[p2])[int(.95*(len(latency[p2])-1))],3)});print(clip['scene'],entry['configs'][-1])
 out['results'].append(entry)
path=ROOT/'media/frc-fuel-dense-hough-range.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
