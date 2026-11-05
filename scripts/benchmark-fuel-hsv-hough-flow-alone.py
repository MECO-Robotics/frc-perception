#!/usr/bin/env python3
"""Isolated CPU replay rate for HSV/Hough 20 Hz anchors plus live-scale LK/Kalman."""
import json,math,statistics,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from scout.detectors import Detection
from scout.tracking import FuelTracker
VIDEO=ROOT/'media/frc-fuel-live-heavy-occlusion-20261103.mp4';REF=json.load(open(ROOT/'media/frc-fuel-live-heavy-occlusion-20261103-replay.json'));OUT=ROOT/'media/frc-fuel-hsv-hough-flow-alone-20261103.json'
FPS=59.966685;STEP=3;SCALE=.2;ROI=(680,500,1010,660);anchor_frames={r['frame'] for r in REF['per_sample']}
def hough(full):
 x0,y0,x1,y1=ROI;crop=full[y0:y1,x0:x1];t=time.perf_counter();hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);mask=cv2.inRange(hsv,(10,150,70),(40,255,255));mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8));gray=cv2.GaussianBlur(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(9,9),2);raw=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1.2,minDist=30,param1=90,param2=25,minRadius=30,maxRadius=65);cs=[]
 if raw is not None:
  yy,xx=np.ogrid[:crop.shape[0],:crop.shape[1]]
  for x,y,r in raw[0]:
   X,Y=float(x+x0),float(y+y0)
   if 740<=X<=960 and 530<=Y<=640:
    disk=(xx-x)**2+(yy-y)**2<=r*r;pur=float(np.mean(mask[disk]>0)) if disk.any() else 0
    if pur>=.25:cs.append((X,Y,float(r),pur))
 pairs=[]
 for i,a in enumerate(cs):
  for b in cs[i+1:]:
   gap=math.dist(a[:2],b[:2]);dy=abs(a[1]-b[1])
   if 20<=gap<=95 and dy<=38:pairs.append((abs((a[0]+b[0])*.5-845)+1.5*dy+.35*abs(a[2]-b[2]),a,b))
 if pairs:_,a,b=min(pairs,key=lambda z:z[0]);sel=sorted((a,b),key=lambda z:z[0])
 elif cs:sel=[min(cs,key=lambda z:abs(z[0]-845)+2*abs(z[1]-585))]
 else:sel=[]
 dt=(time.perf_counter()-t)*1000;ds=[Detection(round((x-r)*SCALE),round((y-r)*SCALE),round((x+r)*SCALE),round((y+r)*SCALE),0,p) for x,y,r,p in sel]
 return ds,dt,len(cs)
cap=cv2.VideoCapture(str(VIDEO));tr=FuelTracker();flow=[];updates=[];detect=[];hist={};rawhist={};per=[];counts=[];i=0;start=time.perf_counter()
while True:
 ok,frame=cap.read()
 if not ok:break
 low=cv2.resize(frame[:,:1600],(320,240),interpolation=cv2.INTER_AREA);now=i/FPS;t=time.perf_counter();tr.advance(low,now,i);flow.append((time.perf_counter()-t)*1000)
 if i in anchor_frames:
  dets,ms,n=hough(frame[:,:1600]);detect.append(ms);hist[len(dets)]=hist.get(len(dets),0)+1;rawhist[n]=rawhist.get(n,0)+1
  t=time.perf_counter();tr.update(dets,now,i);updates.append((time.perf_counter()-t)*1000)
 tracks=tr.snapshot(now);counts.append(len(tracks));
 if i in anchor_frames:per.append({'frame':i,'circle_count':len(dets),'tracked_count':len(tracks),'track_centers_xy_320x240':[[round(q['box'][0],2),round(q['box'][1],2)] for q in tracks]})
 i+=1
cap.release();wall=time.perf_counter()-start
fm,um,dm=statistics.median(flow),statistics.median(updates),statistics.median(detect);budget=fm+um/STEP+dm/STEP
out={'video':str(VIDEO.relative_to(ROOT)),'frames':i,'fps':FPS,'tracker_input_resolution':[320,240],'anchors':len(detect),'anchor_hz':round(len(detect)/(i/FPS),2),'hough_count_histogram':{str(k):v for k,v in sorted(hist.items())},'hsv_confirmed_candidate_histogram':{str(k):v for k,v in sorted(rawhist.items())},'tracked_frame_histogram':{str(k):counts.count(k) for k in sorted(set(counts))},'track_output_rate_hz':FPS,'hsv_hough_anchor_latency_ms':{'median':round(dm,3),'p95':round(float(np.percentile(detect,95)),3)},'lk_advance_latency_ms':{'median':round(fm,3),'p95':round(float(np.percentile(flow,95)),3)},'kalman_update_latency_ms':{'median':round(um,3),'p95':round(float(np.percentile(updates,95)),3)},'serial_stage_budget_ms_per_frame':round(budget,3),'serial_stage_budget_hz':round(1000/budget,1),'replay_wall_s':round(wall,2),'replay_wall_fps_including_mp4_decode':round(i/wall,2),'per_anchor':per,'limitations':['Stationary scene and fixed high-resolution ROI.','The Hough path uses 1600x1200 source pixels for circle cues and scales positions to the live 320x240 tracker input; live deployment needs the corresponding high-resolution ROI branch.','No stereo range, moving-ball, CPU contention, or NetworkTables costs are included.']}
OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='per_anchor'},indent=2));print('saved',OUT)
