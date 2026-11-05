#!/usr/bin/env python3
"""Compare low-res live YOLO anchors vs full-res HSV/Hough anchors + production flow/Kalman."""
import json, math, os, statistics, sys, time
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['SCOUT_CLASS_NAMES']='["ball"]';os.environ['SCOUT_MAX_BOX_ASPECT_RATIO']='2.0'
from scout.detectors import TensorRTDetector,Detection
from scout.tracking import FuelTracker
VIDEO=ROOT/'media/frc-fuel-live-heavy-occlusion-20261103.mp4';ENGINE=ROOT/'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
OUT=ROOT/'media/frc-fuel-hsv-hough-live-resolution-20261103.json';FPS=59.966685;STEP=3;SCALE=.2
ROI_FULL=(680,500,1010,660)

def pct(v,p):return round(float(np.percentile(v,p)),3) if v else None
def med(v):return round(float(statistics.median(v)),3) if v else None

def hsv_hough(full):
 x0,y0,x1,y1=ROI_FULL;crop=full[y0:y1,x0:x1];t=time.perf_counter()
 hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);mask=cv2.inRange(hsv,(10,150,70),(40,255,255));mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
 gray=cv2.GaussianBlur(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(9,9),2)
 raw=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1.2,minDist=30,param1=90,param2=25,minRadius=30,maxRadius=65);cands=[]
 if raw is not None:
  yy,xx=np.ogrid[:crop.shape[0],:crop.shape[1]]
  for x,y,r in raw[0]:
   X,Y=float(x+x0),float(y+y0)
   if not(740<=X<=960 and 530<=Y<=640):continue
   disk=(xx-x)**2+(yy-y)**2<=r*r;purity=float(np.mean(mask[disk]>0)) if disk.any() else 0
   if purity>=.25:cands.append((X,Y,float(r),purity))
 pairs=[]
 for i,a in enumerate(cands):
  for b in cands[i+1:]:
   gap=math.dist(a[:2],b[:2]);dy=abs(a[1]-b[1])
   if 20<=gap<=95 and dy<=38:
    score=abs((a[0]+b[0])*.5-845)+1.5*dy+.35*abs(a[2]-b[2]);pairs.append((score,a,b))
 if pairs:
  _,a,b=min(pairs,key=lambda q:q[0]);selected=sorted((a,b),key=lambda q:q[0])
 elif cands:selected=[min(cands,key=lambda q:abs(q[0]-845)+2*abs(q[1]-585))]
 else:selected=[]
 dt=(time.perf_counter()-t)*1000
 ds=[Detection(round((x-r)*SCALE),round((y-r)*SCALE),round((x+r)*SCALE),round((y+r)*SCALE),0,max(.05,min(.99,p))) for x,y,r,p in selected]
 return ds,dt,len(cands),len(selected)

cap=cv2.VideoCapture(str(VIDEO));detector=TensorRTDetector(str(ENGINE),confidence=.05,iou=.15);trackers={'live_yolo':FuelTracker(),'hsv_hough':FuelTracker()}
flow_ms={k:[] for k in trackers};kalman_ms={k:[] for k in trackers};anchor_ms={k:[] for k in trackers};anchor_hist={k:{} for k in trackers};counts={k:[] for k in trackers};per_frame=[];frame_i=0;start=time.perf_counter();sample_lat=[]
try:
 while True:
  ok,frame=cap.read()
  if not ok:break
  full=frame[:,:1600];low=cv2.resize(full,(320,240),interpolation=cv2.INTER_AREA);now=frame_i/FPS
  for name,tr in trackers.items():
   t=time.perf_counter();tr.advance(low,now,frame_i);flow_ms[name].append((time.perf_counter()-t)*1000)
  pair_info=None
  if frame_i>0 and frame_i%STEP==0:
   boxes,ms=detector.infer(low);sample_lat.append(ms);yolo_boxes=[]
   for b in boxes:
    cx=(b.x1+b.x2)/2;cy=(b.y1+b.y2)/2
    if b.confidence>=.05 and 130<=cx<=210 and 95<=cy<=145:yolo_boxes.append(b)
   yolo_boxes=sorted(yolo_boxes,key=lambda b:b.confidence,reverse=True)[:2]
   ds,hm,ncand,nselected=hsv_hough(full);pair_info={'frame':frame_i,'yolo_boxes':len(yolo_boxes),'hsv_confirmed_hough_candidates':ncand,'hough_selected':nselected}
   for name,dets,tm in [('live_yolo',yolo_boxes,ms),('hsv_hough',ds,hm)]:
    anchor_ms[name].append(tm);anchor_hist[name][len(dets)]=anchor_hist[name].get(len(dets),0)+1
    t=time.perf_counter();trackers[name].update(dets,now,frame_i);kalman_ms[name].append((time.perf_counter()-t)*1000)
  curr={}
  for name,tr in trackers.items():
   ts=tr.snapshot(now);counts[name].append(len(ts));curr[name]=[{'id':x['track_id'],'state':x['state'],'center_xy_px':[round(x['box'][0],1),round(x['box'][1],1)],'confidence':round(x['confidence'],3)} for x in ts]
  if pair_info:pair_info['tracks']=curr;per_frame.append(pair_info)
  frame_i+=1
finally:detector.close();cap.release()
wall=time.perf_counter()-start
variants=[]
for name in trackers:
 flowmed=med(flow_ms[name]);kalmanmed=med(kalman_ms[name]);anchor_m=med(anchor_ms[name]);budget=flowmed+kalmanmed/STEP+anchor_m/STEP
 variants.append({'name':name,'anchor_hz':round(len(anchor_ms[name])/(frame_i/FPS),2),'anchors':len(anchor_ms[name]),'anchor_detection_count_histogram':{str(k):v for k,v in sorted(anchor_hist[name].items())},'frames_zero_tracks':counts[name].count(0),'frames_one_track':counts[name].count(1),'frames_two_tracks':counts[name].count(2),'frames_more_than_two_tracks':sum(x>2 for x in counts[name]),'reported_position_rate_hz':round(FPS,2),'flow_advance_latency_ms':{'median':flowmed,'p95':pct(flow_ms[name],95)},'kalman_update_latency_ms':{'median':kalmanmed,'p95':pct(kalman_ms[name],95)},'anchor_detection_latency_ms':{'median':anchor_m,'p95':pct(anchor_ms[name],95)},'serial_stage_budget_ms_per_frame_estimate':round(budget,3),'serial_stage_budget_rate_hz_estimate':round(1000/budget,1),'replay_wall_fps':round(frame_i/wall,2)})
summary={'method':'Decode the recorded stereo capture sequentially. Downscale the left view to 320x240 before TensorRT and the production FuelTracker, matching live stats. YOLO anchors run every third frame. The hybrid uses the full-resolution left view for ROI HSV segmentation and Hough circles every same anchor frame, then maps circle boxes to 320x240 for the same Kalman+LK tracker.','video':str(VIDEO.relative_to(ROOT)),'frames':frame_i,'video_fps':FPS,'live_tracker_input_resolution':[320,240],'high_resolution_hsv_hough_roi_xyxy_px':ROI_FULL,'anchor_interval_frames':STEP,'anchor_rate_hz':round(FPS/STEP,2),'hough_parameters':{'hue':[10,40],'saturation_min':150,'value_min':70,'hough_param1':90,'hough_param2':25,'radius_fullres_px':[30,65],'min_distance_fullres_px':30,'hsv_disk_purity_min':.25},'tensorRT_engine':str(ENGINE.relative_to(ROOT)),'tensorRT_precision':'FP16 GPU','tensorRT_latency_ms':{'median':med(sample_lat),'p95':pct(sample_lat,95)},'variants':variants,'replay_wall_time_s':round(wall,2),'per_anchor':per_frame,'limitations':['The Hough branch reads the original recorded 1600x1200 left image; the current tracker operates on 320x240. A live deployment would need a parallel high-resolution ROI path to preserve those circle pixels.','This is a stationary two-ball scene. It measures count continuity and compute rate, not true moving-object localization or stereo range.','The replay processes both branches in one pass, and its wall throughput includes software MP4 decode. The serial stage budget excludes decode, stereo, NetworkTables, GStreamer transfer, and CPU/GPU scheduling contention.','The active production size gate rejected the merged YOLO box in the live capture. This controlled comparison isolates the raw YOLO anchor plus tracker and the proposed HSV/Hough anchor plus tracker.']}
OUT.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='per_anchor'},indent=2));print('saved',OUT)
