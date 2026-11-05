#!/usr/bin/env python3
"""Compare YOLO+LK/Kalman and HSV/Hough+LK/Kalman on heavy occlusion video."""
import json, math, statistics, sys, time
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scout.detectors import Detection
from scout.tracking import FuelTracker
VIDEO=ROOT/'media/frc-fuel-live-heavy-occlusion-20261103.mp4'
YOLO=json.load(open(ROOT/'media/frc-fuel-live-heavy-occlusion-20261103-replay.json'))
ROI=(680,500,1010,660);STEP=3;FPS=59.966685

def median(v): return round(statistics.median(v),3) if v else None
def percentile(v,p): return round(float(np.percentile(v,p)),3) if v else None

def hsv_hough(frame):
 x0,y0,x1,y1=ROI;crop=frame[y0:y1,x0:x1]
 t=time.perf_counter();hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
 mask=cv2.inRange(hsv,(10,150,70),(40,255,255));mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
 gray=cv2.GaussianBlur(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(9,9),2)
 found=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1.2,minDist=30,param1=90,param2=25,minRadius=30,maxRadius=65)
 candidates=[]
 if found is not None:
  yy,xx=np.ogrid[:crop.shape[0],:crop.shape[1]]
  for x,y,r in found[0]:
   ax,ay=float(x+x0),float(y+y0)
   if not (740<=ax<=960 and 530<=ay<=640):continue
   disk=(xx-x)**2+(yy-y)**2<=r*r
   purity=float(np.mean(mask[disk]>0)) if np.any(disk) else 0
   if purity>=.25:
    candidates.append((ax,ay,float(r),purity))
 # Select a stable, plausible two-circle pair from HSV-confirmed Hough candidates.
 pairs=[]
 for i,a in enumerate(candidates):
  for b in candidates[i+1:]:
   gap=math.dist(a[:2],b[:2]);dy=abs(a[1]-b[1])
   if 20<=gap<=95 and dy<=38:
    score=abs((a[0]+b[0])*.5-845)+1.5*dy+.35*abs(a[2]-b[2])
    pairs.append((score,a,b))
 if pairs:
  _,a,b=min(pairs,key=lambda x:x[0]);chosen=sorted((a,b),key=lambda z:z[0])
 elif candidates:
  chosen=[min(candidates,key=lambda z:abs(z[0]-845)+2*abs(z[1]-585))]
 else:chosen=[]
 ms=(time.perf_counter()-t)*1000
 detections=[Detection(round(x-r),round(y-r),round(x+r),round(y+r),0,max(.05,min(.99,purity))) for x,y,r,purity in chosen]
 return detections,ms,{'raw_hough_candidates':len(candidates),'selected':len(chosen),'circles':[{'x':round(x,1),'y':round(y,1),'r':round(r,1),'hsv_purity':round(p,3)} for x,y,r,p in chosen]}

def yolo_anchor_map():
 out={}
 for row in YOLO['per_sample']:
  cand=[]
  for b in row['boxes']:
   cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
   if b['confidence']>=.05 and 700<=cx<=1000 and 480<=cy<=700:cand.append(b)
  cand.sort(key=lambda b:b['confidence'],reverse=True)
  out[row['frame']]=[Detection(b['x1'],b['y1'],b['x2'],b['y2'],0,b['confidence']) for b in cand[:2]]
 return out

def run_variant(name,anchor_provider,anchor_ms=None,anchor_frames=None):
 cap=cv2.VideoCapture(str(VIDEO));tracker=FuelTracker();records=[];advance_ms=[];update_ms=[];det_ms=[];anchor_counts={};flow_started=time.perf_counter();frame_i=0;started=time.perf_counter()
 while True:
  ok,frame=cap.read()
  if not ok:break
  left=frame[:,:1600];now=frame_i/FPS
  t=time.perf_counter();tracks=tracker.advance(left,now,frame_i);advance_ms.append((time.perf_counter()-t)*1000)
  anchor_info=None
  if (frame_i in anchor_frames) if anchor_frames is not None else (frame_i%STEP==0):
   dets,dm,anchor_info=anchor_provider(frame_i,left);det_ms.append(dm);anchor_counts[len(dets)]=anchor_counts.get(len(dets),0)+1
   t=time.perf_counter();tracks=tracker.update(dets,now,frame_i);update_ms.append((time.perf_counter()-t)*1000)
  tracks=tracker.snapshot(now)
  records.append({'frame':frame_i,'time_s':round(now,4),'track_count':len(tracks),'tracks':[{'id':x['track_id'],'state':x['state'],'confidence':round(x['confidence'],3),'center_xy_px':[round(x['box'][0],1),round(x['box'][1],1)]} for x in tracks],**({'anchor':anchor_info} if anchor_info else {})})
  frame_i+=1
 cap.release();wall=time.perf_counter()-started
 counts=[r['track_count'] for r in records]
 flow_m=median(advance_ms);up_m=median(update_ms);det_m=median(det_ms)
 # Serial CPU budget estimate; excludes decode, stereo, NT, GPU contention, and async overlap.
 budget=flow_m+up_m/STEP+(det_m/STEP if name.startswith('hsv') else (anchor_ms or 0)/STEP)
 return {'name':name,'video_frames':frame_i,'anchor_hz':round(len(anchor_counts)/(frame_i/FPS),2) if frame_i else 0,'anchor_count_histogram':{str(k):v for k,v in sorted(anchor_counts.items())},
  'frames_with_zero_tracks':counts.count(0),'frames_with_one_track':counts.count(1),'frames_with_two_tracks':counts.count(2),'frames_with_more_than_two_tracks':sum(c>2 for c in counts),
  'output_position_rate_hz':round(FPS,2),'flow_advance_latency_ms':{'median':flow_m,'p95':percentile(advance_ms,95)},'kalman_update_latency_ms':{'median':up_m,'p95':percentile(update_ms,95)},'anchor_detection_latency_ms':{'median':det_m,'p95':percentile(det_ms,95)},
  'serial_cpu_budget_ms_per_frame_estimate':round(budget,3),'serial_stage_rate_hz_estimate':round(1000/budget,1) if budget>0 else None,'replay_wall_s':round(wall,2),'replay_wall_fps':round(frame_i/wall,2),'per_frame':records}

anchors=yolo_anchor_map();
def yolo_provider(idx,frame):return anchors.get(idx,[]),anchor_ms,None
hough_anchor_samples=[]
def hough_provider(idx,frame):
 result=hsv_hough(frame);hough_anchor_samples.append(result[2]);return result
anchor_ms=YOLO['latency_ms']['median']
print('running YOLO anchor + tracker replay')
anchor_frames=set(anchors)
yolo=run_variant('yolo20hz+lk_kalman60hz',yolo_provider,anchor_ms,anchor_frames)
print({k:v for k,v in yolo.items() if k!='per_frame'})
print('running HSV/Hough anchor + tracker replay')
hough=run_variant('hsv_hough20hz+lk_kalman60hz',hough_provider,anchor_frames=anchor_frames)
print({k:v for k,v in hough.items() if k!='per_frame'})
# Counts and radius-separation evidence from actual Hough anchors.
raw_hist={}
for row in hough_anchor_samples:
 key=str(row['raw_hough_candidates']);raw_hist[key]=raw_hist.get(key,0)+1
out={'method':'Use cached TensorRT boxes every 3 frames for the current anchor baseline. In the candidate branch, run HSV color confirmation plus local grayscale Hough circle fitting every 3 frames. Both branches use the production FuelTracker Kalman correction and per-frame sparse LK flow on every decoded left frame. The Hough branch uses a fixed ROI for this stationary clip.','fps':FPS,'anchor_interval_frames':STEP,'flow_tracker':{'flow_scale':.5,'max_age_ms':500,'flow_grace_ms':120,'confidence_decay_ms':250},
 'hough_parameters':{'hsv_hue':[10,40],'saturation_min':150,'value_min':70,'hough_param1':90,'hough_param2':25,'radius_px':[30,65],'min_distance_px':30,'candidate_hsv_disk_purity_min':.25,'roi_xyxy_px':ROI},'hsv_confirmed_hough_candidate_count_histogram':raw_hist,
 'baseline_active_runtime_note':'The live production size gate rejected the one merged YOLO box in this heavy-occlusion capture, resulting in no live tracks. The offline YOLO+tracker branch below intentionally bypasses that gate to isolate whether optical flow/Kalman can recover object count from merged detector anchors.',
 'variants':[yolo,hough],
 'limitations':['Stationary camera and pair only; the Hough region is fixed in the image.','HSV/Hough circle centers are exploratory pseudo-measurements, not independent ground-truth labels.','The comparison measures track count and processing frequency, not metric range or motion accuracy.','The replay-rate estimate is CPU stage arithmetic and excludes full camera capture, stereo, NetworkTables, decode overhead, GPU contention, and concurrent scheduling.']}
path=ROOT/'media/frc-fuel-hsv-hough-flow-hybrid-20261103.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
