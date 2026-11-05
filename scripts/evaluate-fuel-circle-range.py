#!/usr/bin/env python3
"""Evaluate detector-centered Hough and HSV contour diameter estimates."""
import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CAL=ROOT/'media/calibration/stereo_calibration_live_order_candidate.yml'
VIDEOS=('media/frc-2026-live-two-fuel-60fps-cfr.mp4',
        'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4')
TIMES=(1,1.4,1.8,2.2,2.6,3,3.4,3.8,4.2,4.6,5,5.4,5.8,6.2,6.6)
TARGETS=((808,589),(902,591))
PARAM2=(20,25,30,35,40,45)
TIGHT_HOUGH=(('tight45_60_off20_p2_20',20,45,60,20),
             ('tight45_60_off20_p2_25',25,45,60,20),
             ('tight45_60_off20_p2_30',30,45,60,20),
             ('tight48_58_off15_p2_20',20,48,58,15),
             ('tight48_58_off15_p2_25',25,48,58,15),
             ('tight48_58_off15_p2_30',30,48,58,15))
HSV_CONFIGS=(('h8_35_s100_v50',(8,35,100,50)),
             ('h12_32_s120_v60',(12,32,120,60)),
             ('h15_28_s150_v60',(15,28,150,60)))


def med(a):return statistics.median(a) if a else None
def mad(a):
 m=med(a);return med([abs(x-m) for x in a]) if a else None

def summarize(name,video,target,rows):
 ranges=[x['range_m'] for x in rows if x.get('range_m') is not None]
 radii=[x['radius_px'] for x in rows if x.get('radius_px') is not None]
 offsets=[x['center_offset_px'] for x in rows if x.get('center_offset_px') is not None]
 return {'video':video,'target':target+1,'method':name,'detections':len(ranges),
         'range_m':{'median':round(med(ranges),3),'mad':round(mad(ranges),3),
                    'min':round(min(ranges),3),'max':round(max(ranges),3)} if ranges else None,
         'radius_px':{'median':round(med(radii),2),'mad':round(mad(radii),2)} if radii else None,
         'center_offset_px_median':round(med(offsets),2) if offsets else None}

fs=cv2.FileStorage(str(CAL),cv2.FILE_STORAGE_READ);focal=float(fs.getNode('left_camera_matrix').mat()[0,0]);fs.release()
results=[]
for video in VIDEOS:
 cap=cv2.VideoCapture(str(ROOT/video))
 hough_rows={ (p2,t):[] for p2 in PARAM2 for t in range(2)}
 tight_rows={ (name,t):[] for name,_,_,_,_ in TIGHT_HOUGH for t in range(2)}
 hsv_rows={ (name,t):[] for name,_ in HSV_CONFIGS for t in range(2)}
 for time_s in TIMES:
  cap.set(cv2.CAP_PROP_POS_MSEC,time_s*1000);ok,frame=cap.read()
  if not ok:continue
  left=frame[:,:1600];gray=cv2.cvtColor(left,cv2.COLOR_BGR2GRAY);hsv=cv2.cvtColor(left,cv2.COLOR_BGR2HSV)
  for target_i,(cx,cy) in enumerate(TARGETS):
   x0,y0=cx-80,cy-80;crop=gray[y0:cy+80,x0:cx+80]
   blurred=cv2.GaussianBlur(crop,(9,9),2)
   for p2 in PARAM2:
    circles=cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,dp=1.2,minDist=30,param1=90,
                              param2=p2,minRadius=35,maxRadius=65)
    candidates=[]
    if circles is not None:
     for x,y,r in circles[0]:
      ox=x+ x0-cx;oy=y+y0-cy;offset=math.hypot(ox,oy)
      if offset<=35:candidates.append((offset,float(r)))
    if candidates:
     offset,r=min(candidates)
     full_r=r
     hough_rows[(p2,target_i)].append({'radius_px':full_r,'center_offset_px':offset,
                                        'range_m':focal*.15/(2*full_r)})
   for name,p2,minr,maxr,maxoff in TIGHT_HOUGH:
    circles=cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,dp=1.2,minDist=30,param1=90,
                              param2=p2,minRadius=minr,maxRadius=maxr)
    candidates=[]
    if circles is not None:
     for x,y,r in circles[0]:
      offset=math.hypot(x+x0-cx,y+y0-cy)
      if offset<=maxoff:candidates.append((offset,float(r)))
    if candidates:
     offset,r=min(candidates)
     tight_rows[(name,target_i)].append({'radius_px':r,'center_offset_px':offset,
                                         'range_m':focal*.15/(2*r)})
   color_crop=hsv[y0:cy+80,x0:cx+80]
   for name,(hlo,hhi,smin,vmin) in HSV_CONFIGS:
    mask=cv2.inRange(color_crop,np.array([hlo,smin,vmin],np.uint8),np.array([hhi,255,255],np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
     area=cv2.contourArea(contour)
     if area<1400:continue
     (x,y),r=cv2.minEnclosingCircle(contour);offset=math.hypot(x+x0-cx,y+y0-cy)
     if 30<=r<=75 and offset<=45:candidates.append((offset,float(r),area))
    if candidates:
     offset,r,area=min(candidates,key=lambda a:a[0])
     hsv_rows[(name,target_i)].append({'radius_px':r,'center_offset_px':offset,
                                       'area_px2':area,'range_m':focal*.15/(2*r)})
 cap.release()
 for p2 in PARAM2:
  for target_i in range(2):results.append(summarize(f'HoughROI-param2-{p2}',video,target_i,hough_rows[(p2,target_i)]))
 for name,_,_,_,_ in TIGHT_HOUGH:
  for target_i in range(2):results.append(summarize(f'HoughROI-{name}',video,target_i,tight_rows[(name,target_i)]))
 for name,_ in HSV_CONFIGS:
  for target_i in range(2):results.append(summarize(f'HSVContour-{name}',video,target_i,hsv_rows[(name,target_i)]))
output={'method':'Use the same fixed target centers and 160x160 detector-seeded patches on 15 frames per saved clip. Hough searches radius 35-65 px, choosing the nearest circle within 35 px. HSV masks search local contours, retaining area >=1400 px² and circle radius 30-75 px within 45 px. Convert apparent radius to nominal range using official 0.15 m FUEL diameter and unrectified left focal length. These are geometry proxies, not ground truth.',
        'calibration':{'file':'media/calibration/stereo_calibration_live_order_candidate.yml','focal_px':focal,'fuel_diameter_m':0.15},
        'results':results}
(ROOT/'media/frc-fuel-circle-range-sweep.json').write_text(json.dumps(output,indent=2)+'\n')
for r in results:print(r)
