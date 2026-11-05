#!/usr/bin/env python3
"""Compare localized SGBM compute with production full-frame SGBM on saved clips."""
import cv2, json, math, statistics, time
import numpy as np
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
D=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
S=json.load(open(ROOT/'media/frc-fuel-dense-stereo-evaluation.json'))
fs=cv2.FileStorage(str(ROOT/'media/calibration/stereo_calibration_live_order_candidate.yml'),cv2.FILE_STORAGE_READ)
kl=fs.getNode('left_camera_matrix').mat();dl=fs.getNode('left_distortion').mat()
kr=fs.getNode('right_camera_matrix').mat();dr=fs.getNode('right_distortion').mat()
rot=fs.getNode('rotation').mat();trans=fs.getNode('translation').mat();fs.release()
size=(1600,1200);r1,r2,p1,p2,*_=cv2.stereoRectify(kl,dl,kr,dr,size,rot,trans,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
ml=cv2.initUndistortRectifyMap(kl,dl,r1,p1,size,cv2.CV_32FC1)
mr=cv2.initUndistortRectifyMap(kr,dr,r2,p2,size,cv2.CV_32FC1)
matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=5,P1=200,P2=800,uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
BASE=.0635;FOCAL=float(p1[0,0]);CONF=.05;CENTERS=((808,589),(902,591));PADS=(24,48);HORIZONTAL_SCOPES=('full_width',)
def med(xs):return statistics.median(xs) if xs else None
def summary(xs):
 if not xs:return None
 m=med(xs);return {'n':len(xs),'median_m':round(m,4),'mad_m':round(med([abs(x-m) for x in xs]),4),'p05_m':round(sorted(xs)[int(.05*(len(xs)-1))],4),'p95_m':round(sorted(xs)[int(.95*(len(xs)-1))],4)}
def match(row):
 out=[None,None]
 for b in sorted((b for b in row['detections'] if b['confidence']>=CONF),key=lambda b:b['confidence'],reverse=True):
  cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
  ds=[math.hypot(cx-x,cy-y) for x,y in CENTERS];i=int(np.argmin(ds))
  if ds[i]<=60 and out[i] is None:out[i]=b
 return out
out={'method':'Rectify and downsample as production. Compare production full-frame SGBM box median range with SGBM on a full-width vertical band covering the matched FUEL boxes and a configurable top/bottom context pad. Crop size, matcher-only latency, valid depth, range deltas, and pair spacing are recorded.','baseline_m':BASE,'num_disparities_half_px':128,'vertical_pad_half_px':list(PADS),'horizontal_scope':list(HORIZONTAL_SCOPES),'results':[]}
for ci,clip in enumerate(D['results']):
 cap=cv2.VideoCapture(str(ROOT/clip['video']))
 records={(pad,scope):{'ranges':[[],[]],'deltas':[[],[]],'spacings':[],'xyz_spacings':[],'latency_ms':[],'shape':[]}
          for pad in PADS for scope in HORIZONTAL_SCOPES}
 for row,srow in zip(clip['per_frame'],S['results'][ci]['frames']):
  ok,frame=cap.read()
  if not ok:break
  left,right=frame[:,:1600],frame[:,1600:]
  gl=cv2.cvtColor(cv2.remap(left,*ml,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY)
  gr=cv2.cvtColor(cv2.remap(right,*mr,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY)
  gl=cv2.resize(gl,(800,600),interpolation=cv2.INTER_AREA);gr=cv2.resize(gr,(800,600),interpolation=cv2.INTER_AREA)
  boxes=match(row)
  if not all(boxes):continue
  rect_boxes=[]
  for b in boxes:
   corners=np.array([[[b['x1'],b['y1']]],[[b['x2'],b['y1']]],[[b['x2'],b['y2']]],[[b['x1'],b['y2']]]],np.float32)
   q=cv2.undistortPoints(corners,kl,dl,R=r1,P=p1).reshape(-1,2)/2
   rect_boxes.append((float(q[:,0].min()),float(q[:,1].min()),float(q[:,0].max()),float(q[:,1].max())))
  xs=[(b[0],b[2]) for b in rect_boxes];ys=[(b[1],b[3]) for b in rect_boxes]
  # Search context is needed to compare left pixels against right pixels up to 128 px away.
  for scope in HORIZONTAL_SCOPES:
   x0=0;x1=800
   for pad in PADS:
    y0=max(0,int(math.floor(min(y[0] for y in ys)))-pad)
    y1=min(600,int(math.ceil(max(y[1] for y in ys)))+pad)
    lroi=gl[y0:y1,x0:x1];rroi=gr[y0:y1,x0:x1]
    if lroi.shape[1]<=128 or lroi.shape[0]<5:continue
    t=time.perf_counter()
    disp=matcher.compute(lroi,rroi).astype(np.float32)/16.
    rec=records[(pad,scope)];rec['latency_ms'].append((time.perf_counter()-t)*1000)
    rec['shape'].append(lroi.shape[0]*lroi.shape[1]);ranges=[]
    for ti,b in enumerate(rect_boxes):
     bx1=max(0,int(math.floor(b[0]))-x0);bx2=min(disp.shape[1],int(math.ceil(b[2]))-x0)
     by1=max(0,int(math.floor(b[1]))-y0);by2=min(disp.shape[0],int(math.ceil(b[3]))-y0)
     roi=disp[by1:by2,bx1:bx2];vals=roi[np.isfinite(roi)&(roi>3)&(roi<120)]
     z=BASE*FOCAL/(2*float(np.median(vals))) if vals.size>=8 else None
     reference=srow['ranges_m'][ti]
     if z is not None:
      rec['ranges'][ti].append(z)
      if reference is not None:rec['deltas'][ti].append(z-reference)
     ranges.append(z)
    if all(z is not None for z in ranges):
     rec['spacings'].append(abs(ranges[0]-ranges[1]));points=[]
     for b,z in zip(boxes,ranges):
      u,v=cv2.undistortPoints(np.array([[[((b['x1']+b['x2'])/2),((b['y1']+b['y2'])/2)]]],np.float32),kl,dl,R=r1,P=p1)[0,0]
      points.append(((u-p1[0,2])*z/p1[0,0],(v-p1[1,2])*z/p1[1,1],z))
     rec['xyz_spacings'].append(math.dist(points[0],points[1]))
 cap.release();entry={'scene':clip['scene'],'frames_in_video':600,'configs':[]}
 for (pad,scope),rec in records.items():
  lat=rec['latency_ms'];entry['configs'].append({'vertical_pad_half_px':pad,'horizontal_scope':scope,'valid_ranges':[summary(a) for a in rec['ranges']],'range_delta_vs_full_sgbm_m':[summary(a) for a in rec['deltas']],'abs_depth_gap_between_balls_m':summary(rec['spacings']),'xyz_pair_spacing_m':summary(rec['xyz_spacings']),'matcher_ms_median':round(med(lat),3) if lat else None,'matcher_ms_p95':round(sorted(lat)[int(.95*(len(lat)-1))],3) if lat else None,'mean_crop_pixels':round(med(rec['shape'])) if rec['shape'] else None,'valid_pair_frames':len(rec['spacings'])});print(clip['scene'],entry['configs'][-1])
 out['results'].append(entry)
path=ROOT/'media/frc-fuel-stereo-roi-compute-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
