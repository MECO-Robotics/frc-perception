#!/usr/bin/env python3
"""Compare SGBM uniqueness ratios on every fifth frame of the saved FUEL stereo clips."""
import cv2,json,math,statistics,time
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
dense=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
fs=cv2.FileStorage(str(ROOT/'media/calibration/stereo_calibration_live_order_candidate.yml'),cv2.FILE_STORAGE_READ)
kl=fs.getNode('left_camera_matrix').mat();dl=fs.getNode('left_distortion').mat();kr=fs.getNode('right_camera_matrix').mat();dr=fs.getNode('right_distortion').mat();rot=fs.getNode('rotation').mat();trans=fs.getNode('translation').mat();fs.release()
size=(1600,1200);r1,r2,p1,p2,*_=cv2.stereoRectify(kl,dl,kr,dr,size,rot,trans,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
ml=cv2.initUndistortRectifyMap(kl,dl,r1,p1,size,cv2.CV_32FC1);mr=cv2.initUndistortRectifyMap(kr,dr,r2,p2,size,cv2.CV_32FC1)
base=.0635;focal=float(p1[0,0]);sample_step=5

def med(a):return statistics.median(a) if a else None
def stats(a):
 if not a:return None
 c=med(a);return {'n':len(a),'median_m':round(c,4),'mad_m':round(med([abs(x-c) for x in a]),4),'p05_m':round(sorted(a)[int(.05*(len(a)-1))],4),'p95_m':round(sorted(a)[int(.95*(len(a)-1))],4)}
def match(items):
 out=[None,None]
 for b in sorted((d for d in items if d['confidence']>=.1),key=lambda d:d['confidence'],reverse=True):
  cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
  ds=[math.hypot(cx-x,cy-y) for x,y in ((808,589),(902,591))];i=int(np.argmin(ds))
  if ds[i]<=60 and out[i] is None:out[i]=b
 return out
out={'method':f'Production rectification; every {sample_step}th frame, fixed confidence 0.10 detections; compare SGBM uniqueness ratio with block size 3 and all other settings fixed.','baseline_m':base,'match_scale':.5,'num_disparities':128,'sample_step':sample_step,'results':[]}
for uniqueness in (4,8,12):
 bs=3
 matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=bs,P1=8*bs*bs,P2=32*bs*bs,uniquenessRatio=uniqueness,speckleWindowSize=50,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
 config={'block_size':bs,'uniqueness_ratio':uniqueness,'clips':[]}
 for clip in dense['results']:
  cap=cv2.VideoCapture(str(ROOT/clip['video']));ranges=[[],[]];spacing=[];times=[];sampled=processed=0;started=time.perf_counter()
  for idx,row in enumerate(clip['per_frame']):
   ok,frame=cap.read()
   if not ok:break
   processed+=1
   if idx%sample_step:continue
   sampled+=1;left,right=frame[:,:1600],frame[:,1600:]
   gl=cv2.cvtColor(cv2.remap(left,*ml,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY);gr=cv2.cvtColor(cv2.remap(right,*mr,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY)
   gl=cv2.resize(gl,(800,600),interpolation=cv2.INTER_AREA);gr=cv2.resize(gr,(800,600),interpolation=cv2.INTER_AREA)
   t0=time.perf_counter();disp=matcher.compute(gl,gr).astype(np.float32)/16.;times.append((time.perf_counter()-t0)*1000)
   pts=[]
   for b in match(row['detections']):
    if b is None:pts.append(None);continue
    corners=np.array([[[b['x1'],b['y1']],[b['x2'],b['y1']],[b['x2'],b['y2']],[b['x1'],b['y2']]]],np.float32)
    q=cv2.undistortPoints(corners,kl,dl,R=r1,P=p1).reshape(-1,2)*.5
    x1,y1=np.floor(q.min(0)).astype(int);x2,y2=np.ceil(q.max(0)).astype(int);x1=max(0,x1);y1=max(0,y1);x2=min(800,x2);y2=min(600,y2)
    roi=disp[y1:y2,x1:x2];valid=roi[np.isfinite(roi)&(roi>3)&(roi<120)]
    if valid.size<8:pts.append(None);continue
    z=base*focal/(2*float(np.median(valid)));cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
    u,v=cv2.undistortPoints(np.array([[[cx,cy]]],np.float32),kl,dl,R=r1,P=p1)[0,0]
    pt=((u-p1[0,2])*z/p1[0,0],(v-p1[1,2])*z/p1[1,1],z);pts.append(pt)
   # Target indices correspond to the two fixed ROIs, including missing values.
   # Add ranges separately without changing index ordering.
   for ti,pt in enumerate(pts):
    if pt is not None:ranges[ti].append(pt[2])
   if len(pts)==2 and all(pt is not None for pt in pts):spacing.append(math.dist(*pts))
  cap.release()
  line={'scene':clip['scene'],'frames_decoded':processed,'sampled_frames':sampled,'range_by_target_m':[stats(x) for x in ranges],'pair_spacing_m':stats(spacing),'stereo_compute_ms_median':round(med(times),2),'stereo_compute_ms_p95':round(sorted(times)[int(.95*(len(times)-1))],2),'elapsed_s':round(time.perf_counter()-started,1)}
  config['clips'].append(line);print('uniqueness',uniqueness,line)
 out['results'].append(config)
path=ROOT/'media/frc-fuel-sgbm-uniqueness-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
