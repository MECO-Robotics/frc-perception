#!/usr/bin/env python3
"""Test yellow-FUEL color masks as stereo-depth ROIs on the saved clips."""
import cv2,json,math,statistics,time
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
dense=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
fs=cv2.FileStorage(str(ROOT/'media/calibration/stereo_calibration_live_order_candidate.yml'),cv2.FILE_STORAGE_READ)
kl=fs.getNode('left_camera_matrix').mat();dl=fs.getNode('left_distortion').mat();kr=fs.getNode('right_camera_matrix').mat();dr=fs.getNode('right_distortion').mat();rot=fs.getNode('rotation').mat();trans=fs.getNode('translation').mat();fs.release()
size=(1600,1200);r1,r2,p1,p2,*_=cv2.stereoRectify(kl,dl,kr,dr,size,rot,trans,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
ml=cv2.initUndistortRectifyMap(kl,dl,r1,p1,size,cv2.CV_32FC1);mr=cv2.initUndistortRectifyMap(kr,dr,r2,p2,size,cv2.CV_32FC1)
matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=5,P1=200,P2=800,uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
base=.0635;focal=float(p1[0,0]); sat_thresholds=(40,70,100)
def med(a):return statistics.median(a) if a else None
def summary(a):
 if not a:return None
 c=med(a);return {'n':len(a),'median':round(c,4),'mad':round(med([abs(x-c) for x in a]),4),'p05':round(sorted(a)[int(.05*(len(a)-1))],4),'p95':round(sorted(a)[int(.95*(len(a)-1))],4)}
def match(items):
 out=[None,None]
 for b in sorted((d for d in items if d['confidence']>=.1),key=lambda d:d['confidence'],reverse=True):
  cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
  ds=[math.hypot(cx-x,cy-y) for x,y in ((808,589),(902,591))];i=int(np.argmin(ds))
  if ds[i]<=60 and out[i] is None:out[i]=b
 return out
out={'method':'Compare median disparity over full detector box vs HSV yellow masks (OpenCV hue 10-40, saturation threshold, value >=40).','baseline_m':base,'hue_range_opencv':[10,40],'value_min':40,'results':[]}
for clip in dense['results']:
 cap=cv2.VideoCapture(str(ROOT/clip['video'])); records={str(s):{'ranges':[[],[]],'spacing':[],'valid_targets':0} for s in (0,*sat_thresholds)}
 for idx,row in enumerate(clip['per_frame']):
  ok,frame=cap.read()
  if not ok:break
  left,right=frame[:,:1600],frame[:,1600:]
  cl=cv2.remap(left,*ml,cv2.INTER_LINEAR);cr=cv2.remap(right,*mr,cv2.INTER_LINEAR)
  hsv=cv2.cvtColor(cl,cv2.COLOR_BGR2HSV);gl=cv2.cvtColor(cl,cv2.COLOR_BGR2GRAY);gr=cv2.cvtColor(cr,cv2.COLOR_BGR2GRAY)
  gl=cv2.resize(gl,(800,600),interpolation=cv2.INTER_AREA);gr=cv2.resize(gr,(800,600),interpolation=cv2.INTER_AREA)
  disp=matcher.compute(gl,gr).astype(np.float32)/16.
  pts_by={str(s):[] for s in (0,*sat_thresholds)}
  for ti,b in enumerate(match(row['detections'])):
   if b is None:
    for s in pts_by:pts_by[s].append(None)
    continue
   cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
   corners=np.array([[[b['x1'],b['y1']],[b['x2'],b['y1']],[b['x2'],b['y2']],[b['x1'],b['y2']]]],np.float32)
   q=cv2.undistortPoints(corners,kl,dl,R=r1,P=p1).reshape(-1,2)*.5
   x1,y1=np.floor(q.min(0)).astype(int);x2,y2=np.ceil(q.max(0)).astype(int)
   x1=max(0,x1);y1=max(0,y1);x2=min(800,x2);y2=min(600,y2)
   dbox=disp[y1:y2,x1:x2]; hsvbox=hsv[max(0,int(np.floor(q[:,1].min()*2))):min(1200,int(np.ceil(q[:,1].max()*2))),max(0,int(np.floor(q[:,0].min()*2))):min(1600,int(np.ceil(q[:,0].max()*2)))]
   # Use the rectified detector box bounds at full resolution for the color mask.
   rx1=max(0,int(np.floor(q[:,0].min()*2)));rx2=min(1600,int(np.ceil(q[:,0].max()*2)));ry1=max(0,int(np.floor(q[:,1].min()*2)));ry2=min(1200,int(np.ceil(q[:,1].max()*2)))
   hsvbox=hsv[ry1:ry2,rx1:rx2]
   for s_key in pts_by:
    s=int(s_key)
    valid=np.isfinite(dbox)&(dbox>3)&(dbox<120)
    if s:
     mask=(hsvbox[:,:,0]>=10)&(hsvbox[:,:,0]<=40)&(hsvbox[:,:,1]>=s)&(hsvbox[:,:,2]>=40)
     mask=cv2.resize(mask.astype(np.uint8),(dbox.shape[1],dbox.shape[0]),interpolation=cv2.INTER_NEAREST).astype(bool)
     valid &= mask
    vals=dbox[valid]
    if vals.size<8:pts_by[s].append(None);continue
    z=base*focal/(2*float(np.median(vals)));u,v=cv2.undistortPoints(np.array([[[cx,cy]]],np.float32),kl,dl,R=r1,P=p1)[0,0]
    pts_by[s_key].append(((u-p1[0,2])*z/p1[0,0],(v-p1[1,2])*z/p1[1,1],z))
  for s,pts in pts_by.items():
   rec=records[s]
   for ti,p in enumerate(pts):
    if p is not None:rec['ranges'][ti].append(p[2]);rec['valid_targets']+=1
   if len(pts)==2 and all(p is not None for p in pts):rec['spacing'].append(math.dist(*pts))
 cap.release()
 clipout={'scene':clip['scene'],'configs':[]}
 for s,rec in records.items():
  line={'saturation_min':int(s),'valid_target_samples':rec['valid_targets'],'ranges_m':[summary(a) for a in rec['ranges']],'pair_spacing_m':summary(rec['spacing'])}
  clipout['configs'].append(line);print(clip['scene'],line)
 out['results'].append(clipout)
path=ROOT/'media/frc-fuel-color-depth-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
