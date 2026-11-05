#!/usr/bin/env python3
"""Compare centered detector-box crops for dense recorded FUEL stereo depth."""
import cv2, json, math, statistics, time
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
dense=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
cal=ROOT/'media/calibration/stereo_calibration_live_order_candidate.yml'
fs=cv2.FileStorage(str(cal),cv2.FILE_STORAGE_READ)
kl=fs.getNode('left_camera_matrix').mat(); dl=fs.getNode('left_distortion').mat()
kr=fs.getNode('right_camera_matrix').mat(); dr=fs.getNode('right_distortion').mat()
r=fs.getNode('rotation').mat(); t=fs.getNode('translation').mat(); fs.release()
size=(1600,1200); r1,r2,p1,p2,*_=cv2.stereoRectify(kl,dl,kr,dr,size,r,t,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
ml=cv2.initUndistortRectifyMap(kl,dl,r1,p1,size,cv2.CV_32FC1); mr=cv2.initUndistortRectifyMap(kr,dr,r2,p2,size,cv2.CV_32FC1)
matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=5,P1=200,P2=800,uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
base=.0635; scale=.5; focal=float(p1[0,0]); fractions=(1.0,.8,.6,.4)
def med(a): return statistics.median(a) if a else None
def stats(a):
 if not a:return None
 return {'n':len(a),'median_m':round(med(a),4),'mad_m':round(med([abs(x-med(a)) for x in a]),4),'p05_m':round(sorted(a)[int(.05*(len(a)-1))],4),'p95_m':round(sorted(a)[int(.95*(len(a)-1))],4)}
def match(items):
 out=[None,None]
 for b in sorted((x for x in items if x['confidence']>=.1),key=lambda x:x['confidence'],reverse=True):
  cx=(b['x1']+b['x2'])/2; cy=(b['y1']+b['y2'])/2
  d,i=min((math.hypot(cx-x,cy-589. if i==0 else cy-591.),i) for i,x in enumerate((808.,902.)))
  if d<=60 and out[i] is None:out[i]=b
 return out
out={'method':'Fixed dense detector boxes and SGBM; compare centered box crops before median disparity aggregation. Baseline 63.5mm, match scale 0.5, search width 128.','baseline_m':base,'fractions':{},'results':[]}
for frac in fractions:
 out['fractions'][str(frac)]={'clips':[]}
for clip in dense['results']:
 cap=cv2.VideoCapture(str(ROOT/clip['video'])); rows=clip['per_frame']; all_records={f:{'ranges':[[],[]],'spacing':[]} for f in fractions}; started=time.perf_counter()
 for idx,row in enumerate(rows):
  ok,frame=cap.read()
  if not ok:break
  boxes=match(row['detections']); left,right=frame[:,:1600],frame[:,1600:]
  gl=cv2.cvtColor(cv2.remap(left,*ml,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY); gr=cv2.cvtColor(cv2.remap(right,*mr,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY)
  gl=cv2.resize(gl,(800,600),interpolation=cv2.INTER_AREA); gr=cv2.resize(gr,(800,600),interpolation=cv2.INTER_AREA)
  disp=matcher.compute(gl,gr).astype(np.float32)/16.
  for frac in fractions:
   pts=[]; rec=all_records[frac]
   for ti,b in enumerate(boxes):
    if b is None: pts.append(None); continue
    cx=(b['x1']+b['x2'])/2; cy=(b['y1']+b['y2'])/2; hw=(b['x2']-b['x1'])*frac/2; hh=(b['y2']-b['y1'])*frac/2
    corners=np.array([[[cx-hw,cy-hh],[cx+hw,cy-hh],[cx+hw,cy+hh],[cx-hw,cy+hh]]],np.float32)
    q=cv2.undistortPoints(corners,kl,dl,R=r1,P=p1).reshape(-1,2)*scale
    x1,y1=np.floor(q.min(0)).astype(int); x2,y2=np.ceil(q.max(0)).astype(int)
    x1=max(0,x1);y1=max(0,y1);x2=min(800,x2);y2=min(600,y2)
    roi=disp[y1:y2,x1:x2]; valid=roi[np.isfinite(roi)&(roi>3)&(roi<120)]
    if valid.size<8:pts.append(None);continue
    d=float(np.median(valid)); z=base*focal/(2*d); center=np.array([[[cx,cy]]],np.float32)
    u,v=cv2.undistortPoints(center,kl,dl,R=r1,P=p1)[0,0]
    point=((u-p1[0,2])*z/p1[0,0],(v-p1[1,2])*z/p1[1,1],z)
    rec['ranges'][ti].append(z);pts.append(point)
   if all(p is not None for p in pts):rec['spacing'].append(math.dist(*pts))
 cap.release()
 for frac in fractions:
  rec=all_records[frac]; entry={'scene':clip['scene'],'range_by_target_m':[stats(x) for x in rec['ranges']],'spacing_m':stats(rec['spacing']),'merged_below_20cm':sum(x<.20 for x in rec['spacing'])}
  out['fractions'][str(frac)]['clips'].append(entry)
  print(clip['scene'],frac,'range',entry['range_by_target_m'],'spacing',entry['spacing_m'],'<20cm',entry['merged_below_20cm'])
print('elapsed',round(time.perf_counter()-started,1))
path=ROOT/'media/frc-fuel-depth-roi-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
