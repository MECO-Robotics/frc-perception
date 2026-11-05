#!/usr/bin/env python3
"""Compare count and stereo range across low-confidence detector cutoffs."""
import json, math, statistics, time
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'media/frc-fuel-low-confidence-evaluation.json').read_text())
CAL=ROOT/'media/calibration/stereo_calibration_live_order_candidate.yml'
TARGETS=((808,589),(902,591)); THRESHOLDS=(.05,.06,.07,.075,.08,.10)
BASE=.0635; PAD=24
fs=cv2.FileStorage(str(CAL),cv2.FILE_STORAGE_READ)
kl=fs.getNode('left_camera_matrix').mat();dl=fs.getNode('left_distortion').mat()
kr=fs.getNode('right_camera_matrix').mat();dr=fs.getNode('right_distortion').mat()
rot=fs.getNode('rotation').mat();trans=fs.getNode('translation').mat();fs.release()
size=(1600,1200)
r1,r2,p1,p2,*_=cv2.stereoRectify(kl,dl,kr,dr,size,rot,trans,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
ml=cv2.initUndistortRectifyMap(kl,dl,r1,p1,size,cv2.CV_32FC1)
mr=cv2.initUndistortRectifyMap(kr,dr,r2,p2,size,cv2.CV_32FC1)
focal=float(p1[0,0])
matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=5,P1=200,P2=800,
    uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,disp12MaxDiff=-1,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

def box_match(row,threshold):
    hits=[None,None];extras=0
    for b in sorted((b for b in row if b['confidence']>=threshold),key=lambda b:-b['confidence']):
        x=(b['x1']+b['x2'])/2;y=(b['y1']+b['y2'])/2
        d=[math.hypot(x-a,y-c) for a,c in TARGETS];i=int(np.argmin(d))
        if d[i]<=60 and hits[i] is None:hits[i]=b
        else:extras+=1
    return hits,extras

def rect(b):
    c=np.array([[[b['x1'],b['y1']]],[[b['x2'],b['y1']]],[[b['x2'],b['y2']]],[[b['x1'],b['y2']]]],np.float32)
    q=cv2.undistortPoints(c,kl,dl,R=r1,P=p1).reshape(-1,2)*.5
    return (max(0,int(np.floor(q[:,0].min()))),max(0,int(np.floor(q[:,1].min()))),
            min(800,int(np.ceil(q[:,0].max()))),min(600,int(np.ceil(q[:,1].max()))))

def point(b,z):
    c=np.array([[[float(b['x1']+b['x2'])/2,float(b['y1']+b['y2'])/2]]],np.float32)
    u,v=cv2.undistortPoints(c,kl,dl,R=r1,P=p1)[0,0]
    return ((u-p1[0,2])*z/p1[0,0],(v-p1[1,2])*z/p1[1,1],z)

def summary(v):
    if not v:return {'n':0}
    m=statistics.median(v)
    return {'n':len(v),'median':round(m,5),'mad':round(statistics.median(abs(x-m) for x in v),5),
            'p05':round(float(np.percentile(v,5)),5),'p95':round(float(np.percentile(v,95)),5)}

out={'method':'Reuse full-frame low-cutoff detector replay outputs; compute one active SGBM 3WAY disparity map per frame in the vertical band and evaluate confidence thresholds .05-.10.',
     'limitations':'Two stationary clips only; no measured distance truth. Repeatability and cluster geometry do not establish absolute range accuracy.',
     'results':[]}
for clip in DATA['results']:
    cap=cv2.VideoCapture(str(ROOT/clip['video']))
    acc={t:{'tp':0,'fp':0,'fn':0,'exact':0,'ranges':[[],[]],'spacing':[],'times':[]} for t in THRESHOLDS}
    for idx,row in enumerate(clip['per_frame']):
        ok,frame=cap.read()
        if not ok:break
        by={t:box_match(row,t) for t in THRESHOLDS}
        candidates=[b for t in THRESHOLDS for b in by[t][0] if b]
        if not candidates:continue
        rects=[rect(b) for b in candidates]
        y0=max(0,min(r[1] for r in rects)-PAD);y1=min(600,max(r[3] for r in rects)+PAD)
        left=cv2.remap(frame[:,:1600],*ml,cv2.INTER_LINEAR)
        right=cv2.remap(frame[:,1600:],*mr,cv2.INTER_LINEAR)
        gl=cv2.resize(cv2.cvtColor(left,cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
        gr=cv2.resize(cv2.cvtColor(right,cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
        t0=time.perf_counter();disp=matcher.compute(gl[y0:y1],gr[y0:y1]).astype(np.float32)/16
        elapsed=(time.perf_counter()-t0)*1000
        for t,(boxes,extras) in by.items():
            a=acc[t];count=sum(b is not None for b in boxes)
            a['tp']+=count;a['fn']+=2-count;a['fp']+=extras;a['exact']+=count==2 and extras==0
            zs=[];pts=[]
            for i,b in enumerate(boxes):
                if b is None:zs.append(None);pts.append(None);continue
                x1,yy1,x2,yy2=rect(b)
                roi=disp[max(0,yy1-y0):max(0,yy2-y0),x1:x2]
                vals=roi[np.isfinite(roi)&(roi>3)&(roi<120)]
                z=BASE*focal/(2*float(np.median(vals))) if vals.size>=8 else None
                zs.append(z);pts.append(point(b,z) if z is not None else None)
                if z is not None:a['ranges'][i].append(z)
            if all(p is not None for p in pts):a['spacing'].append(math.dist(*pts))
            a['times'].append(elapsed)
    cap.release()
    configs=[]
    for t,a in acc.items():
        prec=a['tp']/(a['tp']+a['fp']) if a['tp']+a['fp'] else 0
        rec=a['tp']/(a['tp']+a['fn']) if a['tp']+a['fn'] else 0
        configs.append({'confidence':t,'exact_two_frames':a['exact'],'tp':a['tp'],'fp':a['fp'],'fn':a['fn'],
            'precision':round(prec,5),'recall':round(rec,5),'ranges_m_by_target':[summary(v) for v in a['ranges']],
            'pair_spacing_m':summary(a['spacing']),'sgbm_ms_median':round(statistics.median(a['times']),2) if a['times'] else None})
    out['results'].append({'scene':clip['scene'],'configs':configs})
    print(clip['scene'],json.dumps(configs))
path=ROOT/'media/frc-fuel-confidence-range-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
