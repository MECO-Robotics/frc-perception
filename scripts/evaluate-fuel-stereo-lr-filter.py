#!/usr/bin/env python3
"""Test left-right consistency filtering before stereo range aggregation."""
import json,math,statistics,time
from pathlib import Path
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'media/frc-fuel-dense-count-evaluation.json').read_text())
CAL=ROOT/'media/calibration/stereo_calibration_live_order_candidate.yml'
TARGETS=((808,589),(902,591));BASE=.0635;PAD=24;STEP=5
THRESHOLDS=(None,.5,1.0,1.5,2.0)
fs=cv2.FileStorage(str(CAL),cv2.FILE_STORAGE_READ)
kl=fs.getNode('left_camera_matrix').mat();dl=fs.getNode('left_distortion').mat()
kr=fs.getNode('right_camera_matrix').mat();dr=fs.getNode('right_distortion').mat()
rot=fs.getNode('rotation').mat();trans=fs.getNode('translation').mat();fs.release()
size=(1600,1200);r1,r2,p1,p2,*_=cv2.stereoRectify(kl,dl,kr,dr,size,rot,trans,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
ml=cv2.initUndistortRectifyMap(kl,dl,r1,p1,size,cv2.CV_32FC1);mr=cv2.initUndistortRectifyMap(kr,dr,r2,p2,size,cv2.CV_32FC1)
focal=float(p1[0,0])
forward=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=5,P1=200,P2=800,
    uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,disp12MaxDiff=-1,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
reverse=cv2.StereoSGBM_create(minDisparity=-128,numDisparities=128,blockSize=5,P1=200,P2=800,
    uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,disp12MaxDiff=-1,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

def match(row):
    hits=[None,None]
    for b in sorted((b for b in row if b['confidence']>=.05),key=lambda b:-b['confidence']):
        x=(b['x1']+b['x2'])/2;y=(b['y1']+b['y2'])/2
        d=[math.hypot(x-a,y-c) for a,c in TARGETS];i=int(np.argmin(d))
        if d[i]<=60 and hits[i] is None:hits[i]=b
    return hits

def rect(b):
    c=np.array([[[b['x1'],b['y1']], [b['x2'],b['y1']], [b['x2'],b['y2']], [b['x1'],b['y2']]]],np.float32)
    q=cv2.undistortPoints(c,kl,dl,R=r1,P=p1).reshape(-1,2)*.5
    return max(0,int(np.floor(q[:,0].min()))),max(0,int(np.floor(q[:,1].min()))),min(800,int(np.ceil(q[:,0].max()))),min(600,int(np.ceil(q[:,1].max())))

def point(b,z):
    c=np.array([[[float(b['x1']+b['x2'])/2,float(b['y1']+b['y2'])/2]]],np.float32)
    u,v=cv2.undistortPoints(c,kl,dl,R=r1,P=p1)[0,0]
    return ((u-p1[0,2])*z/p1[0,0],(v-p1[1,2])*z/p1[1,1],z)

def summary(v):
    if not v:return {'n':0}
    m=statistics.median(v)
    return {'n':len(v),'median':round(m,5),'mad':round(statistics.median(abs(x-m) for x in v),5),
            'p05':round(float(np.percentile(v,5)),5),'p95':round(float(np.percentile(v,95)),5)}

out={'method':'Every fifth saved frame; compute forward and reverse active SGBM on the same rectified vertical band. For each box, compare unfiltered median disparity with per-pixel left-right consistency tolerances in half-resolution pixels.',
     'limitations':'Two static scenes only, no absolute range truth. Reverse-match consistency thresholds are exploratory and need real motion/occlusion validation.',
     'thresholds_half_resolution_px':THRESHOLDS,'sample_stride_frames':STEP,'results':[]}
for clip in DATA['results']:
    cap=cv2.VideoCapture(str(ROOT/clip['video']))
    acc={t:{'ranges':[[],[]],'spacing':[],'valid_targets':0,'samples':[0,0],'keep_fraction':[[],[]],'forward_ms':[],'reverse_ms':[]} for t in THRESHOLDS}
    decoded=0;start=time.perf_counter()
    while True:
        ok,frame=cap.read()
        if not ok:break
        if decoded%STEP==0:
            boxes=match(clip['per_frame'][decoded]['detections'])
            candidates=[b for b in boxes if b]
            if candidates:
                rects=[rect(b) for b in candidates];y0=max(0,min(r[1] for r in rects)-PAD);y1=min(600,max(r[3] for r in rects)+PAD)
                left=cv2.remap(frame[:,:1600],*ml,cv2.INTER_LINEAR);right=cv2.remap(frame[:,1600:],*mr,cv2.INTER_LINEAR)
                gl=cv2.resize(cv2.cvtColor(left,cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
                gr=cv2.resize(cv2.cvtColor(right,cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
                t=time.perf_counter();fl=forward.compute(gl[y0:y1],gr[y0:y1]).astype(np.float32)/16;ft=(time.perf_counter()-t)*1000
                t=time.perf_counter();fr=reverse.compute(gr[y0:y1],gl[y0:y1]).astype(np.float32)/16;rt=(time.perf_counter()-t)*1000
                h,w=fl.shape;xx=np.broadcast_to(np.arange(w,dtype=np.float32),(h,w));xr=np.rint(xx-fl).astype(np.int32)
                inbounds=(xr>=0)&(xr<w);safe=np.clip(xr,0,w-1);rows=np.broadcast_to(np.arange(h)[:,None],(h,w))
                lr_err=np.full(fl.shape,np.inf,np.float32);lr_err[inbounds]=np.abs(fl[inbounds]+fr[rows[inbounds],safe[inbounds]])
                for thresh,a in acc.items():
                    a['forward_ms'].append(ft);a['reverse_ms'].append(rt);zs=[];pts=[]
                    for ti,b in enumerate(boxes):
                        if b is None:zs.append(None);pts.append(None);continue
                        a['samples'][ti]+=1;x1,yy1,x2,yy2=rect(b)
                        roi=fl[max(0,yy1-y0):max(0,yy2-y0),x1:x2]
                        er=lr_err[max(0,yy1-y0):max(0,yy2-y0),x1:x2]
                        valid=np.isfinite(roi)&(roi>3)&(roi<120)
                        before=int(valid.sum())
                        if thresh is not None:valid &= er<=thresh
                        vals=roi[valid];a['keep_fraction'][ti].append(float(vals.size/before) if before else 0.0)
                        z=BASE*focal/(2*float(np.median(vals))) if vals.size>=8 else None
                        zs.append(z);pts.append(point(b,z) if z is not None else None)
                        if z is not None:a['ranges'][ti].append(z);a['valid_targets']+=1
                    if all(z is not None for z in zs):a['spacing'].append(math.dist(*pts))
        decoded+=1
    cap.release();configs=[]
    for thresh,a in acc.items():
        configs.append({'lr_tolerance_half_px':thresh,'sampled_box_targets':a['samples'],'valid_target_ranges':a['valid_targets'],
            'ranges_m_by_target':[summary(v) for v in a['ranges']],'pair_spacing_m':summary(a['spacing']),
            'kept_fraction_by_target':[summary(v) for v in a['keep_fraction']],
            'forward_ms_median':round(statistics.median(a['forward_ms']),2) if a['forward_ms'] else None,
            'reverse_ms_median':round(statistics.median(a['reverse_ms']),2) if a['reverse_ms'] else None})
    out['results'].append({'scene':clip['scene'],'decoded_frames':decoded,'elapsed_s':round(time.perf_counter()-start,2),'configs':configs})
    print(clip['scene'],json.dumps(configs))
path=ROOT/'media/frc-fuel-stereo-lr-filter-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
