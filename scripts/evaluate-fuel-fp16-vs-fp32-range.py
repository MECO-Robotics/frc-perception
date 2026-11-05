#!/usr/bin/env python3
"""Compare FP16 and FP32 detector boxes through the same stereo matcher."""
import json,math,statistics,time
from pathlib import Path
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA={p:json.loads((ROOT/f'media/frc-fuel-dense-count-{p}.json').read_text()) for p in ('evaluation','fp32')}
CAL=ROOT/'media/calibration/stereo_calibration_live_order_candidate.yml'
TARGETS=((808,589),(902,591));BASE=.0635;PAD=24;STEP=1
fs=cv2.FileStorage(str(CAL),cv2.FILE_STORAGE_READ)
kl=fs.getNode('left_camera_matrix').mat();dl=fs.getNode('left_distortion').mat()
kr=fs.getNode('right_camera_matrix').mat();dr=fs.getNode('right_distortion').mat()
rot=fs.getNode('rotation').mat();trans=fs.getNode('translation').mat();fs.release()
size=(1600,1200);r1,r2,p1,p2,*_=cv2.stereoRectify(kl,dl,kr,dr,size,rot,trans,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
ml=cv2.initUndistortRectifyMap(kl,dl,r1,p1,size,cv2.CV_32FC1);mr=cv2.initUndistortRectifyMap(kr,dr,r2,p2,size,cv2.CV_32FC1)
matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=5,P1=200,P2=800,
    uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,disp12MaxDiff=-1,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
focal=float(p1[0,0])

def match(row):
    hits=[None,None];extras=0
    for b in sorted((b for b in row['detections'] if b['confidence']>=.05),key=lambda b:-b['confidence']):
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

out={'method':'Run full-frame saved FP16 and FP32 detections at confidence 0.05 through identical rectification and active SGBM 3WAY/128 disparity/pad24/full-box median range.',
    'limitations':'Same two static clips and fixed labels; no measured distance truth. This compares detector precision effects on localization and stereo repeatability only.',
    'results':[]}
for ci,scene in enumerate(('earlier','corrected-settings')):
    dets={prec:DATA[prec]['results'][ci] for prec in ('evaluation','fp32')}
    cap=cv2.VideoCapture(str(ROOT/dets['evaluation']['video']))
    acc={p:{'exact':0,'tp':0,'fp':0,'fn':0,'center':[[],[]],'ranges':[[],[]],'spacing':[],'times':[]} for p in dets}
    decoded=0;start=time.perf_counter()
    while True:
        ok,frame=cap.read()
        if not ok:break
        if decoded%STEP==0:
            rows={p:dets[p]['per_frame'][decoded]['detections'] for p in dets}
            boxes={p:match({'detections':rows[p]}) for p in dets}
            candidates=[b for pair,_ in boxes.values() for b in pair if b]
            if candidates:
                rects=[rect(b) for b in candidates]
                y0=max(0,min(r[1] for r in rects)-PAD);y1=min(600,max(r[3] for r in rects)+PAD)
                left=cv2.remap(frame[:,:1600],*ml,cv2.INTER_LINEAR);right=cv2.remap(frame[:,1600:],*mr,cv2.INTER_LINEAR)
                gl=cv2.resize(cv2.cvtColor(left,cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
                gr=cv2.resize(cv2.cvtColor(right,cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
                t=time.perf_counter();disp=matcher.compute(gl[y0:y1],gr[y0:y1]).astype(np.float32)/16;ms=(time.perf_counter()-t)*1000
                for precision,(pair,extras) in boxes.items():
                    a=acc[precision];n=sum(b is not None for b in pair);a['tp']+=n;a['fn']+=2-n;a['fp']+=extras;a['exact']+=n==2 and extras==0
                    zs=[];pts=[]
                    for ti,b in enumerate(pair):
                        if not b:zs.append(None);pts.append(None);continue
                        x1,yy1,x2,yy2=rect(b);roi=disp[max(0,yy1-y0):max(0,yy2-y0),x1:x2]
                        valid=roi[np.isfinite(roi)&(roi>3)&(roi<120)]
                        z=BASE*focal/(2*float(np.median(valid))) if valid.size>=8 else None
                        zs.append(z);pts.append(point(b,z) if z is not None else None)
                        cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
                        a['center'][ti].append(math.hypot(cx-TARGETS[ti][0],cy-TARGETS[ti][1]))
                        if z is not None:a['ranges'][ti].append(z)
                    if all(p is not None for p in pts):a['spacing'].append(math.dist(*pts))
                    a['times'].append(ms)
        decoded+=1
    cap.release();rows=[]
    for precision,a in acc.items():
        latency=dets[precision]['single_view_inference_ms_median']
        rows.append({'precision':'FP16' if precision=='evaluation' else 'FP32','frames':decoded,
            'exact_two_frames':a['exact'],'tp':a['tp'],'fp':a['fp'],'fn':a['fn'],
            'center_error_px_by_target':[summary(v) for v in a['center']],
            'range_m_by_target':[summary(v) for v in a['ranges']],'pair_spacing_m':summary(a['spacing']),
            'stereo_match_ms_median':round(statistics.median(a['times']),2) if a['times'] else None,
            'detector_ms_median':latency,'approx_detector_plus_match_ms':round(latency+statistics.median(a['times']),2) if a['times'] else None})
    out['results'].append({'scene':scene,'decoded_frames':decoded,'elapsed_s':round(time.perf_counter()-start,2),'precisions':rows})
    print(scene,rows)
path=ROOT/'media/frc-fuel-fp16-vs-fp32-range-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
