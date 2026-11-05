#!/usr/bin/env python3
"""Compare downsampling kernels for the FRC detector's 1280x960 input."""
import json,math,os,statistics,sys,time
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scout.detectors import TensorRTDetector
VIDEOS=(('earlier','media/frc-2026-live-two-fuel-60fps-cfr.mp4'),
        ('corrected-settings','media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'))
TARGETS=((808,589),(902,591));STRIDE=5;INPUT=(1280,960);COORD_SCALE=1.25
ENGINE=ROOT/'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
METHODS={'linear':None,'area':cv2.INTER_AREA,'cubic':cv2.INTER_CUBIC,'lanczos4':cv2.INTER_LANCZOS4}

def evaluate(boxes,scale):
    hits=[None,None];extras=0
    for b in boxes:
        x=(b.x1+b.x2)*scale/2;y=(b.y1+b.y2)*scale/2
        dist,idx=min((math.hypot(x-tx,y-ty),i) for i,(tx,ty) in enumerate(TARGETS))
        if dist<=60 and hits[idx] is None:hits[idx]=(b,dist)
        else:extras+=1
    return hits,extras

os.environ['SCOUT_CLASS_NAMES']='["ball"]';os.environ['SCOUT_MAX_BOX_ASPECT_RATIO']='2.0'
detector=TensorRTDetector(str(ENGINE),confidence=.05)
out={'method':'Every fifth frame: compare detector-internal linear resize with externally pre-resized 1280x960 inputs using INTER_AREA/CUBIC/LANCZOS4. Scale resized detections back to original pixels; confidence .05.',
     'limitations':'Two stationary videos only; fixed manual target centers. No motion or alternate camera-resolution tests.',
     'sample_stride_frames':STRIDE,'results':[]}
try:
    for scene,video in VIDEOS:
        cap=cv2.VideoCapture(str(ROOT/video));acc={k:{'exact':0,'tp':0,'fp':0,'fn':0,'errors':[[],[]],'resize_ms':[],'infer_ms':[]} for k in METHODS}
        frames=0;idx=0;start=time.perf_counter()
        while True:
            ok,stereo=cap.read()
            if not ok:break
            if idx%STRIDE==0:
                left=stereo[:,:1600]
                for name,interp in METHODS.items():
                    if interp is None:image=left;resize_ms=0;scale=1.
                    else:
                        t=time.perf_counter();image=cv2.resize(left,INPUT,interpolation=interp)
                        resize_ms=(time.perf_counter()-t)*1000;scale=COORD_SCALE
                    boxes,ms=detector.infer(image);hits,extras=evaluate(boxes,scale);a=acc[name]
                    n=sum(h is not None for h in hits);a['tp']+=n;a['fn']+=2-n;a['fp']+=extras
                    a['exact']+=n==2 and extras==0;a['resize_ms'].append(resize_ms);a['infer_ms'].append(ms)
                    for ti,h in enumerate(hits):
                        if h:a['errors'][ti].append(h[1])
                frames+=1
            idx+=1
        cap.release();rows=[]
        for name,a in acc.items():
            p=a['tp']/(a['tp']+a['fp']) if a['tp']+a['fp'] else 0;r=a['tp']/(a['tp']+a['fn']) if a['tp']+a['fn'] else 0
            rows.append({'method':name,'sampled_frames':frames,'exact_two_frames':a['exact'],
                'tp':a['tp'],'fp':a['fp'],'fn':a['fn'],'precision':round(p,5),'recall':round(r,5),
                'center_error_px_by_target':[{'n':len(v),'median':round(statistics.median(v),2),
                    'p95':round(sorted(v)[int(.95*(len(v)-1))],2)} if v else None for v in a['errors']],
                'resize_ms_median':round(statistics.median(a['resize_ms']),3),
                'detector_ms_median':round(statistics.median(a['infer_ms']),2),
                'combined_ms_median':round(statistics.median([x+y for x,y in zip(a['resize_ms'],a['infer_ms'])]),2)})
        out['results'].append({'scene':scene,'video':video,'decoded_frames':idx,'sampled_frames':frames,
            'elapsed_s':round(time.perf_counter()-start,2),'configs':rows})
        print(scene,rows)
finally:detector.close()
path=ROOT/'media/frc-fuel-resize-interpolation-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
