#!/usr/bin/env python3
"""Compare mild image enhancement options on sampled saved FUEL frames."""
import json, math, os, statistics, sys, time
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scout.detectors import TensorRTDetector

VIDEOS=(('earlier','media/frc-2026-live-two-fuel-60fps-cfr.mp4'),
        ('corrected-settings','media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'))
TARGETS=((808,589),(902,591)); RADIUS=60; STRIDE=5
ENGINE=ROOT/'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
CONFIGS=('raw','gamma_0.8','gamma_0.65','clahe_y')
TABLES={name:((__import__('numpy').clip((__import__('numpy').arange(256)/255.)**exp*255,0,255)).astype('uint8'))
        for name,exp in (('gamma_0.8',.8),('gamma_0.65',.65))}


def enhance(frame,name):
    if name=='raw':return frame
    if name in TABLES:return cv2.LUT(frame,TABLES[name])
    ycrcb=cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    ycrcb[:,:,0]=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(ycrcb[:,:,0])
    return cv2.cvtColor(ycrcb,cv2.COLOR_YCrCb2BGR)


def evaluate(boxes):
    hits=[None,None];extras=0
    for b in sorted(boxes,key=lambda b:-b.confidence):
        x,y=(b.x1+b.x2)/2,(b.y1+b.y2)/2
        dist,idx=min((math.hypot(x-tx,y-ty),i) for i,(tx,ty) in enumerate(TARGETS))
        if dist<=RADIUS and hits[idx] is None:hits[idx]=(b,dist)
        else:extras+=1
    return hits,extras


os.environ['SCOUT_CLASS_NAMES']='["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO']='2.0'
detector=TensorRTDetector(str(ENGINE),confidence=.05)
out={'method':'Run FRC YOLO26n 1280x960 FP16 on every fifth saved frame. Compare raw BGR, mild gamma brightening, and CLAHE on Y; all detections scored at confidence 0.05.',
     'limitations':'Two stationary scenes only; fixed target centers; no real motion or occlusion. Enhancement may behave differently under other field lights/camera exposure.',
     'sample_stride_frames':STRIDE,'configs':list(CONFIGS),'results':[]}
try:
    for scene,video in VIDEOS:
        cap=cv2.VideoCapture(str(ROOT/video));stats={n:{'exact':0,'tp':0,'fp':0,'fn':0,'errors':[[],[]],'latency':[],'prep':[]} for n in CONFIGS}
        frames=0;start=time.perf_counter();idx=0
        while True:
            ok,stereo=cap.read()
            if not ok:break
            if idx%STRIDE==0:
                left=stereo[:,:1600]
                for name in CONFIGS:
                    prep_start=time.perf_counter();image=enhance(left,name)
                    prep_ms=(time.perf_counter()-prep_start)*1000
                    boxes,latency=detector.infer(image);hits,extras=evaluate(boxes);a=stats[name]
                    count=sum(h is not None for h in hits);a['tp']+=count;a['fn']+=2-count;a['fp']+=extras
                    a['exact']+=count==2 and extras==0;a['latency'].append(latency);a['prep'].append(prep_ms)
                    for target,h in enumerate(hits):
                        if h:a['errors'][target].append(h[1])
                frames+=1
            idx+=1
        cap.release();rows=[]
        for name,a in stats.items():
            tp,fp,fn=a['tp'],a['fp'],a['fn'];precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
            rows.append({'config':name,'sampled_frames':frames,'exact_two_frames':a['exact'],
                'tp':tp,'fp':fp,'fn':fn,'precision':round(precision,5),'recall':round(recall,5),
                'center_error_px_by_target':[{'n':len(v),'median':round(statistics.median(v),2),
                    'p95':round(sorted(v)[int(.95*(len(v)-1))],2)} if v else None for v in a['errors']],
                'preprocess_ms_median':round(statistics.median(a['prep']),3) if a['prep'] else None,
                'inference_ms_median':round(statistics.median(a['latency']),2) if a['latency'] else None})
        out['results'].append({'scene':scene,'video':video,'decoded_frames':idx,'sampled_frames':frames,
            'elapsed_s':round(time.perf_counter()-start,2),'configs':rows})
        print(scene,rows)
finally:
    detector.close()
path=ROOT/'media/frc-fuel-detector-preprocessing-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
