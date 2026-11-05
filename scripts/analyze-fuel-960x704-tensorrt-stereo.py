#!/usr/bin/env python3
"""Recompute block-3 stereo for stored real-frame 960x704 TensorRT outputs."""
import json, math, statistics
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/"media/frc-fuel-960x704-tensorrt-evaluation.json"
data=json.loads(IN.read_text())
COUNT=json.loads((ROOT/"media/frc-fuel-dense-count-nms015.json").read_text())
TARGETS=((808.0,589.0),(902.0,591.0));BASELINE=.0635
fs=cv2.FileStorage(str(ROOT/"media/calibration/stereo_calibration_live_order_candidate.yml"),cv2.FILE_STORAGE_READ)
kl=fs.getNode("left_camera_matrix").mat();dl=fs.getNode("left_distortion").mat()
kr=fs.getNode("right_camera_matrix").mat();dr=fs.getNode("right_distortion").mat()
rotation=fs.getNode("rotation").mat();translation=fs.getNode("translation").mat();fs.release()
r1,r2,p1,p2,*_=cv2.stereoRectify(kl,dl,kr,dr,(1600,1200),rotation,translation,
                                  flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
map_l=cv2.initUndistortRectifyMap(kl,dl,r1,p1,(1600,1200),cv2.CV_32FC1)
map_r=cv2.initUndistortRectifyMap(kr,dr,r2,p2,(1600,1200),cv2.CV_32FC1)
matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=3,
    P1=72,P2=288,uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

def match(boxes,threshold):
 selected=[b for b in boxes if b["confidence"]>=threshold and
   max(b["x2"]-b["x1"],b["y2"]-b["y1"])/max(1,min(b["x2"]-b["x1"],b["y2"]-b["y1"]))<=2]
 selected.sort(key=lambda b:-b["confidence"]);hits=[None,None]
 for b in selected:
  c=((b["x1"]+b["x2"])/2,(b["y1"]+b["y2"])/2)
  d,i=min((math.dist(c,t),j) for j,t in enumerate(TARGETS))
  if d<=60 and hits[i] is None:hits[i]=b
 return hits

def xyz(box,disp):
 corners=np.array([[[box['x1'],box['y1']]],[[box['x2'],box['y1']]],
                   [[box['x2'],box['y2']]],[[box['x1'],box['y2']]]],np.float32)
 rect=cv2.undistortPoints(corners,kl,dl,R=r1,P=p1).reshape(-1,2)*.5
 x1,y1=np.floor(rect.min(0)).astype(int);x2,y2=np.ceil(rect.max(0)).astype(int)
 roi=disp[max(0,y1):min(600,y2),max(0,x1):min(800,x2)]
 vals=roi[np.isfinite(roi)&(roi>1.5)&(roi<60)]
 if len(vals)<8:return None
 z=BASELINE*float(p1[0,0])/(2*float(np.median(vals)))
 c=np.array([[[ (box['x1']+box['x2'])/2,(box['y1']+box['y2'])/2 ]]],np.float32)
 u,v=cv2.undistortPoints(c,kl,dl,R=r1,P=p1)[0,0]
 return z,np.asarray(((u-p1[0,2])*z/p1[0,0],(v-p1[1,2])*z/p1[1,1],z))

def summary(vals):
 if not vals:return None
 m=float(statistics.median(vals))
 return {'n':len(vals),'median_m':round(m,5),'mad_m':round(statistics.median(abs(x-m) for x in vals),5),
         'p05_m':round(float(np.percentile(vals,5)),5),'p95_m':round(float(np.percentile(vals,95)),5)}

out={'method':'Use the saved per-frame outputs from the actual 960x704 FP16 TensorRT GPU engine, recompute block-3 SGBM for each corresponding recorded stereo frame, and compare candidate confidence 0.02/0.03/0.05 with cached 1280x960 detections on those same frames.',
 'input':str(IN.relative_to(ROOT)),'results':[],
 'limitations':'Only 30 sampled frames from two stationary clips. Range consistency is not absolute accuracy; no tape-measured distances or moving targets.'}
for rec in data['results']:
 scene=rec['scene']
 # Video names are fixed and parallel to the saved evaluation scene order.
 video={'earlier':'media/frc-2026-live-two-fuel-60fps-cfr.mp4',
        'corrected-settings':'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'}[scene]
 base=next(x for x in COUNT['results'] if x['scene']==scene)
 cap=cv2.VideoCapture(str(ROOT/video))
 stats={f'{kind}_{th}':{'ranges':[[],[]],'pairs':[]}
        for kind,thresholds in [('candidate',(.02,.03,.05)),('baseline',(.05,))]
        for th in thresholds}
 for fr in rec['per_frame']:
  idx=int(fr['frame_index']);cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read()
  if not ok:continue
  left,right=frame[:,:1600],frame[:,1600:]
  left=cv2.resize(cv2.cvtColor(cv2.remap(left,map_l[0],map_l[1],cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
  right=cv2.resize(cv2.cvtColor(cv2.remap(right,map_r[0],map_r[1],cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
  disp=matcher.compute(left,right).astype(np.float32)/16
  for threshold in (.02,.03,.05):
   hits=match(fr['candidate_boxes_ge_0p01'],threshold);points=[]
   for target,b in enumerate(hits):
    p=xyz(b,disp) if b else None;points.append(p)
    if p:stats[f'candidate_{threshold}']['ranges'][target].append(p[0])
   if all(points):stats[f'candidate_{threshold}']['pairs'].append(float(np.linalg.norm(points[0][1]-points[1][1])))
  boxes=base['per_frame'][idx]['detections'];hits=match(boxes,.05);points=[]
  for target,b in enumerate(hits):
   p=xyz(b,disp) if b else None;points.append(p)
   if p:stats['baseline_0.05']['ranges'][target].append(p[0])
  if all(points):stats['baseline_0.05']['pairs'].append(float(np.linalg.norm(points[0][1]-points[1][1])))
 cap.release()
 out['results'].append({'scene':scene,'stereo':{key:{'ranges':[summary(v) for v in x['ranges']],
   'pair_spacing_m':summary(x['pairs'])} for key,x in stats.items()}})
 print(scene,json.dumps(out['results'][-1]))
path=ROOT/'media/frc-fuel-960x704-tensorrt-stereo-rescore.json'
path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
