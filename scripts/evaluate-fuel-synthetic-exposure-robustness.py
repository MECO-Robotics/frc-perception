#!/usr/bin/env python3
"""Stress-test detector and stereo output under controlled brightness reductions."""
import json,math,os,statistics,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scout.detectors import TensorRTDetector,Detection
from scout.stereo import StereoDepth
VIDEOS=(('earlier','media/frc-2026-live-two-fuel-60fps-cfr.mp4'),('corrected-settings','media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'))
TIMES=tuple(1.0+.4*i for i in range(15));TARGETS=((808.,589.),(902.,591.));SCALES=(1.0,.5,.25,.125);MODES=('raw','gamma06');CONF=.05;GAMMA=.6;LUT=np.clip((np.arange(256,dtype=np.float32)/255.)**GAMMA*255,0,255).astype(np.uint8)
os.environ.update(SCOUT_CLASS_NAMES='["ball"]',SCOUT_MAX_BOX_ASPECT_RATIO='2.0',SCOUT_NMS_IOU='.15',SCOUT_STEREO_CALIBRATION='media/calibration/stereo_calibration_live_order_candidate.yml',SCOUT_STEREO_RECTIFY='1',SCOUT_STEREO_BASELINE_M='.0635',SCOUT_STEREO_DISPARITY_SCALE='1',SCOUT_STEREO_MATCH_SCALE='.5',SCOUT_STEREO_BLOCK_SIZE='5',SCOUT_STEREO_NUM_DISPARITIES='128',SCOUT_STEREO_VERTICAL_ROI_PAD='24',SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR='.55',SCOUT_FUEL_DIAMETER_M='.15')
def pack(bs):return [{'x1':b.x1,'y1':b.y1,'x2':b.x2,'y2':b.y2,'confidence':float(b.confidence)} for b in bs]
def match(dets):
 hits=[None,None];extras=0
 for b in sorted((b for b in dets if b['confidence']>=CONF),key=lambda b:-b['confidence']):
  x=(b['x1']+b['x2'])/2;y=(b['y1']+b['y2'])/2;err,i=min((math.hypot(x-a,y-c),i) for i,(a,c) in enumerate(TARGETS))
  if err<=60 and hits[i] is None:hits[i]=(b,err)
  else:extras+=1
 return hits,extras
def medmad(values):
 if not values:return {'n':0,'median_m':None,'mad_m':None}
 m=statistics.median(values);return {'n':len(values),'median_m':round(m,5),'mad_m':round(statistics.median(abs(x-m) for x in values),5),'p05_m':round(float(np.percentile(values,5)),5),'p95_m':round(float(np.percentile(values,95)),5)}
det=TensorRTDetector(str(ROOT/'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'),confidence=.02,iou=.15);st=StereoDepth();out={'method':'Sample 15 synchronized stereo pairs from each saved clip. Scale both left/right BGR frames by factors 1.0/0.5/0.25/0.125, run the detector on scaled raw and gamma-0.60 left images, and compute SGBM on the scaled raw stereo pair using the active .55 size gate. Score target hits against fixed centers, extras, center error, stereo range repeatability, pair spacing, and 26 cm merge count.','limitations':'Brightness multiplication is a controlled exposure stress, not a sensor model: it adds no shot/read noise, blur, auto-exposure transition, or changing field background. Two static clips and fixed target centers only; stereo range has no external metric ground truth.','scales':SCALES,'detector_modes':MODES,'confidence':CONF,'nms_iou':.15,'gamma':GAMMA,'sample_times_s':TIMES,'results':[]}
try:
 for scene,video in VIDEOS:
  cap=cv2.VideoCapture(str(ROOT/video));per={f'{scale:g}/{mode}':{'frames':0,'tp':0,'fp':0,'fn':0,'paired':0,'exact':0,'center_errors':[[],[]],'ranges':[[],[]],'spacing':[],'latency':[]} for scale in SCALES for mode in MODES}
  for t in TIMES:
   cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read()
   if not ok:continue
   left0,right0=frame[:,:1600],frame[:,1600:]
   for scale in SCALES:
    left=np.clip(left0.astype(np.float32)*scale,0,255).astype(np.uint8);right=np.clip(right0.astype(np.float32)*scale,0,255).astype(np.uint8)
    for mode in MODES:
     inp=cv2.LUT(left,LUT) if mode=='gamma06' else left;boxes,ms=det.infer(inp);hits,extras=match(pack(boxes));key=f'{scale:g}/{mode}';a=per[key];a['frames']+=1;a['latency'].append(ms);n=sum(x is not None for x in hits);a['tp']+=n;a['fn']+=2-n;a['fp']+=extras;a['paired']+=n==2;a['exact']+=n==2 and extras==0
     for i,item in enumerate(hits):
      if item:a['center_errors'][i].append(item[1])
     selected=[Detection(int(x[0]['x1']),int(x[0]['y1']),int(x[0]['x2']),int(x[0]['y2']),0,float(x[0]['confidence'])) if x else None for x in hits]
     valid=[b for b in selected if b]
     if valid:
      st.estimate(left,right,valid)
      # estimator returns the same selected Detection instances with distance fields.
      for i,b in enumerate(selected):
       if b and b.distance_m is not None:a['ranges'][i].append(float(b.distance_m))
      if all(b and b.distance_m is not None for b in selected):
       depths=[float(b.distance_m) for b in selected]
       # 3D point from rectified camera-center coordinates for pair spacing.
       _,_,_,kl,r1,p1=st.maps
       calib=st.calibration
       pts=[]
       for b,z in zip(selected,depths):
        c=cv2.undistortPoints(np.array([[[float(b.x1+b.x2)/2,float(b.y1+b.y2)/2]]],np.float32),calib['left_camera_matrix'],calib['left_distortion'],R=r1,P=p1)[0,0]
        pts.append(((c[0]-p1[0,2])*z/p1[0,0],(c[1]-p1[1,2])*z/p1[1,1],z))
       a['spacing'].append(float(math.dist(*pts)))
  cap.release();rows=[]
  for key,a in per.items():
   rows.append({'scale_mode':key,'frames':a['frames'],'paired_frames':a['paired'],'exact_two_frames':a['exact'],'tp':a['tp'],'fp':a['fp'],'fn':a['fn'],'median_detector_ms':round(statistics.median(a['latency']),2),'target_median_center_error_px':[round(statistics.median(x),2) if x else None for x in a['center_errors']],'stereo_range_by_target_m':[medmad(x) for x in a['ranges']],'pair_spacing_m':medmad(a['spacing']),'merged_pairs_at_26cm':sum(x<=.26 for x in a['spacing'])})
  out['results'].append({'scene':scene,'video':video,'configs':rows});print(scene,[(r['scale_mode'],r['paired_frames'],r['exact_two_frames'],r['tp'],r['fp'],r['fn'],r['stereo_range_by_target_m'],r['merged_pairs_at_26cm']) for r in rows])
finally:det.close()
p=ROOT/'media/frc-fuel-synthetic-exposure-robustness.json';p.write_text(json.dumps(out,indent=2)+'\n');print('saved',p)
