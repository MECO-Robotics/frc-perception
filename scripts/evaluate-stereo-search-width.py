#!/usr/bin/env python3
"""Sweep SGBM disparity search widths and modes on the saved FUEL stereo clips."""
import json,math,os,statistics,time
from pathlib import Path
import cv2,numpy as np
from scout.detectors import TensorRTDetector

ROOT=Path(__file__).resolve().parents[1]
ENGINE=ROOT/'models/rebuilt_fuel_yolo26n_1280_fp16.engine'
CAL=ROOT/'media/calibration/stereo_calibration_live_order_candidate.yml'
VIDEOS=('media/frc-2026-live-two-fuel-60fps-cfr.mp4','media/frc-2026-fuel-camera-settings-fixed-60fps.mp4')
TIMES=(1,2,3,4,5,6.6)
TARGETS=((808,589),(902,591));BASELINE=.0635
MODES=(('SGBM',cv2.STEREO_SGBM_MODE_SGBM),('HH',cv2.STEREO_SGBM_MODE_HH),
       ('3WAY',cv2.STEREO_SGBM_MODE_SGBM_3WAY))
SEARCH_WIDTHS=(64,96,128)

def med(a):return statistics.median(a) if a else None
def mad(a):
 m=med(a);return med([abs(v-m) for v in a]) if a else None

fs=cv2.FileStorage(str(CAL),cv2.FILE_STORAGE_READ)
kl=fs.getNode('left_camera_matrix').mat();dl=fs.getNode('left_distortion').mat()
kr=fs.getNode('right_camera_matrix').mat();dr=fs.getNode('right_distortion').mat()
rotation=fs.getNode('rotation').mat();translation=fs.getNode('translation').mat();fs.release()
r1,r2,p1,p2,*_=cv2.stereoRectify(kl,dl,kr,dr,(1600,1200),rotation,translation,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
ml=cv2.initUndistortRectifyMap(kl,dl,r1,p1,(1600,1200),cv2.CV_32FC1)
mr=cv2.initUndistortRectifyMap(kr,dr,r2,p2,(1600,1200),cv2.CV_32FC1)
os.environ['SCOUT_CLASS_NAMES']='["ball"]';os.environ['SCOUT_MAX_BOX_ASPECT_RATIO']='2.0'
detector=TensorRTDetector(str(ENGINE),confidence=.10)
configs=[(name,n,mode) for name,mode in MODES for n in SEARCH_WIDTHS]
results=[]
for video_name in VIDEOS:
 cap=cv2.VideoCapture(str(ROOT/video_name))
 records={key:{i:[] for i in range(2)} for key in configs}
 for time_s in TIMES:
  cap.set(cv2.CAP_PROP_POS_MSEC,time_s*1000);ok,frame=cap.read()
  if not ok:continue
  left,right=frame[:,:1600],frame[:,1600:]
  detections,det_ms=detector.infer(left)
  boxes=[None,None]
  for b in detections:
   cx,cy=(b.x1+b.x2)/2,(b.y1+b.y2)/2
   d=[math.hypot(cx-x,cy-y) for x,y in TARGETS];err,i=min((v,i) for i,v in enumerate(d))
   if err<=60 and (boxes[i] is None or b.confidence>boxes[i].confidence):boxes[i]=b
  gl=cv2.cvtColor(left,cv2.COLOR_BGR2GRAY);gr=cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
  gl=cv2.remap(gl,ml[0],ml[1],cv2.INTER_LINEAR);gr=cv2.remap(gr,mr[0],mr[1],cv2.INTER_LINEAR)
  gl=cv2.resize(gl,(800,600),interpolation=cv2.INTER_AREA);gr=cv2.resize(gr,(800,600),interpolation=cv2.INTER_AREA)
  for config in configs:
   name,n,mode=config
   matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=n,blockSize=5,
       P1=8*5*5,P2=32*5*5,uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,mode=mode)
   t0=time.perf_counter();disp=matcher.compute(gl,gr).astype(np.float32)/16.0;elapsed=(time.perf_counter()-t0)*1000
   points=[]
   for i,b in enumerate(boxes):
    if b is None:points.append(None);continue
    corners=np.array([[[b.x1,b.y1]],[[b.x2,b.y1]],[[b.x2,b.y2]],[[b.x1,b.y2]]],np.float32)
    rect=cv2.undistortPoints(corners,kl,dl,R=r1,P=p1).reshape(-1,2)/2
    x1,y1=np.floor(rect.min(axis=0)).astype(int);x2,y2=np.ceil(rect.max(axis=0)).astype(int)
    roi=disp[max(0,y1):min(600,y2),max(0,x1):min(800,x2)]
    valid=roi[np.isfinite(roi)&(roi>3)&(roi<120)]
    z=None
    if valid.size>=8:z=BASELINE*float(p1[0,0])/(float(np.median(valid))*2)
    records[config][i].append({'time_s':time_s,'z_m':z,'valid_pixels':int(valid.size),
                               'detector_ms':det_ms,'match_ms':elapsed,'center':[(b.x1+b.x2)/2,(b.y1+b.y2)/2]})
    if z is None:points.append(None);continue
    c=np.array([[[ (b.x1+b.x2)/2,(b.y1+b.y2)/2 ]]],np.float32)
    rc=cv2.undistortPoints(c,kl,dl,R=r1,P=p1)[0,0]
    points.append(((rc[0]-p1[0,2])*z/p1[0,0],(rc[1]-p1[1,2])*z/p1[1,1],z))
   if all(p is not None for p in points):
    dist=math.sqrt(sum((points[0][j]-points[1][j])**2 for j in range(3)))
    for i in range(2):records[config][i][-1]['pair_spacing_m']=dist
 cap.release()
 for config in configs:
  name,n,mode=config;target_results=[]
  for i in range(2):
   entries=records[config][i];zs=[x['z_m'] for x in entries if x['z_m'] is not None]
   pairs=[x['pair_spacing_m'] for x in entries if x.get('pair_spacing_m') is not None]
   target_results.append({'target':i+1,'valid_ranges':len(zs),'median_z_m':round(med(zs),3) if zs else None,
      'mad_z_m':round(mad(zs),3) if zs else None,'median_valid_pixels':round(med([x['valid_pixels'] for x in entries]),1) if entries else None,
      'pair_spacing_m':{'median':round(med(pairs),3),'mad':round(mad(pairs),3)} if pairs else None})
  match_times=[x['match_ms'] for i in range(2) for x in records[config][i]]
  results.append({'video':video_name,'matcher':name,'num_disparities':n,'min_disparity':0,
      'block_size':5,'uniqueness_ratio':8,'targets':target_results,
      'matcher_compute_ms_median':round(med(match_times),2) if match_times else None})
detector.close()
out={'method':'Active FRC 9550 TensorRT 1280 boxes on six sampled stereo pairs per video. Same candidate live-order rectification, SGBM block size 5, uniqueness 8, speckle settings, 63.5 mm runtime baseline, and full-box ROI; compare three SGBM modes and disparity search widths 64/96/128.','limitations':'Two clear stationary ball clips only; no tape-measured distance or disparity labels. Range MAD and pair spacing measure repeatability, not absolute accuracy.','configuration':{'calibration':'media/calibration/stereo_calibration_live_order_candidate.yml','confidence':.10,'max_box_aspect_ratio':2.0,'rectified_focal_px':float(p1[0,0]),'sample_times_s':list(TIMES)},'results':results}
(ROOT/'media/frc-stereo-search-width-sweep.json').write_text(json.dumps(out,indent=2)+'\n')
print('saved',ROOT/'media/frc-stereo-search-width-sweep.json','rows',len(results))
