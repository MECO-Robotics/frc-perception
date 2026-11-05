#!/usr/bin/env python3
"""Compare available calibration files on identical saved stereo frames/boxes."""
import json,math,statistics
from pathlib import Path
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'media/frc-model-checkpoint-sweep-640-tensorrt.json').read_text())
CALIBRATIONS=('media/calibration/stereo_calibration_live_order_candidate.yml',
              'media/calibration/stereo_calibration.yml',
              'media/calibration/stereo_calibration_unverified.yml')
TARGETS=((808,589),(902,591));BASELINE_M=.0635

def med(x):return statistics.median(x) if x else None
def mad(x):
 m=med(x);return med([abs(v-m) for v in x]) if x else None

results=[]
for cal_name in CALIBRATIONS:
 fs=cv2.FileStorage(str(ROOT/cal_name),cv2.FILE_STORAGE_READ)
 kl=fs.getNode('left_camera_matrix').mat();dl=fs.getNode('left_distortion').mat()
 kr=fs.getNode('right_camera_matrix').mat();dr=fs.getNode('right_distortion').mat()
 rotation=fs.getNode('rotation').mat();translation=fs.getNode('translation').mat();fs.release()
 r1,r2,p1,p2,*_=cv2.stereoRectify(kl,dl,kr,dr,(1600,1200),rotation,translation,
                                      flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
 ml=cv2.initUndistortRectifyMap(kl,dl,r1,p1,(1600,1200),cv2.CV_32FC1)
 mr=cv2.initUndistortRectifyMap(kr,dr,r2,p2,(1600,1200),cv2.CV_32FC1)
 for video_result in DATA['results']:
  cap=cv2.VideoCapture(str(ROOT/video_result['video']))
  zs=[[],[]];valid_counts=[[],[]];spacing=[];valid_pairs=0
  for time_s,boxes in zip(DATA['sample_times_s'],video_result['frames']):
   cap.set(cv2.CAP_PROP_POS_MSEC,time_s*1000);ok,frame=cap.read()
   if not ok:continue
   left,right=frame[:,:1600],frame[:,1600:]
   gl=cv2.cvtColor(left,cv2.COLOR_BGR2GRAY);gr=cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
   gl=cv2.remap(gl,ml[0],ml[1],cv2.INTER_LINEAR);gr=cv2.remap(gr,mr[0],mr[1],cv2.INTER_LINEAR)
   gl=cv2.resize(gl,(800,600),interpolation=cv2.INTER_AREA);gr=cv2.resize(gr,(800,600),interpolation=cv2.INTER_AREA)
   matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=5,
       P1=8*5*5,P2=32*5*5,uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,
       mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
   disp=matcher.compute(gl,gr).astype(np.float32)/16
   selected=[None,None]
   for b in boxes:
    if b.get('confidence',0)<.10:continue
    x1,y1,x2,y2=b['xyxy'];cx=(x1+x2)/2;cy=(y1+y2)/2
    distance=[math.hypot(cx-tx,cy-ty) for tx,ty in TARGETS];dist,i=min((v,i) for i,v in enumerate(distance))
    if dist<=60 and selected[i] is None:selected[i]=b
   points=[]
   for i,b in enumerate(selected):
    if b is None:points.append(None);continue
    x1,y1,x2,y2=b['xyxy']
    pts=np.array([[[x1,y1]],[[x2,y1]],[[x2,y2]],[[x1,y2]]],np.float32)
    rect=cv2.undistortPoints(pts,kl,dl,R=r1,P=p1).reshape(-1,2)/2
    rx1,ry1=np.floor(rect.min(axis=0)).astype(int);rx2,ry2=np.ceil(rect.max(axis=0)).astype(int)
    roi=disp[max(0,ry1):min(600,ry2),max(0,rx1):min(800,rx2)]
    valid=roi[np.isfinite(roi)&(roi>3)&(roi<120)]
    valid_counts[i].append(int(valid.size))
    if valid.size<8:points.append(None);continue
    z=BASELINE_M*float(p1[0,0])/(float(np.median(valid))*2)
    zs[i].append(z)
    c=np.array([[[ (x1+x2)/2,(y1+y2)/2 ]]],np.float32)
    rectc=cv2.undistortPoints(c,kl,dl,R=r1,P=p1)[0,0]
    x=(rectc[0]-p1[0,2])*z/p1[0,0];y=(rectc[1]-p1[1,2])*z/p1[1,1]
    points.append((float(x),float(y),float(z)))
   if all(p is not None for p in points):
    spacing.append(math.sqrt(sum((points[0][j]-points[1][j])**2 for j in range(3))));valid_pairs+=1
  cap.release()
  results.append({'calibration':cal_name,'video':video_result['video'],
      'rectified_focal_px':float(p1[0,0]),'translation_baseline_m':float(np.linalg.norm(translation)),
      'runtime_baseline_override_m':BASELINE_M,
      'target_ranges':[{'target':i+1,'valid_frames':len(zs[i]),'median_m':round(med(zs[i]),3) if zs[i] else None,
          'mad_m':round(mad(zs[i]),3) if zs[i] else None,'valid_disparity_pixels_median':round(med(valid_counts[i]),1) if valid_counts[i] else None} for i in range(2)],
      'paired_valid_frames':valid_pairs,
      'pair_xyz_spacing_m':{'median':round(med(spacing),3),'mad':round(mad(spacing),3)} if spacing else None})
out={'method':'Same 15 sampled synchronized pairs per saved clip, same stored detector boxes and SGBM b5/u8, with each available calibration applied to the left|right video order. Mechanical baseline override 63.5 mm is held fixed so this isolates intrinsics/rectification. Results are consistency diagnostics, not ground-truth accuracy.',
     'calibrations':CALIBRATIONS,'results':results}
(ROOT/'media/frc-stereo-calibration-sweep.json').write_text(json.dumps(out,indent=2)+'\n')
for r in results:print(r)
