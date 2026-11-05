#!/usr/bin/env python3
"""Prototype 3D sphere fitting from stereo disparity points inside FUEL boxes."""
import json,math,statistics
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
DATA=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
TIMES=tuple(1.0+.4*i for i in range(15));TARGETS=((808,589),(902,591))
BASE=.0635;DIAM=.15;CAL=ROOT/'media/calibration/stereo_calibration_live_order_candidate.yml'
fs=cv2.FileStorage(str(CAL),cv2.FILE_STORAGE_READ)
kl=fs.getNode('left_camera_matrix').mat();dl=fs.getNode('left_distortion').mat()
kr=fs.getNode('right_camera_matrix').mat();dr=fs.getNode('right_distortion').mat()
rotation=fs.getNode('rotation').mat();translation=fs.getNode('translation').mat();fs.release()
size=(1600,1200);r1,r2,p1,p2,*_=cv2.stereoRectify(kl,dl,kr,dr,size,rotation,translation,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
ml=cv2.initUndistortRectifyMap(kl,dl,r1,p1,size,cv2.CV_32FC1);mr=cv2.initUndistortRectifyMap(kr,dr,r2,p2,size,cv2.CV_32FC1)
matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=5,P1=200,P2=800,uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
FOCAL=float(p1[0,0]);
def med(x):return statistics.median(x) if x else None
def match(row):
 out=[None,None]
 for b in sorted((b for b in row['detections'] if b['confidence']>=.1),key=lambda q:q['confidence'],reverse=True):
  cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2;errs=[math.hypot(cx-x,cy-y) for x,y in TARGETS];i=int(np.argmin(errs))
  if errs[i]<=60 and out[i] is None:out[i]=b
 return out
def fit_sphere(points,seed):
 if len(points)<30:return None
 rng=np.random.default_rng(seed)
 if len(points)>600:points=points[rng.choice(len(points),600,replace=False)]
 best=None;best_n=0
 for _ in range(120):
  pick=points[rng.choice(len(points),4,replace=False)];a=2*(pick[1:]-pick[0]);b=np.sum(pick[1:]**2,axis=1)-np.sum(pick[0]**2)
  try:center=np.linalg.solve(a,b)
  except np.linalg.LinAlgError:continue
  radius=float(np.linalg.norm(pick[0]-center))
  if not .02<radius<.3:continue
  residual=np.abs(np.linalg.norm(points-center,axis=1)-radius);inliers=residual<.015;n=int(inliers.sum())
  if n>best_n:best_n=n;best=(center,inliers)
 if best is None or best_n<max(25,int(.2*len(points))):return None
 center,inliers=best
 for _ in range(3):
  pts=points[inliers]
  if len(pts)<4:break
  a=np.column_stack((2*pts,np.ones(len(pts))));b=np.sum(pts*pts,axis=1)
  solution=np.linalg.lstsq(a,b,rcond=None)[0];center=solution[:3]
  radius2=float(center@center+solution[3])
  if radius2<=0:return None
  radius=math.sqrt(radius2);residual=np.abs(np.linalg.norm(points-center,axis=1)-radius)
  median_res=np.median(residual);mad_res=np.median(np.abs(residual-median_res));cutoff=max(.012,median_res+3*1.4826*mad_res)
  inliers=residual<=cutoff
 resid=np.linalg.norm(points[inliers]-center,axis=1)-radius
 return {'center_xyz_m':center,'radius_m':radius,'inlier_fraction':float(inliers.mean()),
         'residual_median_m':float(np.median(np.abs(resid))),'residual_p95_m':float(np.percentile(np.abs(resid),95)),
         'inliers':int(inliers.sum()),'points':int(len(points))}
def xyz_points(disp,roi_bounds,mask=None):
 x1,y1,x2,y2=roi_bounds;roi=disp[y1:y2,x1:x2]
 valid=np.isfinite(roi)&(roi>3)&(roi<120)
 if mask is not None:valid &= mask[y1:y2,x1:x2]
 yy,xx=np.nonzero(valid);d=roi[yy,xx]
 if len(d)<30:return None
 u=(xx+x1+.5)*2;v=(yy+y1+.5)*2;z=BASE*FOCAL/(2*d)
 return np.column_stack(((u-p1[0,2])*z/p1[0,0],(v-p1[1,2])*z/p1[1,1],z))
def summarize(vals):
 return {'n':len(vals),'median_m':round(med(vals),4),'mad_m':round(med([abs(x-med(vals)) for x in vals]),4)} if vals else None
out={'method':'On 15 stable frames per clip, rectify stereo and compute full 800x600 SGBM b5/u8. Convert valid disparity points in each detector box to camera XYZ, then compare unconstrained RANSAC sphere fits and robust least-squares fits fixed to the nominal 7.5cm radius. Also compare raw full-box, central 80%, and yellow-HSV-mask medians. Sphere fits are geometry hypotheses; no distance ground truth is available.','fuel_diameter_m':DIAM,'ransac_inlier_threshold_m':.015,'fixed_fit_radius_m':DIAM/2,'results':[]}
for ci,clip in enumerate(DATA['results']):
 cap=cv2.VideoCapture(str(ROOT/clip['video']));fps=cap.get(cv2.CAP_PROP_FPS);records=[{'box_z':[],'center80_z':[],'hsv_z':[],'sphere_z':[],'sphere_r':[],'inlier':[],'resid':[],'fixed_sphere_z':[],'fixed_resid':[],'fixed_inlier':[],'points':[]} for _ in range(2)]
 for t in TIMES:
  idx=round(t*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read()
  if not ok:continue
  boxes=match(clip['per_frame'][idx]);left,right=frame[:,:1600],frame[:,1600:]
  cl=cv2.remap(left,*ml,cv2.INTER_LINEAR);cr=cv2.remap(right,*mr,cv2.INTER_LINEAR)
  gl=cv2.resize(cv2.cvtColor(cl,cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
  gr=cv2.resize(cv2.cvtColor(cr,cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
  disp=matcher.compute(gl,gr).astype(np.float32)/16;hsv=cv2.cvtColor(cl,cv2.COLOR_BGR2HSV)
  for ti,b in enumerate(boxes):
   if b is None:continue
   corners=np.array([[[b['x1'],b['y1']]],[[b['x2'],b['y1']]],[[b['x2'],b['y2']]],[[b['x1'],b['y2']]]],np.float32)
   q=cv2.undistortPoints(corners,kl,dl,R=r1,P=p1).reshape(-1,2)/2
   x1=max(0,int(np.floor(q[:,0].min())));y1=max(0,int(np.floor(q[:,1].min())))
   x2=min(800,int(np.ceil(q[:,0].max())));y2=min(600,int(np.ceil(q[:,1].max())))
   roi=disp[y1:y2,x1:x2];valid=roi[np.isfinite(roi)&(roi>3)&(roi<120)]
   if len(valid)<30:continue
   records[ti]['box_z'].append(BASE*FOCAL/(2*float(np.median(valid))))
   cy1=y1+int(.1*(y2-y1));cy2=y2-int(.1*(y2-y1));cx1=x1+int(.1*(x2-x1));cx2=x2-int(.1*(x2-x1))
   core=disp[cy1:cy2,cx1:cx2];corev=core[np.isfinite(core)&(core>3)&(core<120)]
   if len(corev)>=20:records[ti]['center80_z'].append(BASE*FOCAL/(2*float(np.median(corev))))
   mask=(hsv[:,:,0]>=10)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>=40)&(hsv[:,:,2]>=40)
   mask=cv2.resize(mask.astype(np.uint8),(800,600),interpolation=cv2.INTER_NEAREST).astype(bool)
   vals=roi[mask[y1:y2,x1:x2]&np.isfinite(roi)&(roi>3)&(roi<120)]
   if len(vals)>=20:records[ti]['hsv_z'].append(BASE*FOCAL/(2*float(np.median(vals))))
   pts=xyz_points(disp,(x1,y1,x2,y2));fit=fit_sphere(pts,seed=ci*1000+idx+ti) if pts is not None else None
   if fit:
    records[ti]['sphere_z'].append(float(fit['center_xyz_m'][2]));records[ti]['sphere_r'].append(fit['radius_m'])
    records[ti]['inlier'].append(fit['inlier_fraction']);records[ti]['resid'].append(fit['residual_p95_m']);records[ti]['points'].append(fit['points'])
   pts=xyz_points(disp,(x1,y1,x2,y2))
   if pts is not None and len(pts)>=30:
    sample=pts if len(pts)<=1200 else pts[np.linspace(0,len(pts)-1,1200).astype(int)]
    initial=fit['center_xyz_m'] if fit else np.median(sample,axis=0)+np.array((0,0,DIAM/2))
    model=least_squares(lambda c:np.linalg.norm(sample-c,axis=1)-DIAM/2,initial,loss='soft_l1',f_scale=.01,max_nfev=60)
    residual=np.abs(np.linalg.norm(sample-model.x,axis=1)-DIAM/2)
    records[ti]['fixed_sphere_z'].append(float(model.x[2]));records[ti]['fixed_resid'].append(float(np.percentile(residual,95)))
    records[ti]['fixed_inlier'].append(float(np.mean(residual<.015)))
 cap.release();entry={'scene':clip['scene'],'targets':[]}
 for ti,r in enumerate(records):
  target={'target':ti+1,'box_median_range_m':summarize(r['box_z']),'center80_range_m':summarize(r['center80_z']),
          'yellow_mask_range_m':summarize(r['hsv_z']),'sphere_center_range_m':summarize(r['sphere_z']),
          'sphere_radius_m':summarize(r['sphere_r']),'sphere_inlier_fraction_median':round(med(r['inlier']),3) if r['inlier'] else None,
          'sphere_residual_p95_median':round(med(r['resid']),4) if r['resid'] else None,
          'fixed_radius_sphere_center_range_m':summarize(r['fixed_sphere_z']),
          'fixed_radius_residual_p95_median':round(med(r['fixed_resid']),4) if r['fixed_resid'] else None,
          'fixed_radius_inlier_fraction_median':round(med(r['fixed_inlier']),3) if r['fixed_inlier'] else None,
          'valid_points_median':round(med(r['points']),1) if r['points'] else None}
  entry['targets'].append(target);print(clip['scene'],target)
 out['results'].append(entry)
path=ROOT/'media/frc-fuel-stereo-sphere-fit-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
