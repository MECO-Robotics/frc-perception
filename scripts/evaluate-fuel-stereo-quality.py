#!/usr/bin/env python3
"""Evaluate disparity support and spread as range-quality indicators under occlusion."""
import json, math, os, statistics
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
TIMES=tuple(1.0+.4*i for i in range(15));FRACS=(0,.25,.5,.75);PADS=(8,24)
TARGETS=((808,589),(902,591));BASE=.0635;DIAM=.15
CAL='media/calibration/stereo_calibration_live_order_candidate.yml'
fs=cv2.FileStorage(str(ROOT/CAL),cv2.FILE_STORAGE_READ)
kl=fs.getNode('left_camera_matrix').mat();dl=fs.getNode('left_distortion').mat()
kr=fs.getNode('right_camera_matrix').mat();dr=fs.getNode('right_distortion').mat()
rotation=fs.getNode('rotation').mat();translation=fs.getNode('translation').mat();fs.release()
size=(1600,1200);r1,r2,p1,p2,*_=cv2.stereoRectify(kl,dl,kr,dr,size,rotation,translation,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
ml=cv2.initUndistortRectifyMap(kl,dl,r1,p1,size,cv2.CV_32FC1)
mr=cv2.initUndistortRectifyMap(kr,dr,r2,p2,size,cv2.CV_32FC1)
matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=5,P1=200,P2=800,uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
right_matcher=cv2.StereoSGBM_create(minDisparity=-128,numDisparities=128,blockSize=5,P1=200,P2=800,uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
FOCAL=float(p1[0,0]);RAW_FOCAL=float(kl[0,0]);
def match(row):
 out=[None,None]
 for b in sorted((b for b in row['detections'] if b['confidence']>=.1),key=lambda x:x['confidence'],reverse=True):
  cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
  ds=[math.hypot(cx-x,cy-y) for x,y in TARGETS];i=min(range(2),key=lambda j:ds[j])
  if ds[i]<=60 and out[i] is None:out[i]=b
 return out
def rect_box(b):
 corners=np.array([[[b['x1'],b['y1']]],[[b['x2'],b['y1']]],[[b['x2'],b['y2']]],[[b['x1'],b['y2']]]],np.float32)
 q=cv2.undistortPoints(corners,kl,dl,R=r1,P=p1).reshape(-1,2)/2
 return (max(0,int(np.floor(q[:,0].min()))),max(0,int(np.floor(q[:,1].min()))),
         min(800,int(np.ceil(q[:,0].max()))),min(600,int(np.ceil(q[:,1].max()))))
def metrics(disp,box,yoffset=0,right_disp=None):
 x1,y1,x2,y2=box; roi=disp[max(0,y1-yoffset):max(0,y2-yoffset),x1:x2]
 alln=roi.size;valid=roi[np.isfinite(roi)&(roi>3)&(roi<120)]
 if not valid.size:return {'range_m':None,'valid_fraction':0.0,'valid_pixels':0,'pixels':alln,'mad_px':None,'iqr_px':None,'dominant_bin_fraction':0.0,'lr_consistency_fraction':0.0,'lr_residual_median_px':None}
 m=float(np.median(valid));mad=float(np.median(np.abs(valid-m)));q25,q75=np.percentile(valid,(25,75))
 bins=np.round(valid*2).astype(np.int32);_,counts=np.unique(bins,return_counts=True)
 lr_fraction=None;lr_residual=None
 if right_disp is not None:
  ys,xs=np.nonzero(np.isfinite(roi)&(roi>3)&(roi<120));xs=xs+x1
  ds=roi[np.isfinite(roi)&(roi>3)&(roi<120)];xr=np.rint(xs-ds).astype(np.int32)
  inside=(xr>=0)&(xr<right_disp.shape[1]);rr=[]
  if np.any(inside):
   rd=right_disp[ys[inside]+y1-yoffset,xr[inside]]
   good=np.isfinite(rd)&(rd>-120)&(rd<-3)
   rr=np.abs(ds[inside][good]+rd[good])
  lr_fraction=float(np.sum(rr<=1.0)/len(valid)) if len(valid) else 0.0
  lr_residual=float(np.median(rr)) if len(rr) else None
 return {'range_m':BASE*FOCAL/(2*m),'valid_fraction':float(len(valid)/alln) if alln else 0.0,
         'valid_pixels':int(len(valid)),'pixels':int(alln),'mad_px':mad,'iqr_px':float(q75-q25),
         'dominant_bin_fraction':float(counts.max()/len(valid)),
         'lr_consistency_fraction':lr_fraction,'lr_residual_median_px':lr_residual}
def summary(vals):
 return round(float(np.median(vals)),4) if vals else None
def evaluate(rows,method):
 subset=[r for r in rows if r['method']==method]
 # A corrupted range is defined as >5 cm from same-frame unmasked full-frame SGBM.
 good=[r for r in subset if r['abs_error_m'] is not None and r['abs_error_m']<=.05]
 bad=[r for r in subset if r['abs_error_m'] is not None and r['abs_error_m']>.05]
 tests=[]
 for feature,direction,thresholds in (
  ('valid_fraction','low',[i/100 for i in range(0,101,2)]),
  ('mad_px','high',[i/20 for i in range(0,61)]),
  ('iqr_px','high',[i/10 for i in range(0,121)]),
  ('lr_consistency_fraction','low',[i/100 for i in range(0,101,2)]),
  ('lr_residual_median_px','high',[i/4 for i in range(0,33)]),
  ('dominant_bin_fraction','low',[i/100 for i in range(0,101,2)]),
  ('size_stereo_error','high',[i/100 for i in range(0,101,2)])):
  observed=[r[feature] for r in subset if r.get(feature) is not None]
  if not observed:continue
  best=None;best_metric=None
  for threshold in thresholds:
   def reject(r):
    v=r.get(feature)
    return v is not None and (v<threshold if direction=='low' else v>threshold)
   fp=sum(reject(r) for r in good);tp=sum(reject(r) for r in bad)
   fpr=fp/len(good) if good else 0;recall=tp/len(bad) if bad else 0
   point={'threshold':threshold,'good_rejected':fp,'good_total':len(good),'bad_rejected':tp,'bad_total':len(bad),'false_reject_rate':round(fpr,4),'bad_range_recall':round(recall,4)}
   if fpr<=.05 and (best_metric is None or recall>best_metric[0] or (recall==best_metric[0] and fpr<best_metric[1])):
    best=point;best_metric=(recall,fpr)
  tests.append({'feature':feature,'direction':direction,'best_at_max_5pct_false_reject':best})
 combined=[]
 for feature_a,dir_a,thresholds_a,feature_b,dir_b,thresholds_b in (
  ('mad_px','high',[i/4 for i in range(0,17)],'size_stereo_error','high',[i/100 for i in range(0,21)]),
  ('mad_px','high',[i/4 for i in range(0,17)],'lr_consistency_fraction','low',[i/100 for i in range(0,101,2)])):
  best=None;best_metric=None
  for threshold_a in thresholds_a:
   for threshold_b in thresholds_b:
    def reject(r):
     a,b=r.get(feature_a),r.get(feature_b)
     return ((a is not None and (a>threshold_a if dir_a=='high' else a<threshold_a)) or
             (b is not None and (b>threshold_b if dir_b=='high' else b<threshold_b)))
    fp=sum(reject(r) for r in good);tp=sum(reject(r) for r in bad)
    fpr=fp/len(good) if good else 0;recall=tp/len(bad) if bad else 0
    point={'threshold_a':threshold_a,'threshold_b':threshold_b,'good_rejected':fp,'good_total':len(good),
           'bad_rejected':tp,'bad_total':len(bad),'false_reject_rate':round(fpr,4),'bad_range_recall':round(recall,4)}
    if fpr<=.05 and (best_metric is None or recall>best_metric[0] or
                     (recall==best_metric[0] and fpr<best_metric[1])):
     best=point;best_metric=(recall,fpr)
  combined.append({'features':[[feature_a,dir_a],[feature_b,dir_b]],
                   'best_at_max_5pct_false_reject':best})
 return {'method':method,'good_ranges_le_5cm_error':len(good),'bad_ranges_gt_5cm_error':len(bad),
         'feature_summaries':{feature:{'good_median':summary([r[feature] for r in good if r.get(feature) is not None]),
                                      'bad_median':summary([r[feature] for r in bad if r.get(feature) is not None])}
                             for feature in ('valid_fraction','mad_px','iqr_px','lr_consistency_fraction','lr_residual_median_px','dominant_bin_fraction','size_stereo_error')},
         'threshold_sweeps':tests,'combined_threshold_sweeps':combined}
out={'method':'For matched clean detections on 15 stable saved frames per video, compare per-box SGBM quality metrics with the same-frame unmasked full-frame stereo range. Mask lower fractions of the target in both stereo views, shifting the right mask by the measured disparity. Evaluate full-frame and full-width vertical-band maps with the listed pads. Left-right consistency uses reverse SGBM and flags a left disparity pixel consistent when |d_left + d_right(x-d_left)| <= 1 half-resolution pixel. A bad range is >5cm from the unmasked full-frame estimate; this is an occlusion-induced-error label, not metric ground truth.','vertical_roi_pads_half_px':list(PADS),'bad_range_error_threshold_m':.05,'left_right_consistency_threshold_half_px':1.0,'occlusion_fractions':list(FRACS),'results':[],'threshold_evaluation':[]}
for ci,clip in enumerate(DATA['results']):
 cap=cv2.VideoCapture(str(ROOT/clip['video']));fps=cap.get(cv2.CAP_PROP_FPS)
 for t in TIMES:
  idx=round(t*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read()
  if not ok:continue
  boxes=match(clip['per_frame'][idx])
  if not all(boxes):continue
  rects=[rect_box(b) for b in boxes]
  left,right=frame[:,:1600].copy(),frame[:,1600:].copy()
  gl=cv2.resize(cv2.cvtColor(cv2.remap(left,*ml,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
  gr=cv2.resize(cv2.cvtColor(cv2.remap(right,*mr,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
  clean_disp=matcher.compute(gl,gr).astype(np.float32)/16
  clean_right=right_matcher.compute(gr,gl).astype(np.float32)/16
  clean=[metrics(clean_disp,b,right_disp=clean_right) for b in rects]
  clean_bands={}
  for pad in PADS:
   y0=max(0,min(b[1] for b in rects)-pad);y1=min(600,max(b[3] for b in rects)+pad)
   dmap=matcher.compute(gl[y0:y1,:],gr[y0:y1,:]).astype(np.float32)/16
   rmap=right_matcher.compute(gr[y0:y1,:],gl[y0:y1,:]).astype(np.float32)/16
   clean_bands[pad]=(y0,y1,[metrics(dmap,b,y0,rmap) for b in rects])
  for frac in FRACS:
   lcopy=left.copy();rcopy=right.copy()
   if frac:
    for b,z in zip(boxes,[q['range_m'] for q in clean]):
     cover=max(1,round((b['y2']-b['y1'])*frac));ya=max(0,b['y2']-cover);yb=min(1200,b['y2'])
     lcopy[ya:yb,b['x1']:b['x2']]=0;disp_full=BASE*FOCAL/z
     rx1=max(0,round(b['x1']-disp_full));rx2=min(1600,round(b['x2']-disp_full));rcopy[ya:yb,rx1:rx2]=0
   mlgray=cv2.resize(cv2.cvtColor(cv2.remap(lcopy,*ml,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
   mrgray=cv2.resize(cv2.cvtColor(cv2.remap(rcopy,*mr,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
   full=matcher.compute(mlgray,mrgray).astype(np.float32)/16
   full_right=right_matcher.compute(mrgray,mlgray).astype(np.float32)/16
   bands={}
   for pad in PADS:
    by0=max(0,min(b[1] for b in rects)-pad);by1=min(600,max(b[3] for b in rects)+pad)
    bands[pad]=(by0,matcher.compute(mlgray[by0:by1,:],mrgray[by0:by1,:]).astype(np.float32)/16,
                right_matcher.compute(mrgray[by0:by1,:],mlgray[by0:by1,:]).astype(np.float32)/16)
   for ti,(box,rect) in enumerate(zip(boxes,rects)):
    size_z=RAW_FOCAL*DIAM/math.sqrt((box['x2']-box['x1'])*(box['y2']-box['y1']))
    methods=[('full_frame',full,full_right,0)]
    methods.extend(('vertical_band_%d'%pad,bands[pad][1],bands[pad][2],bands[pad][0]) for pad in PADS)
    for method,dmap,rmap,yoff in methods:
     q=metrics(dmap,rect,yoff,rmap);err=abs(q['range_m']-clean[ti]['range_m']) if q['range_m'] is not None else None
     quality=dict(q,method=method,scene=clip['scene'],time_s=t,target=ti+1,occlusion_fraction=frac,
                  clean_range_m=clean[ti]['range_m'],clean_band_range_m=clean_bands[int(method.split('_')[-1])][2][ti] if method!='full_frame' else clean[ti]['range_m'],
                  abs_error_m=err,size_stereo_error=abs(size_z-q['range_m'])/q['range_m'] if q['range_m'] else None)
     out['results'].append(quality)
 cap.release()
for method in ('full_frame',)+tuple('vertical_band_%d'%pad for pad in PADS):
 out['threshold_evaluation'].append(evaluate(out['results'],method))
path=ROOT/'media/frc-fuel-stereo-quality-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n')
for item in out['threshold_evaluation']:print(json.dumps(item,indent=2))
print('saved',path)
