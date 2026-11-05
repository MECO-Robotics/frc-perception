#!/usr/bin/env python3
"""Compare raw SGBM, LR consistency and WLS confidence on saved stereo clips."""
import json
import math
import os
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from scout.detectors import TensorRTDetector

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_1280_fp16.engine'
CALIBRATION = ROOT / 'media/calibration/stereo_calibration_live_order_candidate.yml'
VIDEOS = ('media/frc-2026-live-two-fuel-60fps-cfr.mp4',
          'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4')
SAMPLE_TIMES = (1, 1.4, 1.8, 2.2, 2.6, 3, 3.4, 3.8, 4.2, 4.6, 5, 5.4, 5.8, 6.2, 6.6)
TARGETS = ((808, 589), (902, 591))
MIN_VALID = 8


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    m = median(values)
    return median([abs(x-m) for x in values]) if values else None


def box_roi(box, k, d, r, p, shape):
    x1, y1, x2, y2 = map(float, (box.x1, box.y1, box.x2, box.y2))
    corners = np.array([[[x1,y1]],[[x2,y1]],[[x2,y2]],[[x1,y2]]],np.float32)
    pts = cv2.undistortPoints(corners,k,d,R=r,P=p).reshape(-1,2)/2
    x1,y1=np.floor(pts.min(axis=0)).astype(int);x2,y2=np.ceil(pts.max(axis=0)).astype(int)
    h,w=shape
    return slice(max(0,y1),min(h,y2)),slice(max(0,x1),min(w,x2))


fs=cv2.FileStorage(str(CALIBRATION),cv2.FILE_STORAGE_READ)
kl=fs.getNode('left_camera_matrix').mat();dl=fs.getNode('left_distortion').mat()
kr=fs.getNode('right_camera_matrix').mat();dr=fs.getNode('right_distortion').mat()
rotation=fs.getNode('rotation').mat();translation=fs.getNode('translation').mat();fs.release()
r1,r2,p1,p2,_,_,_=cv2.stereoRectify(kl,dl,kr,dr,(1600,1200),rotation,translation,
                                      flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
ml=cv2.initUndistortRectifyMap(kl,dl,r1,p1,(1600,1200),cv2.CV_32FC1)
mr=cv2.initUndistortRectifyMap(kr,dr,r2,p2,(1600,1200),cv2.CV_32FC1)

os.environ['SCOUT_CLASS_NAMES']='["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO']='2.0'
detector=TensorRTDetector(str(ENGINE),confidence=0.10)
methods=['SGBM-full-box','LR-consistent-0.5px','WLS-lambda2000','WLS-lambda8000',
         'WLS-lambda16000','WLS8000-confidence64','WLS8000-confidence128','WLS8000-confidence192']
all_records=[]
for video_name in VIDEOS:
    cap=cv2.VideoCapture(str(ROOT/video_name))
    samples={method:{i:[] for i in range(2)} for method in methods}
    filter_times={method:[] for method in methods if method!='SGBM-full-box'}
    for sample_i, time_s in enumerate(SAMPLE_TIMES):
        cap.set(cv2.CAP_PROP_POS_MSEC,time_s*1000);ok,frame=cap.read()
        if not ok:continue
        left,right=frame[:,:1600],frame[:,1600:]
        detections,det_ms=detector.infer(left)
        selected=[None,None]
        for box in detections:
            cx,cy=(box.x1+box.x2)/2,(box.y1+box.y2)/2
            distances=[math.hypot(cx-x,cy-y) for x,y in TARGETS]
            dist,i=min((v,i) for i,v in enumerate(distances))
            if dist<=60 and (selected[i] is None or box.confidence>selected[i].confidence):
                selected[i]=box
        gl=cv2.cvtColor(left,cv2.COLOR_BGR2GRAY);gr=cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
        gl=cv2.remap(gl,ml[0],ml[1],cv2.INTER_LINEAR);gr=cv2.remap(gr,mr[0],mr[1],cv2.INTER_LINEAR)
        gl=cv2.resize(gl,(800,600),interpolation=cv2.INTER_AREA)
        gr=cv2.resize(gr,(800,600),interpolation=cv2.INTER_AREA)
        matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=5,
                    P1=8*5*5,P2=32*5*5,uniquenessRatio=8,speckleWindowSize=50,
                    speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        left_disp=matcher.compute(gl,gr)
        t0=time.perf_counter();right_matcher=cv2.ximgproc.createRightMatcher(matcher)
        right_disp=right_matcher.compute(gr,gl);filter_times['LR-consistent-0.5px'].append((time.perf_counter()-t0)*1000)
        lfloat=left_disp.astype(np.float32)/16; rfloat=right_disp.astype(np.float32)/16
        h,w=lfloat.shape;xx=np.broadcast_to(np.arange(w),(h,w));xr=np.rint(xx-lfloat).astype(np.int32)
        inbounds=(xr>=0)&(xr<w)&(lfloat>3)&(lfloat<120)
        sampled_r=np.zeros_like(rfloat);yy=np.broadcast_to(np.arange(h)[:,None],(h,w))
        sampled_r[inbounds]=rfloat[yy[inbounds],xr[inbounds]]
        lr_mask=inbounds&(np.abs(lfloat+sampled_r)<=0.5)
        wls_results={};confidence_maps={}
        for lam in (2000,8000,16000):
            wls=cv2.ximgproc.createDisparityWLSFilter(matcher)
            wls.setLambda(float(lam));wls.setSigmaColor(1.5)
            t0=time.perf_counter();filtered=wls.filter(left_disp,gl,None,right_disp)
            filter_times[f'WLS-lambda{lam}'].append((time.perf_counter()-t0)*1000)
            wls_results[lam]=filtered.astype(np.float32)/16
            if lam==8000:confidence_maps[lam]=wls.getConfidenceMap()
        for target_i,box in enumerate(selected):
            if box is None:continue
            ys,xs=box_roi(box,kl,dl,r1,p1,lfloat.shape)
            roi=lfloat[ys,xs]; raw_valid=np.isfinite(roi)&(roi>3)&(roi<120)
            candidates={
                'SGBM-full-box':(roi,raw_valid),
            }
            lroi=lr_mask[ys,xs]
            candidates['LR-consistent-0.5px']=(roi,raw_valid&lroi)
            for lam,dispmap in wls_results.items():
                q=dispmap[ys,xs];valid=np.isfinite(q)&(q>3)&(q<120)
                candidates[f'WLS-lambda{lam}']=(q,valid)
                if lam==8000:
                    conf=confidence_maps[lam][ys,xs]
                    for cutoff in (64,128,192):
                        candidates[f'WLS8000-confidence{cutoff}']=(q,valid&(conf>=cutoff))
            for method,(values,mask) in candidates.items():
                valid_values=values[mask]
                z=None
                if valid_values.size>=MIN_VALID:
                    disp_px=float(np.median(valid_values)*2)
                    z=0.0635*float(p1[0,0])/disp_px
                samples[method][target_i].append({'time_s':time_s,'z_m':z,
                    'valid_pixels':int(valid_values.size),'confidence':box.confidence,
                    'center':[(box.x1+box.x2)/2,(box.y1+box.y2)/2]})
    cap.release()
    for method in methods:
        per_target=[]
        for target_i in range(2):
            entries=samples[method][target_i];zs=[x['z_m'] for x in entries if x['z_m'] is not None]
            per_target.append({'target':target_i+1,'detections':len(entries),'valid_ranges':len(zs),
                'median_z_m':round(median(zs),3) if zs else None,'mad_z_m':round(mad(zs),3) if zs else None,
                'median_valid_pixels':round(median([x['valid_pixels'] for x in entries]),1) if entries else None})
        lookup=[{x['time_s']:x for x in samples[method][i] if x['z_m'] is not None} for i in range(2)]
        distances=[]
        for t in set(lookup[0])&set(lookup[1]):
            pts=[]
            for i in range(2):
                entry=lookup[i][t];u,v=entry['center']
                rect=cv2.undistortPoints(np.array([[[u,v]]],np.float32),kl,dl,R=r1,P=p1)[0,0]
                z=entry['z_m'];pts.append(((rect[0]-p1[0,2])*z/p1[0,0],(rect[1]-p1[1,2])*z/p1[1,1],z))
            distances.append(math.sqrt(sum((pts[1][j]-pts[0][j])**2 for j in range(3))))
        timing_key = ('WLS-lambda8000' if method.startswith('WLS8000-confidence') else
                      method if method in filter_times else None)
        filter_ms = median(filter_times[timing_key]) if timing_key else 0.0
        all_records.append({'video':video_name,'method':method,'targets':per_target,
            'pair_3d_spacing_m':{'n':len(distances),'median':round(median(distances),3),
                'mad':round(mad(distances),3)} if distances else None,
            'filter_ms_median':round(filter_ms,2) if filter_ms is not None else None})
detector.close()
out={'method':'FRC 9550 YOLO26n TensorRT 1280 detections at confidence 0.10 and box aspect <=2, 15 stereo pairs per video. Rectified full-box SGBM b5/u8 baseline compared with left-right consistency and OpenCV ximgproc WLS configurations. Depth uses same 63.5 mm baseline and rectified focal length.','limitations':'Stationary two-ball room clips only; no labeled disparity/depth ground truth. Filter tuning compares temporal repeatability and point spacing, which cannot establish absolute range accuracy. WLS CPU costs are offline and do not include the full robot pipeline.','configuration':{'confidence':0.10,'max_box_aspect_ratio':2.0,'sgbm_block_size':5,'uniqueness_ratio':8,'wls_sigma_color':1.5,'confidence_cutoffs':[64,128,192],'left_right_consistency_threshold_px':0.5},'results':all_records}
(ROOT/'media/frc-stereo-confidence-sweep.json').write_text(json.dumps(out,indent=2)+'\n')
print('saved',ROOT/'media/frc-stereo-confidence-sweep.json','records',len(all_records))
