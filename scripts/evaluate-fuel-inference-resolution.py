#!/usr/bin/env python3
"""Compare FRC YOLO26n TensorRT input resolutions on identical saved stereo frames."""
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
CALIBRATION = ROOT / 'media/calibration/stereo_calibration_live_order_candidate.yml'
VARIANTS = (
    ('640', 'models/rebuilt_fuel_yolo26n_640_fp16.engine'),
    ('1280', 'models/rebuilt_fuel_yolo26n_1280_fp16.engine'),
    ('1280x960', 'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'),
    ('1600', 'models/rebuilt_fuel_yolo26n_1600_fp16.engine'),
)
VIDEOS = (
    ('earlier', 'media/frc-2026-live-two-fuel-60fps-cfr.mp4'),
    ('corrected-settings', 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4'),
)
SAMPLE_TIMES = (1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.4, 5.8, 6.2, 6.6)
TARGETS = ((808, 589), (902, 591))
CONFIDENCES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
               0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
BASELINE_M = 0.0635
FUEL_DIAMETER_M = 0.15


def median(values):
    return statistics.median(values) if values else None


def mad(values):
    center = median(values)
    return median([abs(value-center) for value in values]) if values else None


def range_summary(values):
    return {'median_m': round(median(values), 4), 'mad_m': round(mad(values), 4)} if values else None


def box_range(box, disparity_map, kl, dl, r1, p1):
    raw_x1,raw_y1,raw_x2,raw_y2=(box['x1'],box['y1'],box['x2'],box['y2'])
    corners = np.array([[[raw_x1, raw_y1]], [[raw_x2, raw_y1]],
                        [[raw_x2, raw_y2]], [[raw_x1, raw_y2]]], dtype=np.float32)
    rect = cv2.undistortPoints(corners, kl, dl, R=r1, P=p1).reshape(-1, 2) / 2.0
    roi_x1, roi_y1 = np.floor(rect.min(axis=0)).astype(int)
    roi_x2, roi_y2 = np.ceil(rect.max(axis=0)).astype(int)
    height, width = disparity_map.shape
    roi = disparity_map[max(0,roi_y1):min(height,roi_y2), max(0,roi_x1):min(width,roi_x2)]
    valid = roi[np.isfinite(roi) & (roi > 3.0) & (roi < 120.0)]
    if valid.size < 8:
        return None
    disparity_px = float(np.median(valid) * 2.0)
    distance_m = BASELINE_M * float(p1[0,0]) / disparity_px
    raw_k = 1046.1599129444185
    size_proxy_m = raw_k * FUEL_DIAMETER_M / math.sqrt(
        max(1.0, (raw_x2-raw_x1)*(raw_y2-raw_y1)))
    center = np.array([[[ (raw_x1+raw_x2)/2, (raw_y1+raw_y2)/2 ]]], np.float32)
    rect_center = cv2.undistortPoints(center,kl,dl,R=r1,P=p1)[0,0]
    return {'distance_m':distance_m,'disparity_px':disparity_px,
            'size_proxy_m':size_proxy_m,
            'center_rectified_px':[float(rect_center[0]),float(rect_center[1])],
            'valid_disparity_pixels':int(valid.size)}


def evaluate_frames(frames, variant, confidence):
    tp=fp=fn=pair=exact=0
    latency=[]
    target_ranges=[[],[]]
    size_deltas=[[],[]]
    pair_spacing=[]
    target_short_sides=[[],[]]
    frame_rows=[]
    for sample in frames:
        filtered=[box for box in sample['detections'][variant]
                  if box['confidence']>=confidence]
        filtered.sort(key=lambda box:box['confidence'],reverse=True)
        matched=[None,None]
        extras=0
        for box in filtered:
            cx=(box['x1']+box['x2'])/2
            cy=(box['y1']+box['y2'])/2
            errors=[math.hypot(cx-x,cy-y) for x,y in TARGETS]
            error,target_i=min((value,i) for i,value in enumerate(errors))
            if error<=60 and matched[target_i] is None:
                matched[target_i]=box
            else:
                extras+=1
        hits=sum(item is not None for item in matched)
        tp+=hits; fn+=2-hits; fp+=extras
        pair+=all(item is not None for item in matched)
        exact+=all(item is not None for item in matched) and extras==0
        latency.append(sample['latency_ms'][variant])
        target_metrics=[]
        points=[]
        for target_i,box in enumerate(matched):
            if box is None:
                target_metrics.append(None);points.append(None);continue
            short_side=min(box['x2']-box['x1'],box['y2']-box['y1'])
            target_short_sides[target_i].append(short_side)
            geometry=box_range(box=box, disparity_map=sample['disparity_map'],
                               kl=sample['kl'],dl=sample['dl'],r1=sample['r1'],p1=sample['p1'])
            target_metrics.append(geometry)
            if geometry is not None:
                target_ranges[target_i].append(geometry['distance_m'])
                size_deltas[target_i].append(abs(geometry['distance_m']-geometry['size_proxy_m']))
                u,v=geometry['center_rectified_px'];z=geometry['distance_m'];p1=sample['p1']
                points.append(((u-p1[0,2])*z/p1[0,0],(v-p1[1,2])*z/p1[1,1],z))
            else:
                points.append(None)
        spacing=None
        if all(point is not None for point in points):
            spacing=math.dist(points[0],points[1]);pair_spacing.append(spacing)
        frame_rows.append({'time_s':sample['time_s'],'target_hits':hits,'extra_boxes':extras,
                           'inference_latency_ms':sample['latency_ms'][variant],
                           'target_confidences':[item['confidence'] if item else None for item in matched],
                           'target_rectified_centers_px':[item['center_rectified_px']
                               if item is not None else None for item in target_metrics],
                           'target_short_sides_px':[min(item['x2']-item['x1'],item['y2']-item['y1'])
                                                    if item else None for item in matched],
                           'target_ranges':[item['distance_m'] if item else None for item in target_metrics],
                           'pair_spacing_m':spacing})
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0
    return {'confidence':confidence,'frames':len(frames),'tp':tp,'fp':fp,'fn':fn,
            'paired_frames':pair,'exact_two_frames':exact,'precision':round(precision,4),
            'recall':round(recall,4),'f1':round(f1,4),
            'median_inference_ms':round(median(latency),2) if latency else None,
            'p95_inference_ms':round(sorted(latency)[min(len(latency)-1,math.ceil(.95*len(latency))-1)],2)
                if latency else None,
            'target_ranges':[{'target':i+1,'valid_samples':len(target_ranges[i]),
                'range':range_summary(target_ranges[i]),
                'median_abs_delta_to_size_proxy_m':round(median(size_deltas[i]),4)
                    if size_deltas[i] else None,
                'median_short_side_px':round(median(target_short_sides[i]),1)
                    if target_short_sides[i] else None} for i in range(2)],
            'paired_3d_spacing':range_summary(pair_spacing),
            'merged_pair_frames_at_radius_m':{str(radius):sum(value<=radius for value in pair_spacing)
                for radius in (0.15,0.20,0.25)},
            'sample_frames':frame_rows}


fs=cv2.FileStorage(str(CALIBRATION),cv2.FILE_STORAGE_READ)
kl=fs.getNode('left_camera_matrix').mat();dl=fs.getNode('left_distortion').mat()
kr=fs.getNode('right_camera_matrix').mat();dr=fs.getNode('right_distortion').mat()
rotation=fs.getNode('rotation').mat();translation=fs.getNode('translation').mat();fs.release()
r1,r2,p1,p2,*_=cv2.stereoRectify(kl,dl,kr,dr,(1600,1200),rotation,translation,
                                  flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
map_left=cv2.initUndistortRectifyMap(kl,dl,r1,p1,(1600,1200),cv2.CV_32FC1)
map_right=cv2.initUndistortRectifyMap(kr,dr,r2,p2,(1600,1200),cv2.CV_32FC1)
matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=5,P1=200,P2=800,
    uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
os.environ['SCOUT_CLASS_NAMES']='["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO']='2.0'
detectors={name:TensorRTDetector(str(ROOT/path),confidence=0.05) for name,path in VARIANTS}
warmup_capture=cv2.VideoCapture(str(ROOT/VIDEOS[-1][1]))
warmup_capture.set(cv2.CAP_PROP_POS_MSEC,500)
warmup_ok,warmup_frame=warmup_capture.read();warmup_capture.release()
warmup_latencies={}
if warmup_ok:
    warmup_left=warmup_frame[:,:1600]
    for variant,_ in VARIANTS:
        _,warmup_latencies[variant]=detectors[variant].infer(warmup_left)
frames_by_video={}
for scene,video_name in VIDEOS:
    capture=cv2.VideoCapture(str(ROOT/video_name))
    frames=[]
    for time_s in SAMPLE_TIMES:
        capture.set(cv2.CAP_PROP_POS_MSEC,time_s*1000)
        ok,frame=capture.read()
        if not ok:continue
        left,right=frame[:,:1600],frame[:,1600:]
        detections={};latencies={}
        for variant,_ in VARIANTS:
            found,elapsed=detectors[variant].infer(left)
            detections[variant]=[{'x1':box.x1,'y1':box.y1,'x2':box.x2,'y2':box.y2,
                'confidence':box.confidence} for box in found]
            latencies[variant]=elapsed
        rect_left=cv2.cvtColor(cv2.remap(left,map_left[0],map_left[1],cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY)
        rect_right=cv2.cvtColor(cv2.remap(right,map_right[0],map_right[1],cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY)
        rect_left=cv2.resize(rect_left,(800,600),interpolation=cv2.INTER_AREA)
        rect_right=cv2.resize(rect_right,(800,600),interpolation=cv2.INTER_AREA)
        disparity=matcher.compute(rect_left,rect_right).astype(np.float32)/16.0
        frames.append({'time_s':time_s,'detections':detections,'latency_ms':latencies,
            'disparity_map':disparity,'kl':kl,'dl':dl,'r1':r1,'p1':p1})
    capture.release()
    frames_by_video[scene]=frames
for detector in detectors.values():detector.close()

groups=[]
for scene,video_name in VIDEOS:
    frames=frames_by_video[scene]
    for variant,engine in VARIANTS:
        groups.append({'scene':scene,'video':video_name,'variant':variant,'engine':engine,
            'input_resolution':variant if 'x' in variant else f'{variant}x{variant}','outputs':len(frames),
            'threshold_sweep':[evaluate_frames(frames,variant,confidence) for confidence in CONFIDENCES]})
output={'method':'Run identical FRC 9550 YOLO26n weights through 640, 1280 square, 1280x960 4:3, and native 1600 square FP16 TensorRT engines on the same synchronized frames per clip. Detector base threshold 0.05 captures candidates, then operating confidence is swept from 0.10 through 0.80. Aspect limit 2.0 is fixed. SGBM b5/u8 ranges and camera-relative pair spacing are computed from the same stereo frames.',
 'limitations':'Two static balls at about 1.4-1.6 m only. Fixed target windows are scene-specific. Stereo and apparent-size values are not measured ground truth. The native 1600 FP16 engine emitted TensorRT warnings for subnormal weights during build; its outputs should be treated as provisional until sanity checked beyond these clips.',
 'configuration':{'checkpoint':'models/rebuilt_fuel_yolo26n.pt','calibration':str(CALIBRATION.relative_to(ROOT)),
  'sample_times_s':list(SAMPLE_TIMES),'median_single_view_detector_ms':{
   variant:round(median([frame['latency_ms'][variant] for group in frames_by_video.values()
                         for frame in group]),2) for variant,_ in VARIANTS},
  'excluded_warmup_inference_ms':{key:round(value,2) for key,value in warmup_latencies.items()}},
 'results':groups}
out_path=ROOT/'media/frc-fuel-inference-resolution-sweep.json'
out_path.write_text(json.dumps(output,indent=2)+'\n')
for group in groups:
 print(group['scene'],group['variant'])
 for row in group['threshold_sweep']:
  print(row['confidence'],'P/R/F1',row['precision'],row['recall'],row['f1'],
   'pair/exact',row['paired_frames'],row['exact_two_frames'],'lat ms',row['median_inference_ms'],row['p95_inference_ms'],
   'ranges',[(x['range']['median_m'] if x['range'] else None,x['range']['mad_m'] if x['range'] else None)
             for x in row['target_ranges']],'spacing',row['paired_3d_spacing'])
