#!/usr/bin/env python3
"""Sweep vertical SGBM context pads on representative frames from both saved videos."""
import copy,json,math,os,statistics,time
from pathlib import Path
import cv2
import numpy as np
from scout.detectors import Detection
from scout.stereo import StereoDepth

ROOT=Path(__file__).resolve().parents[1]
DATA=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
TIMES=tuple(1.0+.4*i for i in range(15));PADS=(0,8,12,16,20,24)
TARGETS=((808,589),(902,591));CONF=.05;CAL='media/calibration/stereo_calibration_live_order_candidate.yml'
os.environ.update(SCOUT_CLASS_NAMES='["ball"]',SCOUT_STEREO_CALIBRATION=CAL,SCOUT_STEREO_RECTIFY='1',
 SCOUT_STEREO_BASELINE_M='.0635',SCOUT_STEREO_MATCH_SCALE='.5',SCOUT_STEREO_NUM_DISPARITIES='128',
 SCOUT_FUEL_DIAMETER_M='.15',SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR='.55',SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU='.90')
def med(v):return statistics.median(v) if v else None
def mad(v):
 m=med(v);return med([abs(x-m) for x in v]) if v else None
def match(boxes):
 out=[None,None]
 for b in sorted(boxes,key=lambda x:x.confidence,reverse=True):
  cx=(b.x1+b.x2)/2;cy=(b.y1+b.y2)/2;d=[math.hypot(cx-x,cy-y) for x,y in TARGETS];i=int(np.argmin(d))
  if d[i]<=60 and out[i] is None:out[i]=b
 return out
stereo=StereoDepth();output={'method':'For 15 stable timestamps in each saved stereo clip, run active confidence-0.05 detections and calibrated SGBM with full-frame or full-width vertical-band matching. Sweep top/bottom context pad; compare per-target range with same-frame full SGBM, valid paired 3D coverage, pair spacing, band height, and matcher time. All detections contribute to band extent.','times_s':list(TIMES),'vertical_pads_half_px':list(PADS),'results':[]}
for ci,clip in enumerate(DATA['results']):
 cap=cv2.VideoCapture(str(ROOT/clip['video']));fps=cap.get(cv2.CAP_PROP_FPS)
 rec={pad:{'ranges':[[],[]],'deltas':[[],[]],'pair_spacing':[],'latency':[],'band_height':[]} for pad in PADS}
 for t in TIMES:
  idx=round(t*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read()
  if not ok:continue
  row=clip['per_frame'][idx]
  all_boxes=[Detection(int(b['x1']),int(b['y1']),int(b['x2']),int(b['y2']),0,float(b['confidence'])) for b in row['detections'] if b['confidence']>=CONF]
  l,r=frame[:,:1600],frame[:,1600:]
  target_source=match(all_boxes)
  if not all(target_source):continue
  full_boxes=copy.deepcopy(all_boxes);stereo.vertical_roi_pad=0
  ts=time.perf_counter();stereo.estimate(l,r,full_boxes);full_ms=(time.perf_counter()-ts)*1000
  full_targets=match(full_boxes);ref=[b.distance_m for b in full_targets]
  # Record the full-frame matcher as the reference configuration.
  rec[0]['latency'].append(full_ms)
  for ti,b in enumerate(full_targets):
   if b and b.distance_m is not None:rec[0]['ranges'][ti].append(float(b.distance_m))
  if all(b and b.distance_m for b in full_targets):
   # Use the same camera projection as the runtime path for baseline pair spacing.
   pts=[]
   for b in full_targets:
    _,_,_,kl,r1,p1=stereo.maps;center=np.array([[[float(b.x1+b.x2)/2,float(b.y1+b.y2)/2]]],np.float32)
    uv=cv2.undistortPoints(center,kl,stereo.calibration['left_distortion'],R=r1,P=p1)[0,0];z=float(b.distance_m)
    pts.append(((uv[0]-p1[0,2])*z/p1[0,0],(uv[1]-p1[1,2])*z/p1[1,1],z))
   rec[0]['pair_spacing'].append(math.dist(pts[0],pts[1]))
  for pad in PADS[1:]:
   items=copy.deepcopy(all_boxes);stereo.vertical_roi_pad=pad
   ts=time.perf_counter();stereo.estimate(l,r,items);rec[pad]['latency'].append((time.perf_counter()-ts)*1000)
   targets=match(items);zs=[b.distance_m if b else None for b in targets]
   for ti,z in enumerate(zs):
    if z is not None:
     rec[pad]['ranges'][ti].append(float(z))
     if ref[ti] is not None:rec[pad]['deltas'][ti].append(float(z-ref[ti]))
   if all(z is not None for z in zs):
    pts=[]
    for b,z in zip(targets,zs):
     _,_,_,kl,r1,p1=stereo.maps;center=np.array([[[float(b.x1+b.x2)/2,float(b.y1+b.y2)/2]]],np.float32)
     uv=cv2.undistortPoints(center,kl,stereo.calibration['left_distortion'],R=r1,P=p1)[0,0]
     pts.append(((uv[0]-p1[0,2])*z/p1[0,0],(uv[1]-p1[1,2])*z/p1[1,1],z))
    rec[pad]['pair_spacing'].append(math.dist(pts[0],pts[1]))
   if stereo.maps:
    h=stereo.maps[0][1]*stereo.match_scale
    rect_y=[]
    for b in items:
     corners=np.array([[[b.x1,b.y1]],[[b.x2,b.y1]],[[b.x2,b.y2]],[[b.x1,b.y2]]],np.float32)
     q=cv2.undistortPoints(corners,stereo.maps[3],stereo.calibration['left_distortion'],R=stereo.maps[4],P=stereo.maps[5]).reshape(-1,2)*stereo.match_scale
     rect_y.extend((q[:,1].min(),q[:,1].max()))
    rec[pad]['band_height'].append(min(h,int(math.ceil(max(rect_y)))+pad)-max(0,int(math.floor(min(rect_y)))-pad))
 cap.release();entry={'scene':clip['scene'],'sampled_pairs':len(TIMES),'configs':[]}
 for pad in PADS:
  q=rec[pad];entry['configs'].append({'vertical_pad_half_px':pad,'valid_target_ranges':[len(x) for x in q['ranges']],
   'valid_pair_frames':len(q['pair_spacing']),'range_delta_vs_full_sgbm_m':[{'median':round(med(x),4),'mad':round(mad(x),4),'max_abs':round(max(abs(v) for v in x),4)} if x else None for x in q['deltas']],
   'pair_spacing_m':{'median':round(med(q['pair_spacing']),4),'mad':round(mad(q['pair_spacing']),4)} if q['pair_spacing'] else None,
   'matcher_ms_median':round(med(q['latency']),2) if q['latency'] else None,'matcher_ms_p95':round(sorted(q['latency'])[int(.95*(len(q['latency'])-1))],2) if q['latency'] else None,
   'band_height_half_px_median':round(med(q['band_height']),1) if q['band_height'] else 600})
  print(clip['scene'],entry['configs'][-1])
 output['results'].append(entry)
path=ROOT/'media/frc-fuel-stereo-vertical-pad-sweep.json';path.write_text(json.dumps(output,indent=2)+'\n');print('saved',path)
