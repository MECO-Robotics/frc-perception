#!/usr/bin/env python3
"""Evaluate current confidence/gate/vertical-band profile on all saved stereo frames."""
import json, math, os, statistics, time
from pathlib import Path
import cv2
import numpy as np
from scout.detectors import Detection
from scout.stereo import StereoDepth

ROOT=Path(__file__).resolve().parents[1]
DATA=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
CAL='media/calibration/stereo_calibration_live_order_candidate.yml'
os.environ.update(SCOUT_CLASS_NAMES='["ball"]',SCOUT_STEREO_CALIBRATION=CAL,SCOUT_STEREO_RECTIFY='1',
 SCOUT_STEREO_BASELINE_M='.0635',SCOUT_STEREO_MATCH_SCALE='.5',SCOUT_STEREO_NUM_DISPARITIES='128',
 SCOUT_STEREO_VERTICAL_ROI_PAD='24',SCOUT_FUEL_DIAMETER_M='.15',
 SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR='.55',SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU='.90')
CONF=.05;TARGETS=((808,589),(902,591));MATCH_RADIUS=60
RADII=tuple(round(i/1000,3) for i in range(180,321,5))
def med(v):return statistics.median(v) if v else None
def mad(v):
 m=med(v);return med([abs(x-m) for x in v]) if v else None
def summary(v):
 return {'n':len(v),'median_m':round(med(v),4),'mad_m':round(mad(v),4),
         'p95_m':round(sorted(v)[int(.95*(len(v)-1))],4),'max_m':round(max(v),4)} if v else None
def match(boxes):
 out=[None,None]
 for b in sorted(boxes,key=lambda item:item.confidence,reverse=True):
  cx=(b.x1+b.x2)/2;cy=(b.y1+b.y2)/2
  errors=[math.hypot(cx-x,cy-y) for x,y in TARGETS];i=int(np.argmin(errors))
  if errors[i]<=MATCH_RADIUS and out[i] is None:out[i]=b
 return out
stereo=StereoDepth()
results=[]
for clip in DATA['results']:
 cap=cv2.VideoCapture(str(ROOT/clip['video']));rows=clip['per_frame'];pair_dist=[];lateral=[];depth_gap=[]
 range_samples=[[],[]];piece_points=[[],[]];centers=[];count_hist={0:0,1:0,2:0};fp_total=gate_rejected=duplicate_rejected=valid_pair=0;matcher_ms=[]
 for index,row in enumerate(rows):
  ok,frame=cap.read()
  if not ok:break
  boxes=[Detection(int(b['x1']),int(b['y1']),int(b['x2']),int(b['y2']),0,float(b['confidence']))
         for b in row['detections'] if b['confidence']>=CONF]
  left,right=frame[:,:1600],frame[:,1600:]
  t=time.perf_counter();stereo.estimate(left,right,boxes);matcher_ms.append((time.perf_counter()-t)*1000)
  filtered=stereo.filter_fuel_size_inconsistent(boxes)
  gate_rejected+=stereo.last_size_gate_rejected;duplicate_rejected+=stereo.last_size_gate_duplicate_rejected
  chosen=match(filtered);num=sum(b is not None for b in chosen);count_hist[num]=count_hist.get(num,0)+1
  fp_total+=max(0,len(filtered)-num)
  points=[]
  for ti,b in enumerate(chosen):
   if b is None or b.distance_m is None:points.append(None);continue
   range_samples[ti].append(float(b.distance_m))
   center=np.array([[[float(b.x1+b.x2)/2,float(b.y1+b.y2)/2]]],np.float32)
   _,_,_,kl,r1,p1=stereo.maps
   uv=cv2.undistortPoints(center,kl,stereo.calibration['left_distortion'],R=r1,P=p1)[0,0]
   z=float(b.distance_m);point=((uv[0]-p1[0,2])*z/p1[0,0],(uv[1]-p1[1,2])*z/p1[1,1],z)
   points.append(point);piece_points[ti].append(point)
  if all(point is not None for point in points):
   valid_pair+=1;delta=[points[1][i]-points[0][i] for i in range(3)]
   d=math.sqrt(sum(x*x for x in delta));dxy=math.hypot(delta[0],delta[1])
   pair_dist.append(d);lateral.append(dxy);depth_gap.append(abs(delta[2]))
   centers.append(tuple((points[0][i]+points[1][i])/2 for i in range(3)))
 cap.release()
 radius_sweep=[]
 for radius in RADII:
  merged=sum(d<=radius for d in pair_dist)
  radius_sweep.append({'radius_m':radius,'merged_pairs':merged,'valid_pair_frames':len(pair_dist),
                       'merged_fraction':round(merged/len(pair_dist),5) if pair_dist else 0})
 result={'scene':clip['scene'],'frames_read':len(rows),'target_count_histogram':{str(k):v for k,v in count_hist.items()},
         'gate_rejected_boxes':gate_rejected,'duplicate_rejected_boxes':duplicate_rejected,'off_target_boxes_after_gate':fp_total,
         'valid_pair_3d_frames':valid_pair,'valid_pair_fraction':round(valid_pair/len(rows),5) if rows else 0,
         'range_by_target_m':[summary(v) for v in range_samples],'pair_distance_3d_m':summary(pair_dist),
         'lateral_xy_distance_m':summary(lateral),'depth_gap_m':summary(depth_gap),
         'cluster_centroid_xyz_camera_m':{axis:{'median':round(med([p[i] for p in centers]),4),'mad':round(mad([p[i] for p in centers]),4)} for i,axis in enumerate(('x','y','z'))} if centers else None,
         'matcher_ms_median':round(med(matcher_ms),2) if matcher_ms else None,
         'matcher_ms_p95':round(sorted(matcher_ms)[int(.95*(len(matcher_ms)-1))],2) if matcher_ms else None,
         'cluster_radius_sweep':radius_sweep}
 results.append(result);print(json.dumps(result,indent=2))
output={'method':'Replay all saved stereo frames using the active FRC profile settings: confidence >=0.05, calibrated SGBM 128 disparities at 0.5 scale, full-width vertical band pad 24, 15cm apparent-size/depth gate at 55% relative tolerance, and duplicate IoU 0.90 with 3D proximity suppression. Match outputs to manually labeled target windows, then sweep Euclidean 3D cluster radii.','profile':{'confidence':CONF,'vertical_roi_pad':24,'size_gate_relative_tolerance':.55,'duplicate_iou':.90,'stereo_baseline_m':.0635},'limitations':'Only two fixed adjacent balls at one pose/range; cluster count assumes the pair is one group. The recordings cannot measure false merges, moving-object behavior, robot-frame accuracy, or absolute metric range.','results':results}
path=ROOT/'media/frc-fuel-dense-active-profile-evaluation.json';path.write_text(json.dumps(output,indent=2)+'\n');print('saved',path)
