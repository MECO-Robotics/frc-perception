#!/usr/bin/env python3
"""Sweep FUEL cluster merge radii on the current NMS 0.15 saved-video outputs."""
import json, math, statistics
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DETECTIONS=json.loads((ROOT/'media/frc-fuel-dense-count-nms015.json').read_text())
STEREO=json.loads((ROOT/'media/frc-fuel-dense-stereo-nms015-conf005.json').read_text())
TARGETS=((808.,589.),(902.,591.)); RADII=(.22,.23,.24,.245,.25,.255,.26,.265,.27,.28)
fs=cv2.FileStorage(str(ROOT/'media/calibration/stereo_calibration_live_order_candidate.yml'),cv2.FILE_STORAGE_READ)
kl=fs.getNode('left_camera_matrix').mat();dl=fs.getNode('left_distortion').mat();kr=fs.getNode('right_camera_matrix').mat();dr=fs.getNode('right_distortion').mat();rot=fs.getNode('rotation').mat();trans=fs.getNode('translation').mat();fs.release()
r1,r2,p1,p2,*_=cv2.stereoRectify(kl,dl,kr,dr,(1600,1200),rot,trans,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
fx,fy,cx,cy=float(p1[0,0]),float(p1[1,1]),float(p1[0,2]),float(p1[1,2])
def medmad(v):
 if not v:return {'n':0}
 m=statistics.median(v);return {'n':len(v),'median_m':round(m,5),'mad_m':round(statistics.median(abs(x-m) for x in v),5),'p95_m':round(float(np.percentile(v,95)),5),'max_m':round(max(v),5)}
def choose(dets):
 out=[None,None]
 for b in sorted((b for b in dets if b['confidence']>=.05),key=lambda q:q['confidence'],reverse=True):
  x=(b['x1']+b['x2'])/2;y=(b['y1']+b['y2'])/2;dist,i=min((math.hypot(x-a,y-c),i) for i,(a,c) in enumerate(TARGETS))
  if dist<=60 and out[i] is None:out[i]=b
 return out
def xyz(box,z):
 x=(box['x1']+box['x2'])/2;y=(box['y1']+box['y2'])/2
 q=cv2.undistortPoints(np.array([[[x,y]]],np.float32),kl,dl,R=r1,P=p1)[0,0]
 return ((q[0]-cx)*z/fx,(q[1]-cy)*z/fy,z)
def axis_summary(points):
 if not points:return None
 labels=('x','y','z');out={}
 for j,label in enumerate(labels):
  values=[p[j] for p in points];m=statistics.median(values)
  out[label]={'median_m':round(m,5),'mad_m':round(statistics.median(abs(v-m) for v in values),5),
              'p95_abs_dev_m':round(float(np.percentile([abs(v-m) for v in values],95)),5)}
 return out
summary={'method':'Use current NMS 0.15 / confidence 0.05 detections and matching saved-video SGBM ranges; associate boxes to the two fixed visible targets, reconstruct rectified 3D box-center points, and sweep Euclidean merge radii.','limitations':'Only two static visible FUEL pieces are labeled. This measures merging of that known pair, not false merges across separate clusters; SGBM absolute ranges are not tape-verified.','detection_source':'media/frc-fuel-dense-count-nms015.json','stereo_source':'media/frc-fuel-dense-stereo-nms015-conf005.json','merge_radii_m':RADII,'results':[]}
for ci,clip in enumerate(DETECTIONS['results']):
 depth=STEREO['results'][ci];distances=[];pair_count=0;bothdet=0;frame_count=len(clip['per_frame']);centroids=[];points_by_target=[[],[]]
 for i,row in enumerate(clip['per_frame']):
  boxes=choose(row['detections']);bothdet+=all(boxes)
  ranges=depth['frames'][i]['ranges_m']
  if all(boxes) and all(z is not None for z in ranges):
   pts=[xyz(boxes[j],ranges[j]) for j in range(2)]
   distances.append(math.dist(pts[0],pts[1]));centroids.append(tuple((pts[0][j]+pts[1][j])/2 for j in range(3)))
   for j in range(2):points_by_target[j].append(pts[j])
   pair_count+=1
 measures=medmad(distances)
 measures['merges_by_radius']=[{'radius_m':r,'merged':sum(d<=r for d in distances),'valid_pairs':len(distances),'merge_fraction':round(sum(d<=r for d in distances)/len(distances),5) if distances else None} for r in RADII]
 summary['results'].append({'scene':clip['scene'],'frames':frame_count,'both_target_boxes':bothdet,'valid_3d_pairs':pair_count,'pair_separation':measures,'cluster_centroid_xyz_repeatability':axis_summary(centroids),'target_xyz_repeatability':[axis_summary(points) for points in points_by_target]})
 print(clip['scene'],json.dumps(summary['results'][-1]))
out=ROOT/'media/frc-fuel-current-cluster-radius.json';out.write_text(json.dumps(summary,indent=2)+'\n');print('saved',out)
