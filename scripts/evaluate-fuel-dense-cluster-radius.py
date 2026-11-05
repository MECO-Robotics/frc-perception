#!/usr/bin/env python3
"""Sweep 3D proximity cluster radii over all valid saved-video FUEL detections."""
import json, math, statistics
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
S=json.load(open(ROOT/'media/frc-fuel-dense-stereo-evaluation.json'))
CAL=ROOT/'media/calibration/stereo_calibration_live_order_candidate.yml'
TARGETS=((808,589),(902,591));CONF=.10;MATCH_PX=60
RADII=tuple(round(i/1000,3) for i in range(180,321,5))
fs=cv2.FileStorage(str(CAL),cv2.FILE_STORAGE_READ)
k=fs.getNode('left_camera_matrix').mat();dist=fs.getNode('left_distortion').mat()
kr=fs.getNode('right_camera_matrix').mat();dr=fs.getNode('right_distortion').mat()
rotation=fs.getNode('rotation').mat();translation=fs.getNode('translation').mat();fs.release()
r1,r2,p1,p2,*_=cv2.stereoRectify(k,dist,kr,dr,(1600,1200),rotation,translation,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
def med(x):return statistics.median(x) if x else None
def mad(x):
 m=med(x);return med([abs(v-m) for v in x]) if x else None
def point_summary(points):
 return {axis:{'median_m':round(med([point[i] for point in points]),4),
              'mad_m':round(mad([point[i] for point in points]),4)}
         for i,axis in enumerate(('x','y','z'))} if points else None
def match(row):
 out=[None,None]
 for b in sorted((b for b in row['detections'] if b['confidence']>=CONF),key=lambda x:x['confidence'],reverse=True):
  cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
  errs=[math.hypot(cx-x,cy-y) for x,y in TARGETS];i=int(np.argmin(errs))
  if errs[i]<=MATCH_PX and out[i] is None:out[i]=b
 return out
output={'method':'Use full dense FRC9550 confidence >=0.10 matched detections and saved production SGBM ranges; unproject box centers through the rectified candidate calibration. Sweep DBSCAN-equivalent min_samples=1 Euclidean radii, treating the two known adjacent balls as one expected visual cluster. The clips contain no unrelated pieces, so they can measure split sensitivity but not false merges.','confidence':CONF,'match_radius_px':MATCH_PX,'radii_m':list(RADII),'limitations':'Two adjacent stationary balls at one pose/range only. Cluster labels assume these two should form one group; unrelated-piece false merges and absolute positions are unmeasured. Stereo ranges have no tape-measured ground truth.','results':[]}
for ci,clip in enumerate(D['results']):
 frame_points=[];piece_xyz=[[],[]];distances=[];lateral=[];depth_gaps=[];pair_centers=[];lost_count=0
 for row,srow in zip(clip['per_frame'],S['results'][ci]['frames']):
  boxes=match(row);pts=[]
  for ti,b in enumerate(boxes):
   z=srow['ranges_m'][ti]
   if b is None or z is None:pts.append(None);continue
   cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
   uv=cv2.undistortPoints(np.array([[[cx,cy]]],np.float32),k,dist,R=r1,P=p1)[0,0]
   point=((uv[0]-p1[0,2])*z/p1[0,0],(uv[1]-p1[1,2])*z/p1[1,1],z)
   pts.append(point);piece_xyz[ti].append(point)
  if not all(pts):lost_count+=1;continue
  delta=[pts[1][i]-pts[0][i] for i in range(3)]
  d=math.sqrt(sum(v*v for v in delta));dxy=math.hypot(delta[0],delta[1])
  distances.append(d);lateral.append(dxy);depth_gaps.append(abs(delta[2]))
  center=tuple((pts[0][i]+pts[1][i])/2 for i in range(3));pair_centers.append(center)
  frame_points.append({'frame':row['frame'],'time_s':row['time_s'],'pair_distance_m':d,'cluster_centroid_camera_m':center})
 config=[]
 for radius in RADII:
  merged=sum(d<=radius for d in distances)
  config.append({'radius_m':radius,'merged_pairs':merged,'valid_pair_frames':len(distances),
                 'merged_fraction':round(merged/len(distances),5) if distances else 0,
                 'cluster_count_histogram':{'1':merged,'2':len(distances)-merged}})
 result={'scene':clip['scene'],'frames':len(clip['per_frame']),'valid_pair_frames':len(distances),'frames_without_two_valid_3d_points':lost_count,
         'pair_distance_m':{'median':round(med(distances),4),'mad':round(mad(distances),4),'p95':round(sorted(distances)[int(.95*(len(distances)-1))],4),'max':round(max(distances),4)} if distances else None,
         'lateral_xy_distance_m':{'median':round(med(lateral),4),'mad':round(mad(lateral),4)} if lateral else None,
         'depth_gap_m':{'median':round(med(depth_gaps),4),'mad':round(mad(depth_gaps),4)} if depth_gaps else None,
         'cluster_centroid_camera_xyz_repeatability_m':point_summary(pair_centers),
         'per_piece_median_camera_xyz_m':[{'x':round(med([p[0] for p in points]),3),'y':round(med([p[1] for p in points]),3),'z':round(med([p[2] for p in points]),3)} for points in piece_xyz],
         'cluster_radius_sweep':config,'pair_frames':frame_points}
 output['results'].append(result)
 print(clip['scene'],result['valid_pair_frames'],result['pair_distance_m'],[(x['radius_m'],x['merged_fraction']) for x in config if x['merged_fraction'] in (1.0,0.99)])
path=ROOT/'media/frc-fuel-dense-cluster-radius-sweep.json';path.write_text(json.dumps(output,indent=2)+'\n');print('saved',path)
