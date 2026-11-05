#!/usr/bin/env python3
"""Compare stereo, apparent-size, and fused FUEL range estimates on dense clips."""
import json,math,statistics
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
dets=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
stereo=json.load(open(ROOT/'media/frc-fuel-dense-stereo-evaluation.json'))
cal=ROOT/'media/calibration/stereo_calibration_live_order_candidate.yml';fs=cv2.FileStorage(str(cal),cv2.FILE_STORAGE_READ)
k=fs.getNode('left_camera_matrix').mat();d=fs.getNode('left_distortion').mat();kr=fs.getNode('right_camera_matrix').mat();dr=fs.getNode('right_distortion').mat();rot=fs.getNode('rotation').mat();trans=fs.getNode('translation').mat();fs.release()
r1,r2,p1,p2,*_=cv2.stereoRectify(k,d,kr,dr,(1600,1200),rot,trans,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
focal=float(k[0,0]);diameter=.15;targets=((808,589),(902,591));weights=(0,.25,.5,.75,1)
def median(v):return statistics.median(v) if v else None
def stats(v):
 if not v:return {'n':0}
 m=median(v);return {'n':len(v),'median_m':round(m,4),'mad_m':round(median([abs(x-m) for x in v]),4),'p05_m':round(sorted(v)[int(.05*(len(v)-1))],4),'p95_m':round(sorted(v)[int(.95*(len(v)-1))],4)}
def matched(row):
 out=[None,None]
 for b in sorted((x for x in row['detections'] if x['confidence']>=.10),key=lambda x:x['confidence'],reverse=True):
  cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
  i=min(range(2),key=lambda j:math.hypot(cx-targets[j][0],cy-targets[j][1]))
  if math.hypot(cx-targets[i][0],cy-targets[i][1])<=60 and out[i] is None:out[i]=b
 return out
out={'method':'Size range = raw focal length * nominal 15 cm FUEL diameter / geometric-mean detector-box size. Fused range = alpha*stereo + (1-alpha)*size. Stereo uses active 63.5 mm baseline, full dense b5/u8 output. This compares repeatability and agreement only; neither range source is measured ground truth.','weights_stereo':list(weights),'results':[]}
for di,clip in enumerate(dets['results']):
 sr=stereo['results'][di];rows=[]
 for boxrow,srow in zip(clip['per_frame'],sr['frames']):
  boxes=matched(boxrow);frame={'size':[None,None],'stereo':srow['ranges_m'],'uv':[None,None]}
  for ti,b in enumerate(boxes):
   if b is None:continue
   w=b['x2']-b['x1'];h=b['y2']-b['y1']
   if w<=0 or h<=0:continue
   frame['size'][ti]=focal*diameter/math.sqrt(w*h)
   c=np.array([[[ (b['x1']+b['x2'])/2, (b['y1']+b['y2'])/2 ]]],np.float32)
   frame['uv'][ti]=cv2.undistortPoints(c,k,d,R=r1,P=p1)[0,0].tolist()
  rows.append(frame)
 clipout={'scene':clip['scene'],'weights':[]}
 for a in weights:
  ranges=[[],[]];spacing=[];diffs=[]
  for f in rows:
   zs=[]
   for ti in range(2):
    s=f['stereo'][ti];zsize=f['size'][ti]
    z=(a*s+(1-a)*zsize) if s is not None and zsize is not None else None
    zs.append(z)
    if z is not None:ranges[ti].append(z);diffs.append(abs(s-zsize))
   if all(z is not None for z in zs) and all(u is not None for u in f['uv']):
    pts=[]
    for u,v,z in ( (f['uv'][i][0],f['uv'][i][1],zs[i]) for i in range(2)):
     pts.append(((u-p1[0,2])*z/p1[0,0],(v-p1[1,2])*z/p1[1,1],z))
    spacing.append(math.dist(*pts))
  clipout['weights'].append({'stereo_weight':a,'range_by_target_m':[stats(x) for x in ranges],'pair_spacing_m':stats(spacing),'median_size_stereo_abs_difference_m':round(median(diffs),4) if diffs else None})
 out['results'].append(clipout)
 print(clip['scene'])
 for x in clipout['weights']:print(x)
# Compare how close each estimated target range median is between the two saved clips.
for a in weights:
 entry={'stereo_weight':a,'cross_clip_target_median_gaps_m':[]}
 for ti in range(2):
  medians=[]
  for clip in out['results']:
   medians.append(clip['weights'][weights.index(a)]['range_by_target_m'][ti].get('median_m'))
  entry['cross_clip_target_median_gaps_m'].append(round(abs(medians[0]-medians[1]),4))
 out.setdefault('cross_clip',[]).append(entry)
print('cross clip',out['cross_clip'])
path=ROOT/'media/frc-fuel-range-fusion-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
