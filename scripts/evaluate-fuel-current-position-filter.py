#!/usr/bin/env python3
"""Compare causal position smoothing on the current detector/stereo outputs."""
import json, math, statistics
from pathlib import Path
import cv2, numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'media/frc-fuel-dense-count-nms015.json').read_text())
S=json.loads((ROOT/'media/frc-fuel-dense-stereo-nms015-conf005.json').read_text())
TARGETS=((808.,589.),(902.,591.));STRIDE=4;CONF=.05
fs=cv2.FileStorage(str(ROOT/'media/calibration/stereo_calibration_live_order_candidate.yml'),cv2.FILE_STORAGE_READ)
k=fs.getNode('left_camera_matrix').mat();dist=fs.getNode('left_distortion').mat();kr=fs.getNode('right_camera_matrix').mat();dr=fs.getNode('right_distortion').mat();rot=fs.getNode('rotation').mat();trans=fs.getNode('translation').mat();fs.release()
r1,r2,p1,p2,*_=cv2.stereoRectify(k,dist,kr,dr,(1600,1200),rot,trans,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
fx,fy,cx,cy=float(p1[0,0]),float(p1[1,1]),float(p1[0,2]),float(p1[1,2])
def choose(dets):
 out=[None,None]
 for b in sorted((b for b in dets if b['confidence']>=CONF),key=lambda q:-q['confidence']):
  x=(b['x1']+b['x2'])/2;y=(b['y1']+b['y2'])/2;dist,i=min((math.hypot(x-a,y-c),i) for i,(a,c) in enumerate(TARGETS))
  if dist<=60 and out[i] is None:out[i]=b
 return out
def point(box,z):
 x=(box['x1']+box['x2'])/2;y=(box['y1']+box['y2'])/2
 q=cv2.undistortPoints(np.array([[[x,y]]],np.float32),k,dist,R=r1,P=p1)[0,0]
 return np.array(((q[0]-cx)*z/fx,(q[1]-cy)*z/fy,z),dtype=float)
def median_point(points):return np.median(np.stack(points),axis=0)
def summarize(vals,reference):
 if not vals:return None
 errors=[float(np.linalg.norm(v-reference)) for v in vals]
 med=statistics.median(errors)
 return {'n':len(vals),'median_3d_deviation_m':round(med,5),'p95_3d_deviation_m':round(float(np.percentile(errors,95)),5)}
def filter_track(values,kind):
 out=[];hist=[];ema=None
 for v in values:
  if v is None:out.append(None);continue
  a=np.asarray(v,dtype=float)
  if kind=='raw':out.append(a);continue
  if kind=='median3':
   hist.append(a);out.append(median_point(hist[-3:]));continue
  ema=a if ema is None else .5*a+.5*ema;out.append(ema.copy())
 return out
out={'method':'Rescore current NMS 0.15 / confidence 0.05 boxes with saved SGBM b5/u8 ranges. Subsample every fourth frame (~15 Hz) across all four phases. Compare raw XYZ positions with causal last-3-observation coordinate medians and EMA alpha 0.5. Measure 3D deviation from each clip/target median position; smoothing does not create outputs on missing detections.','limitations':'The clips are stationary, so this measures position repeatability only; motion lag is not validated and the clip median is not true position. Stereo range has no external ground truth.','sample_stride_frames':STRIDE,'approx_hz':59.900166/STRIDE,'results':[]}
for ci,clip in enumerate(D['results']):
 depth=S['results'][ci]['frames'];tracks=[[None]*len(clip['per_frame']) for _ in range(2)];centroid=[None]*len(clip['per_frame'])
 for idx,row in enumerate(clip['per_frame']):
  boxes=choose(row['detections']);ranges=depth[idx]['ranges_m']
  for ti in range(2):
   if boxes[ti] is not None and ranges[ti] is not None:tracks[ti][idx]=point(boxes[ti],ranges[ti])
  if all(tracks[ti][idx] is not None for ti in range(2)):centroid[idx]=(tracks[0][idx]+tracks[1][idx])/2
 per_phase=[]
 for phase in range(STRIDE):
  ids=list(range(phase,len(clip['per_frame']),STRIDE));phase_rows=[]
  for name,series in [('cluster_centroid',[centroid[i] for i in ids]),('target_1',[tracks[0][i] for i in ids]),('target_2',[tracks[1][i] for i in ids])]:
   valid=[x for x in series if x is not None]
   if not valid:continue
   reference=median_point(valid)
   row={'track':name,'raw':summarize(valid,reference)}
   for method in ('median3','ema50'):
    filtered=filter_track(series,method);row[method]=summarize([x for x in filtered if x is not None],reference)
   phase_rows.append(row)
  per_phase.append({'phase':phase,'outputs':len(ids),'tracks':phase_rows})
 # Mean the phase-specific error summaries; phase data are paired and avoid one favorable offset.
 aggregate=[]
 for name in ('cluster_centroid','target_1','target_2'):
  rows=[next(x for x in p['tracks'] if x['track']==name) for p in per_phase]
  aggregate.append({'track':name,'methods':{m:{'mean_median_deviation_m':round(statistics.mean(r[m]['median_3d_deviation_m'] for r in rows),5),'mean_p95_deviation_m':round(statistics.mean(r[m]['p95_3d_deviation_m'] for r in rows),5),'mean_valid_outputs':round(statistics.mean(r[m]['n'] for r in rows),1)} for m in ('raw','median3','ema50')}})
 out['results'].append({'scene':clip['scene'],'aggregate':aggregate,'phases':per_phase})
 print(clip['scene'],json.dumps(aggregate))
path=ROOT/'media/frc-fuel-current-position-filter.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
