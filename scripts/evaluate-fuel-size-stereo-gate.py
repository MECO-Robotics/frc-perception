#!/usr/bin/env python3
"""Test a physical-size/stereo consistency gate on all low-score FUEL candidates."""
import os,json,math
from pathlib import Path
import cv2
from scout.detectors import Detection
from scout.stereo import StereoDepth
ROOT=Path(__file__).resolve().parents[1]
D=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
S=json.load(open(ROOT/'media/frc-fuel-dense-stereo-evaluation.json'))
TARGETS=((808,589),(902,591)); FOCAL=1046.1599129444185; DIAM=.15
os.environ.update(SCOUT_STEREO_CALIBRATION='media/calibration/stereo_calibration_live_order_candidate.yml',SCOUT_STEREO_RECTIFY='1',SCOUT_STEREO_BASELINE_M='0.0635',SCOUT_STEREO_MATCH_SCALE='0.5',SCOUT_STEREO_NUM_DISPARITIES='128',SCOUT_CLASS_NAMES='["ball"]')
stereo=StereoDepth()
def iou(a,b):
 x1=max(a['x1'],b['x1']);y1=max(a['y1'],b['y1']);x2=min(a['x2'],b['x2']);y2=min(a['y2'],b['y2'])
 inter=max(0,x2-x1)*max(0,y2-y1);aa=(a['x2']-a['x1'])*(a['y2']-a['y1']);bb=(b['x2']-b['x1'])*(b['y2']-b['y1'])
 return inter/(aa+bb-inter) if aa+bb>inter else 0
def nms(boxes,t):
 keep=[]
 for b in sorted(boxes,key=lambda b:b['confidence'],reverse=True):
  if all(iou(b,k)<=t for k in keep):keep.append(b)
 return keep
def target_idx(b):
 cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2;ds=[math.hypot(cx-x,cy-y) for x,y in TARGETS];i=min(range(2),key=lambda j:ds[j])
 return i if ds[i]<=60 else None
out={'method':'Detector candidates at confidence >=0.05. Require valid rectified SGBM depth and relative agreement between stereo Z and nominal 15 cm FUEL apparent-size Z; then same-frame IoU NMS. Stereo is recomputed for frames containing low-score candidates and otherwise reused from the dense stereo artifact. Fixed-center labeling covers only two stationary balls.','minimum_confidence':.05,'fuel_diameter_m':DIAM,'focal_px':FOCAL,'results':[]}
for ci,clip in enumerate(D['results']):
 cap=cv2.VideoCapture(str(ROOT/clip['video'])); rows=clip['per_frame']; srows=S['results'][ci]['frames']; per_tol=[]
 # Store valid ranges for each detection only when the frame has a low-score candidate.
 low_frames={i for i,r in enumerate(rows) if any(.05<=b['confidence']<.1 for b in r['detections'])}
 for tol in (.10,.15,.20,.25,.40,.50,.55,.60,.65,1.0):
  tp=fp=fn=exact=0;valid_target=0;errors=[];candidate_extras=[]
  for i,(r,sr) in enumerate(zip(rows,srows)):
   ds=[dict(b) for b in r['detections'] if b['confidence']>=.05]
   z_by_box={}
   if i in low_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read()
    if not ok:continue
    objs=[Detection(*(int(b[k]) for k in ('x1','y1','x2','y2')),0,float(b['confidence'])) for b in ds]
    stereo.estimate(frame[:,:1600],frame[:,1600:],objs)
    for b,o in zip(ds,objs):z_by_box[id(b)]=o.distance_m
   # Reuse target stereo outputs for all regular-confidence target detections.
   chosen=[None,None]
   for b in ds:
    ti=target_idx(b)
    if ti is not None and (chosen[ti] is None or b['confidence']>chosen[ti]['confidence']):chosen[ti]=b
   for b in ds:
    ti=target_idx(b)
    z=z_by_box.get(id(b))
    if z is None and ti is not None:z=sr['ranges_m'][ti]
    w=b['x2']-b['x1'];h=b['y2']-b['y1']
    if w<=0 or h<=0:continue
    zsize=FOCAL*DIAM/math.sqrt(w*h)
    if z is None:
     if i in low_frames and b['confidence']<.1 and ti is None:
      candidate_extras.append({'frame':i,'confidence':round(b['confidence'],3),'stereo_valid':False,'size_range_m':round(zsize,3)})
     continue
    err=abs(zsize-z)/z;errors.append(err)
    if i in low_frames and b['confidence']<.1 and ti is None:
     candidate_extras.append({'frame':i,'confidence':round(b['confidence'],3),'range_m':round(z,3),'size_range_m':round(zsize,3),'relative_error':round(err,3)})
    if err<=tol:b['_range']=z;b['_size_error']=err
   filtered=[b for b in ds if b.get('_range') is not None]
   filtered=nms(filtered,.9)
   hits=[None,None];extras=[]
   for b in filtered:
    ti=target_idx(b)
    if ti is not None and hits[ti] is None:hits[ti]=b
    else:extras.append(b)
   count=sum(x is not None for x in hits);tp+=count;fn+=2-count;fp+=len(extras);exact+=count==2 and not extras;valid_target+=count
  result={'tolerance_fraction':tol,'tp':tp,'fn':fn,'fp':fp,'exact_two_frames':exact,'valid_target_detections':valid_target,'size_stereo_error_p95':round(sorted(errors)[int(.95*(len(errors)-1))],3) if errors else None,'candidate_extra_diagnostics':candidate_extras if tol==.2 else []}
  per_tol.append(result);print(clip['scene'],result)
 cap.release();out['results'].append({'scene':clip['scene'],'low_candidate_frames':sorted(low_frames),'sweep':per_tol})
path=ROOT/'media/frc-fuel-size-stereo-gate-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
