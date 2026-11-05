#!/usr/bin/env python3
"""Compare full-frame and vertical-band SGBM ranges under synthetic paired occlusion."""
import copy, json, math, os, statistics, time
from pathlib import Path
import cv2
from scout.detectors import Detection
from scout.stereo import StereoDepth

ROOT=Path(__file__).resolve().parents[1]
DATA=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
TIMES=tuple(1.0+.4*i for i in range(15));FRACS=(0,.25,.5,.75);PADS=(0,24,48)
TARGETS=((808,589),(902,591));CAL='media/calibration/stereo_calibration_live_order_candidate.yml'
os.environ.update(SCOUT_STEREO_CALIBRATION=CAL,SCOUT_STEREO_RECTIFY='1',SCOUT_STEREO_BASELINE_M='.0635',SCOUT_STEREO_MATCH_SCALE='.5',SCOUT_STEREO_NUM_DISPARITIES='128')
def match(row):
 out=[None,None]
 for b in sorted((b for b in row['detections'] if b['confidence']>=.1),key=lambda x:x['confidence'],reverse=True):
  cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
  ds=[math.hypot(cx-x,cy-y) for x,y in TARGETS];i=min(range(2),key=lambda j:ds[j])
  if ds[i]<=60 and out[i] is None:out[i]=Detection(int(b['x1']),int(b['y1']),int(b['x2']),int(b['y2']),0,float(b['confidence']))
 return out
def med(vals):return statistics.median(vals) if vals else None
out={'method':'Use saved FRC9550 boxes on 15 stable frames per clip, estimate unmasked stereo, then mask the lower 0/25/50/75% of each ball box in left and disparity-shifted right views. Compare full-frame SGBM and full-width vertical-band SGBM with 24/48 half-resolution pixel context. Boxes are held fixed to isolate stereo behavior; this does not measure detector recall under occlusion.','occlusion_fractions':list(FRACS),'vertical_roi_pads':list(PADS),'results':[]}
stereo=StereoDepth()
for ci,clip in enumerate(DATA['results']):
 cap=cv2.VideoCapture(str(ROOT/clip['video']));fps=cap.get(cv2.CAP_PROP_FPS);records={(f,p):{'ranges':[[],[]],'errors':[[],[]],'valid_pairs':0,'latency_ms':[]} for f in FRACS for p in PADS}
 for t in TIMES:
  idx=round(t*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read()
  if not ok:continue
  left,right=frame[:,:1600].copy(),frame[:,1600:].copy();base_boxes=match(clip['per_frame'][idx])
  selected=[b for b in base_boxes if b]
  if len(selected)!=2:continue
  clean=copy.deepcopy(selected);stereo.vertical_roi_pad=0;stereo.estimate(left,right,clean)
  clean_z=[b.distance_m for b in clean]
  for frac in FRACS:
   lcopy=left.copy();rcopy=right.copy()
   if frac:
    for b in clean:
     cover=max(1,round((b.y2-b.y1)*frac));y1=max(0,b.y2-cover);y2=min(1200,b.y2)
     lcopy[y1:y2,b.x1:b.x2]=0
     disp=b.disparity_px or 45.0;rx1=max(0,round(b.x1-disp));rx2=min(1600,round(b.x2-disp));rcopy[y1:y2,rx1:rx2]=0
   for pad in PADS:
    items=copy.deepcopy(selected);stereo.vertical_roi_pad=pad
    t0=time.perf_counter();stereo.estimate(lcopy,rcopy,items);records[(frac,pad)]['latency_ms'].append((time.perf_counter()-t0)*1000)
    z=[b.distance_m for b in items];rec=records[(frac,pad)]
    for i,value in enumerate(z):
     if value is not None:
      rec['ranges'][i].append(value)
      if clean_z[i] is not None:rec['errors'][i].append(abs(value-clean_z[i]))
    if all(value is not None for value in z):rec['valid_pairs']+=1
 cap.release();entry={'scene':clip['scene'],'sample_frames':len(TIMES),'configs':[]}
 for frac in FRACS:
  for pad in PADS:
   rec=records[(frac,pad)];lat=rec['latency_ms']
   line={'occlusion_fraction':frac,'vertical_roi_pad_half_px':pad,'valid_ranges':[len(a) for a in rec['ranges']],'valid_pair_frames':rec['valid_pairs'],'range_median':[round(med(a),4) if a else None for a in rec['ranges']],'abs_error_vs_unmasked_median_m':[round(med(a),4) if a else None for a in rec['errors']],'matcher_ms_median':round(med(lat),2) if lat else None}
   entry['configs'].append(line);print(clip['scene'],line)
 out['results'].append(entry)
path=ROOT/'media/frc-fuel-stereo-roi-occlusion-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
