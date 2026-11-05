#!/usr/bin/env python3
"""Compare stricter size-depth gates with paired masks on saved FUEL videos."""
import os,json,math,time,statistics
from pathlib import Path
import cv2
from scout.detectors import TensorRTDetector,Detection
from scout.stereo import StereoDepth
ROOT=Path(__file__).resolve().parents[1]
CAL='media/calibration/stereo_calibration_live_order_candidate.yml'
TARGETS=((808,589),(902,591)); TIMES=tuple(1.0+.4*i for i in range(15));FRAC=(0,.25,.5,.75);TOLERANCES=(.075,.1,.15,.2,.55)
os.environ.update(SCOUT_CLASS_NAMES='["ball"]',SCOUT_MAX_BOX_ASPECT_RATIO='2.0',SCOUT_STEREO_CALIBRATION=CAL,SCOUT_STEREO_RECTIFY='1',SCOUT_STEREO_BASELINE_M='0.0635',SCOUT_STEREO_MATCH_SCALE='.5',SCOUT_STEREO_NUM_DISPARITIES='128',SCOUT_FUEL_DIAMETER_M='.15',SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR='.20',SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU='.90')
def match(boxes):
 chosen=[None,None]
 for b in sorted(boxes,key=lambda b:b.confidence,reverse=True):
  cx=(b.x1+b.x2)/2;cy=(b.y1+b.y2)/2;ds=[math.hypot(cx-x,cy-y) for x,y in TARGETS];i=min(range(2),key=lambda j:ds[j])
  if ds[i]<=60 and chosen[i] is None:chosen[i]=b
 return chosen
def summary(rows):
 return {'frames':len(rows),'exact_two':sum(r['after_count']==2 and r['after_fp']==0 for r in rows),'paired':sum(r['after_count']==2 for r in rows),'tp':sum(r['after_count'] for r in rows),'fp':sum(r['after_fp'] for r in rows),'fn':sum(2-r['after_count'] for r in rows),'raw_exact_two':sum(r['before_count']==2 and r['before_fp']==0 for r in rows),'gate_rejected':sum(r['gate_rejected'] for r in rows),'duplicate_rejected':sum(r['duplicate_rejected'] for r in rows),'median_detector_ms':round(statistics.median(r['detector_ms'] for r in rows),2),'median_stereo_ms':round(statistics.median(r['stereo_ms'] for r in rows),2)}
out={'method':'For stable frames at 1.0-6.6s, mask the lower fraction of each visible target box in both stereo views (right-view mask shifted by the pre-mask measured disparity). Run FRC9550 1280x960 at confidence .05, rectify/SGBM, then sweep size-vs-stereo gates and 0.90 IoU / 7.5cm 3D dedup. Black masks simulate missing target texture, not a real foreground object; judge count only.','confidence':.05,'gate_tolerances':list(TOLERANCES),'fuel_diameter_m':.15,'results':[]}
detector=TensorRTDetector(str(ROOT/'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'),confidence=.05,iou=.15)
stereo=StereoDepth()
try:
 for scene,video in [('earlier','media/frc-2026-live-two-fuel-60fps-cfr.mp4'),('corrected-settings','media/frc-2026-fuel-camera-settings-fixed-60fps.mp4')]:
  cap=cv2.VideoCapture(str(ROOT/video));fps=cap.get(cv2.CAP_PROP_FPS);dense=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))['results'][0 if scene=='earlier' else 1]['per_frame'];rows_by={f:{tol:[] for tol in TOLERANCES} for f in FRAC}
  for t in TIMES:
   idx=round(t*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read()
   if not ok:continue
   left,right=frame[:,:1600].copy(),frame[:,1600:].copy()
   seeds=[Detection(int(b['x1']),int(b['y1']),int(b['x2']),int(b['y2']),0,float(b['confidence'])) for b in dense[idx]['detections'] if b['confidence']>=.1]
   seedmatch=match(seeds);stereo.estimate(left,right,[b for b in seedmatch if b])
   for frac in FRAC:
    lcopy=left.copy();rcopy=right.copy()
    if frac:
     for b in seedmatch:
      if b is None:continue
      height=b.y2-b.y1;width=b.x2-b.x1;cover=max(1,round(height*frac));disp=b.disparity_px or 45.0
      y1=max(0,b.y2-cover);y2=min(1200,b.y2)
      x1=max(0,b.x1);x2=min(1600,b.x2)
      lcopy[y1:y2,x1:x2]=0
      rx1=max(0,round(b.x1-disp));rx2=min(1600,round(b.x2-disp))
      rcopy[y1:y2,rx1:rx2]=0
    detections,det_ms=detector.infer(lcopy)
    raw=list(detections);before=match(raw);stereo_start=time.perf_counter();stereo.estimate(lcopy,rcopy,detections);stereo_ms=(time.perf_counter()-stereo_start)*1000
    before_count=sum(b is not None for b in before);before_fp=max(0,len(raw)-before_count)
    for tolerance in TOLERANCES:
     stereo.size_gate_max_relative_error=tolerance
     filtered=stereo.filter_fuel_size_inconsistent(detections);after=match(filtered);after_count=sum(b is not None for b in after);after_fp=max(0,len(filtered)-after_count)
     rows_by[frac][tolerance].append({'frame':idx,'time_s':t,'before_count':before_count,'before_fp':before_fp,'after_count':after_count,'after_fp':after_fp,'gate_rejected':stereo.last_size_gate_rejected,'duplicate_rejected':stereo.last_size_gate_duplicate_rejected,'detector_ms':det_ms,'stereo_ms':stereo_ms})
    stereo.size_gate_max_relative_error=.20
  cap.release();entry={'scene':scene,'fractions':[]}
  for frac in FRAC:
   configs=[]
   for tolerance in TOLERANCES:
    stats=summary(rows_by[frac][tolerance]);configs.append({'gate_tolerance':tolerance,**stats,'frames_detail':rows_by[frac][tolerance]});print(scene,frac,tolerance,stats)
   entry['fractions'].append({'occlusion_fraction':frac,'gate_configs':configs})
  out['results'].append(entry)
finally:detector.close()
path=ROOT/'media/frc-fuel-size-gate-occlusion-stricter-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
