#!/usr/bin/env python3
"""Compare the mechanical and calibration-derived stereo baselines on saved data."""
import json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
S=json.load(open(ROOT/'media/frc-fuel-dense-stereo-evaluation.json'))
CAL='media/calibration/stereo_calibration_live_order_candidate.yml'
import cv2,numpy as np
fs=cv2.FileStorage(str(ROOT/CAL),cv2.FILE_STORAGE_READ);t=fs.getNode('translation').mat().reshape(-1);k=fs.getNode('left_camera_matrix').mat();fs.release()
BASELINES={'mechanical_override':.0635,'calibration_translation_norm_m':float(np.linalg.norm(t))}
FOCAL=float(k[0,0]);DIAM=.15;TARGETS=((808,589),(902,591))
def med(a):return statistics.median(a) if a else None
def mad(a):
 m=med(a);return med([abs(x-m) for x in a]) if a else None
def stats(a):return {'n':len(a),'median_m':round(med(a),4),'mad_m':round(mad(a),4),'p05_m':round(sorted(a)[int(.05*(len(a)-1))],4),'p95_m':round(sorted(a)[int(.95*(len(a)-1))],4)} if a else None
out={'method':'Reuse saved detector boxes and active SGBM ranges, which were computed with the mechanical 0.0635 m baseline. Rescale all stereo XYZ/ranges linearly to the norm of the stereo calibration translation (or retain 0.0635 m), and compare ranges with the nominal-diameter box-size proxy on both clips.','limitations':'The 0.15 m apparent-size estimate depends on detector box boundaries and is not independent metric ground truth. Baseline change scales every reconstructed coordinate uniformly; this comparison cannot choose an accurate baseline without measured range.','calibration':CAL,'baseline_candidates_m':BASELINES,'focal_raw_px':FOCAL,'fuel_diameter_m':DIAM,'results':[]}
for ci,clip in enumerate(D['results']):
 depth=S['results'][ci]['frames'];per_target={'box_size':[[],[]],'stereo':{name:[[],[]] for name in BASELINES},'abs_diff':{name:[[],[]] for name in BASELINES}}
 for detections,drow in zip(clip['per_frame'],depth):
  chosen=[None,None]
  for b in sorted((x for x in detections['detections'] if x['confidence']>=.05),key=lambda x:x['confidence'],reverse=True):
   cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2;dist,i=min((math.hypot(cx-x,cy-y),i) for i,(x,y) in enumerate(TARGETS))
   if dist<=60 and chosen[i] is None:chosen[i]=b
  for i,b in enumerate(chosen):
   if b is None:continue
   size_z=FOCAL*DIAM/math.sqrt((b['x2']-b['x1'])*(b['y2']-b['y1']));per_target['box_size'][i].append(size_z)
   active=drow['ranges_m'][i]
   for name,base in BASELINES.items():
    if active is None:continue
    z=active*(base/.0635);per_target['stereo'][name][i].append(z);per_target['abs_diff'][name][i].append(abs(z-size_z))
 entry={'scene':clip['scene'],'targets':[]}
 for i in range(2):
  entry['targets'].append({'target':i+1,'box_size_proxy':stats(per_target['box_size'][i]),'baselines':{name:{'stereo_range':stats(per_target['stereo'][name][i]),'per_frame_abs_difference_to_size_proxy_m':stats(per_target['abs_diff'][name][i])} for name in BASELINES}})
 out['results'].append(entry)
 print(clip['scene'])
 for row in entry['targets']:print(row)
p=ROOT/'media/frc-fuel-baseline-sensitivity.json';p.write_text(json.dumps(out,indent=2)+'\n');print('saved',p)
