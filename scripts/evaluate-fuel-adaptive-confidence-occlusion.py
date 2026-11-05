#!/usr/bin/env python3
"""Stress-test candidate-count confidence adaptation under synthetic occlusion."""
import json
import math
import os
import statistics
import sys
from pathlib import Path

import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=json.loads((ROOT/'media/frc-fuel-dense-count-nms010.json').read_text())
TARGETS=((808.0,589.0),(902.0,591.0))
TIMES=tuple(1.0+.4*i for i in range(15))
FRACTIONS=(0.0,.25,.50,.75)
BASE=.05
TRIGGER=3
HIGH=.20
os.environ['SCOUT_CLASS_NAMES']='["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO']='2.0'
from scout.detectors import TensorRTDetector
ENGINE=ROOT/'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'


def score(boxes, threshold):
    hits=[None,None];extras=0
    for b in sorted((x for x in boxes if x['confidence']>=threshold),key=lambda x:-x['confidence']):
        cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
        distance,i=min((math.hypot(cx-x,cy-y),i) for i,(x,y) in enumerate(TARGETS))
        if distance<=60 and hits[i] is None:hits[i]=b
        else:extras+=1
    return sum(x is not None for x in hits),extras,hits


def summarize(rows):
    tp=sum(x['tp'] for x in rows);fp=sum(x['fp'] for x in rows);fn=sum(x['fn'] for x in rows)
    exact=sum(x['hits']==2 and x['fp']==0 for x in rows)
    return {'frames':len(rows),'exact_two':exact,'tp':tp,'fp':fp,'fn':fn,
            'precision':round(tp/(tp+fp),5) if tp+fp else None,
            'recall':round(tp/(tp+fn),5) if tp+fn else None}


detector=TensorRTDetector(str(ENGINE),confidence=BASE,iou=.10)
output={'method':('On 15 timestamps in each saved stereo clip, black-mask the lower 0/25/50/75% of each '
                  'known ball box in the left view. Compare fixed confidence 0.05 with the adaptive rule '
                  'that raises the cutoff to 0.20 when at least three post-NMS boxes remain. NMS IoU is 0.10.'),
        'settings':{'base_confidence':BASE,'trigger_candidate_count':TRIGGER,
                    'high_confidence':HIGH,'nms_iou':.10,'times_s':TIMES,
                    'mask_fractions':FRACTIONS,'mask_orientation':'lower portion of each target box'},
        'limitations':'Synthetic black masks on stationary targets; masks do not reproduce physical occluder parallax, lighting, or moving balls.',
        'results':[]}
try:
 for clip in DATA['results']:
  capture=cv2.VideoCapture(str(ROOT/clip['video']));fps=capture.get(cv2.CAP_PROP_FPS)
  scene={'scene':clip['scene'],'video':clip['video'],'fractions':[]}
  for fraction in FRACTIONS:
   fixed=[];adaptive=[];details=[]
   for time_s in TIMES:
    index=round(time_s*fps);capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=capture.read()
    if not ok:continue
    left=frame[:,:1600].copy()
    targets=[None,None]
    for b in sorted((b for b in clip['per_frame'][index]['detections'] if b['confidence']>=.10),key=lambda b:-b['confidence']):
     cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
     distance,i=min((math.hypot(cx-x,cy-y),i) for i,(x,y) in enumerate(TARGETS))
     if distance<=60 and targets[i] is None:targets[i]=b
    for b in targets:
     if b is None or not fraction:continue
     x1,y1,x2,y2=map(int,(b['x1'],b['y1'],b['x2'],b['y2']))
     top=max(0,y2-max(1,round((y2-y1)*fraction)))
     left[top:y2,x1:x2]=0
    boxes,latency=detector.infer(left)
    raw=[{'x1':b.x1,'y1':b.y1,'x2':b.x2,'y2':b.y2,'confidence':float(b.confidence)} for b in boxes]
    adaptive_cutoff=HIGH if len(raw)>=TRIGGER else BASE
    n_fixed,fp_fixed,hits_fixed=score(raw,BASE)
    n_adaptive,fp_adaptive,hits_adaptive=score(raw,adaptive_cutoff)
    fixed.append({'hits':n_fixed,'fp':fp_fixed,'tp':n_fixed,'fn':2-n_fixed})
    adaptive.append({'hits':n_adaptive,'fp':fp_adaptive,'tp':n_adaptive,'fn':2-n_adaptive})
    details.append({'time_s':time_s,'frame':index,'candidates_at_base':len(raw),
                    'adaptive_cutoff':adaptive_cutoff,'fixed_hits':n_fixed,'fixed_fp':fp_fixed,
                    'adaptive_hits':n_adaptive,'adaptive_fp':fp_adaptive,
                    'fixed_target_confidences':[x['confidence'] if x else None for x in hits_fixed],
                    'adaptive_target_confidences':[x['confidence'] if x else None for x in hits_adaptive],
                    'inference_ms':round(latency,2)})
   scene['fractions'].append({'occlusion_fraction':fraction,
      'fixed_0_05':summarize(fixed),'adaptive':summarize(adaptive),
      'adaptive_triggered_frames':sum(x['candidates_at_base']>=TRIGGER for x in details),
      'frames_detail':details})
  capture.release();output['results'].append(scene)
  print(scene['scene'],[(x['occlusion_fraction'],x['fixed_0_05']['exact_two'],x['adaptive']['exact_two'],x['adaptive_triggered_frames']) for x in scene['fractions']],flush=True)
finally:
 detector.close()
path=ROOT/'media/frc-fuel-adaptive-confidence-occlusion-evaluation.json'
path.write_text(json.dumps(output,indent=2)+'\n')
print('saved',path)
