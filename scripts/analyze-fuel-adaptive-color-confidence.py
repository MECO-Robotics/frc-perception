#!/usr/bin/env python3
"""Analyze a count-triggered confidence cutoff that preserves yellow weak boxes."""
import json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'media/frc-fuel-color-purity-gate-evaluation.json').read_text())
PUB=DATA['cached_public_box_features'];VID=DATA['cached_video_box_features']
CUTOFFS=(.01,.025,.05,.075,.10,.15,.20,.30)
TRIGGER=3;HIGH=.20;BASE=.05;IOU=.5;TARGETS=((808.,589.),(902.,591.))

def iou(a,b):
 x=max(0,min(a['x2'],b['x2'])-max(a['x1'],b['x1']));y=max(0,min(a['y2'],b['y2'])-max(a['y1'],b['y1']));z=x*y;aa=max(0,a['x2']-a['x1'])*max(0,a['y2']-a['y1']);bb=max(0,b['x2']-b['x1'])*max(0,b['y2']-b['y1']);return z/(aa+bb-z) if aa+bb>z else 0
def match(p,g):
 pairs=sorted(((iou(a,b),i,j) for i,a in enumerate(p) for j,b in enumerate(g)),reverse=True);up=set();ug=set()
 for q,i,j in pairs:
  if q<IOU:break
  if i not in up and j not in ug:up.add(i);ug.add(j)
 return len(up),len(p)-len(up),len(g)-len(ug)
def filter_boxes(boxes, purity_key, cutoff):
 candidates=[b for b in boxes if b['confidence']>=BASE]
 if len(candidates)<TRIGGER:return candidates
 return [b for b in candidates if b['confidence']>=HIGH or b['_purity'][purity_key]>=cutoff]
def pub_score(key,cutoff):
 tp=fp=fn=exact=0;errs=[];rows=[]
 for r in PUB:
  p=filter_boxes(r['boxes'],key,cutoff);a,b,c=match(p,r['labels']);tp+=a;fp+=b;fn+=c;e=abs(len(p)-len(r['labels']));errs.append(e);exact+=e==0;rows.append({'image_id':r['image_id'],'gt':len(r['labels']),'pred':len(p),'tp':a,'fp':b,'fn':c,'error':e})
 return {'images':len(PUB),'tp':tp,'fp':fp,'fn':fn,'precision':round(tp/(tp+fp),5) if tp+fp else None,'recall':round(tp/(tp+fn),5) if tp+fn else None,'exact_count_images':exact,'count_mae':round(statistics.mean(errs),5),'per_image':rows}
def video_score(clip,key,cutoff):
 pairs=exact=tp=fp=fn=0;rows=[]
 for r in clip['frames']:
  boxes=filter_boxes(r['boxes'],key,cutoff);hits=[False,False];extras=0
  for b in sorted(boxes,key=lambda b:-b['confidence']):
   cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2;d,i=min((math.hypot(cx-x,cy-y),i) for i,(x,y) in enumerate(TARGETS))
   if d<=60 and not hits[i]:hits[i]=True
   else:extras+=1
  n=sum(hits);tp+=n;fn+=2-n;fp+=extras;pairs+=n==2;exact+=n==2 and extras==0;rows.append({'frame':r['frame'],'targets':n,'extras':extras})
 return {'scene':clip['scene'],'frames':len(clip['frames']),'paired_frames':pairs,'exact_two_frames':exact,'tp':tp,'fp':fp,'fn':fn,'per_frame':rows}

results=[]
for color in DATA['color_configs']:
 for cutoff in CUTOFFS:
  results.append({'color':color,'purity_cutoff':cutoff,'trigger_candidates':TRIGGER,'high_confidence':HIGH,'public':pub_score(color['name'],cutoff),'videos':[video_score(v,color['name'],cutoff) for v in VID]})
results.sort(key=lambda x:(-x['public']['exact_count_images'],x['public']['count_mae'],-x['public']['recall']))
out={'method':'If >=3 boxes exceed active .05 confidence, retain boxes >=.20; for boxes in [.05,.20), retain only those whose central-70% HSV yellow-pixel fraction meets cutoff. Otherwise keep all .05 boxes. Scores cached from `evaluate-fuel-color-purity-gate.py`; no detector rerun.','trigger_candidates':TRIGGER,'base_confidence':BASE,'high_confidence':HIGH,'purity_cutoffs':CUTOFFS,'color_configs':DATA['color_configs'],'fixed_baseline':DATA['baseline'],'global_adaptive_reference':DATA['adaptive_confidence_reference'],'results':results,'limitations':'The color scores are tuned on the same 53 Project516 stills with no split manifest, plus two stationary stereo clips. No test of variable exposure, moving pieces, or absolute depth. This post-hoc result is not production runtime validation.'}
path=ROOT/'media/frc-fuel-adaptive-color-confidence-evaluation.json';path.write_text(json.dumps(out,indent=2)+'\n')
for r in results[:20]:print(r['color']['name'],r['purity_cutoff'],'pub',r['public']['exact_count_images'],r['public']['count_mae'],r['public']['tp'],r['public']['fp'],r['public']['fn'],'vid',[(v['scene'],v['paired_frames'],v['exact_two_frames'],v['fp']) for v in r['videos']])
print('saved',path)
