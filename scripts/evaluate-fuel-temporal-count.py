#!/usr/bin/env python3
"""Evaluate causal per-target count confirmation at Scout replay cadence."""
import json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'media/frc-fuel-low-confidence-evaluation.json'
TARGETS=((808,589),(902,591)); CONF=.05; STRIDE=4; WINDOWS=(3,5,7); GRACE=(1,2); FPS=59.900166

def med(x):return statistics.median(x) if x else None
source=json.load(open(SRC));output={'method':'Rescore both saved 600-frame videos at confidence 0.05. Subsample at stride 4 (~15 Hz) across all four phase offsets. Compare causal majority confirmation over 3/5/7 observations with 1/2-observation dropout grace. Count-only smoothing; position age is measured from the most recent actual detection.','limitations':'The clips contain two stationary visible targets, so they cannot measure false retention after a ball leaves, crosses, or becomes fully occluded. Majority/grace filtering adds persistence and can hide a real count change. The samples simulate the active ~15 fps inference cadence, not every exact runtime scheduling interval.','confidence':CONF,'source_fps':FPS,'sample_stride':STRIDE,'windows':WINDOWS,'dropout_grace_samples':GRACE,'results':[]}
for clip in source['results']:
 frames=clip['per_frame']; clipout={'scene':clip['scene'],'video':clip['video'],'phases':[]}
 for phase in range(STRIDE):
  ids=list(range(phase,len(frames),STRIDE));observations=[]
  for idx in ids:
   chosen=[False,False]
   for b in sorted((b for b in frames[idx] if b['confidence']>=CONF),key=lambda b:b['confidence'],reverse=True):
    cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
    d,i=min((math.hypot(cx-x,cy-y),i) for i,(x,y) in enumerate(TARGETS))
    if d<=60 and not chosen[i]:chosen[i]=True
   observations.append(chosen)
  raw_pair=sum(all(x) for x in observations)
  methods={}
  for window in WINDOWS:
   confirmed=[[],[]];ages=[[],[]]
   for i,event in enumerate(observations):
    hist=observations[max(0,i-window+1):i+1];need=len(hist)//2+1
    for t in range(2):
     active=sum(row[t] for row in hist)>=need;confirmed[t].append(active)
     if active:
      last=max(j for j,row in enumerate(hist) if row[t]);ages[t].append((len(hist)-1-last)*STRIDE/FPS)
   pair=sum(confirmed[0][i] and confirmed[1][i] for i in range(len(ids)))
   methods[str(window)]={'confirmed_pair_outputs':pair,'raw_pair_outputs':raw_pair,'sampled_outputs':len(ids),'additional_pair_outputs':pair-raw_pair,'per_target_max_stale_age_s':[round(max(a),3) if a else None for a in ages],'per_target_median_stale_age_s':[round(med(a),3) if a else None for a in ages]}
  for grace in GRACE:
   confirmed=[[],[]];ages=[[],[]];window=grace+1
   for i,event in enumerate(observations):
    hist=observations[max(0,i-window+1):i+1]
    for t in range(2):
     active=any(row[t] for row in hist);confirmed[t].append(active)
     if active:
      last=max(j for j,row in enumerate(hist) if row[t]);ages[t].append((len(hist)-1-last)*STRIDE/FPS)
   pair=sum(confirmed[0][i] and confirmed[1][i] for i in range(len(ids)))
   methods[f'dropout_grace_{grace}']={'confirmed_pair_outputs':pair,'raw_pair_outputs':raw_pair,'sampled_outputs':len(ids),'additional_pair_outputs':pair-raw_pair,'per_target_max_stale_age_s':[round(max(a),3) if a else None for a in ages],'per_target_median_stale_age_s':[round(med(a),3) if a else None for a in ages]}
  clipout['phases'].append({'phase':phase,'sampled_outputs':len(ids),'raw_pair_outputs':raw_pair,'methods':methods})
 clipout['phase_average']={}
 for w in WINDOWS:
  clipout['phase_average'][str(w)]={'raw_pair_outputs':round(statistics.mean(x['raw_pair_outputs'] for x in clipout['phases']),2),'confirmed_pair_outputs':round(statistics.mean(x['methods'][str(w)]['confirmed_pair_outputs'] for x in clipout['phases']),2),'max_stale_age_s_by_target':[max(x['methods'][str(w)]['per_target_max_stale_age_s'][t] or 0 for x in clipout['phases']) for t in range(2)]}
 for grace in GRACE:
  key=f'dropout_grace_{grace}'
  clipout['phase_average'][key]={'raw_pair_outputs':round(statistics.mean(x['raw_pair_outputs'] for x in clipout['phases']),2),'confirmed_pair_outputs':round(statistics.mean(x['methods'][key]['confirmed_pair_outputs'] for x in clipout['phases']),2),'max_stale_age_s_by_target':[max(x['methods'][key]['per_target_max_stale_age_s'][t] or 0 for x in clipout['phases']) for t in range(2)]}
 output['results'].append(clipout)
 print(clip['scene'],clipout['phase_average'])
p=ROOT/'media/frc-fuel-temporal-count-sweep.json';p.write_text(json.dumps(output,indent=2)+'\n');print('saved',p)
