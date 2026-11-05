#!/usr/bin/env python3
"""Evaluate low-confidence detections with same-frame deduplication and persistence."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
data=json.load(open(ROOT/'media/frc-fuel-dense-count-evaluation.json'))
TARGETS=((808,589),(902,591))
def iou(a,b):
 x1=max(a['x1'],b['x1']);y1=max(a['y1'],b['y1']);x2=min(a['x2'],b['x2']);y2=min(a['y2'],b['y2'])
 inter=max(0,x2-x1)*max(0,y2-y1);aa=(a['x2']-a['x1'])*(a['y2']-a['y1']);bb=(b['x2']-b['x1'])*(b['y2']-b['y1'])
 return inter/(aa+bb-inter) if aa+bb>inter else 0
def nms(boxes,threshold):
 kept=[]
 for b in sorted(boxes,key=lambda x:x['confidence'],reverse=True):
  if all(iou(b,k)<=threshold for k in kept):kept.append(b)
 return kept
out={'method':'Offline simulation on saved 60 fps detector candidates. Apply confidence cutoff, same-frame IoU NMS, then greedy IoU association over the two prior frames; emit tracks after 2 detections. Ground truth is two fixed target centers within 60 px. Stationary-scene only; motion/latency are not validated.','results':[]}
for clip in data['results']:
 cr={'scene':clip['scene'],'configs':[]}
 for conf in (.05,.075,.10):
  for nms_iou in (.5,.9):
   tracks=[];rows=[];fp=fn=tp=exact=0;latency_frames=[]
   for idx,row in enumerate(clip['per_frame']):
    boxes=nms([b for b in row['detections'] if b['confidence']>=conf],nms_iou)
    active=[t for t in tracks if idx-t['last_frame']<=2]
    used=set();assign={}
    candidates=[]
    for bi,b in enumerate(boxes):
     opts=[(iou(b,t['last_box']),ti) for ti,t in enumerate(active) if ti not in used]
     best=max(opts,default=(0,-1))
     if best[0]>=.5:
      used.add(best[1]);assign[bi]=active[best[1]]
     else:
      t={'id':len(tracks),'hits':0,'last_frame':idx,'last_box':b,'first_frame':idx};tracks.append(t);assign[bi]=t
     t=assign[bi];t['hits']+=1;t['last_frame']=idx;t['last_box']=b
     if t['hits']>=2:candidates.append(b)
    hits=[None,None];extras=0
    for b in sorted(candidates,key=lambda x:x['confidence'],reverse=True):
     cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2;dist=[math.hypot(cx-x,cy-y) for x,y in TARGETS];ti=min(range(2),key=lambda j:dist[j])
     if dist[ti]<=60 and hits[ti] is None:hits[ti]=b
     else:extras+=1
    count=sum(x is not None for x in hits);tp+=count;fn+=2-count;fp+=extras;exact+=count==2 and extras==0
    rows.append({'frame':idx,'target_count':count,'false_positives':extras,'tracks_emitted':len(candidates)})
   # Ignore short startup where targets are entering; report all-frame and steady-state exact counts separately.
   steady=[r for r in rows if r['frame']>=30]
   config={'confidence':conf,'nms_iou':nms_iou,'track_iou':.5,'confirm_hits':2,'max_gap_frames':2,'tp':tp,'fn':fn,'fp':fp,'exact_two_all_frames':exact,'steady_frames':len(steady),'exact_two_steady_frames':sum(r['target_count']==2 and r['false_positives']==0 for r in steady),'steady_fp':sum(r['false_positives'] for r in steady),'steady_fn':sum(2-r['target_count'] for r in steady)}
   cr['configs'].append(config);print(cr['scene'],config)
 out['results'].append(cr)
path=ROOT/'media/frc-fuel-low-confidence-tracking-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n');print('saved',path)
