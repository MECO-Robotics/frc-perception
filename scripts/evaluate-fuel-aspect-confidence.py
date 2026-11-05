#!/usr/bin/env python3
"""Sweep confidence and symmetric box-aspect gates on saved FUEL detections."""
import json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'media/frc-fuel-low-confidence-evaluation.json'
CONFIDENCES=(.02,.03,.05,.075,.10)
ASPECT_LIMITS=(1.05,1.10,1.15,1.20,1.30,1.50,2.0)
TARGETS=((808,589),(902,591)); MATCH=60

def evaluate(frames, confidence, limit):
    tp=fp=fn=exact=0
    for row in frames:
        hits=[None,None]; extras=0
        candidates=[]
        for b in row:
            w=b['x2']-b['x1']; h=b['y2']-b['y1']
            aspect=max(w/h,h/w) if w>0 and h>0 else float('inf')
            if b['confidence']>=confidence and aspect<=limit:
                candidates.append(b)
        for b in sorted(candidates,key=lambda x:x['confidence'],reverse=True):
            cx=(b['x1']+b['x2'])/2; cy=(b['y1']+b['y2'])/2
            dist,i=min((math.hypot(cx-x,cy-y),i) for i,(x,y) in enumerate(TARGETS))
            if dist<=MATCH and hits[i] is None: hits[i]=b
            else: extras+=1
        n=sum(x is not None for x in hits); tp+=n; fn+=2-n; fp+=extras
        exact+=n==2 and extras==0
    return {'confidence':confidence,'max_symmetric_aspect':limit,'tp':tp,'fp':fp,'fn':fn,
            'exact_two_frames':exact,'precision':round(tp/(tp+fp),6) if tp+fp else 0,
            'recall':round(tp/(tp+fn),6) if tp+fn else 0}

source=json.loads(INPUT.read_text()); results=[]
for clip in source['results']:
    rows=[evaluate(clip['per_frame'],c,a) for c in CONFIDENCES for a in ASPECT_LIMITS]
    results.append({'scene':clip['scene'],'video':clip['video'],'frames':clip['frames'],'sweep':rows})
out={'method':'Rescore saved 0.005-cutoff TensorRT outputs on both full stereo clips. Sweep confidence threshold and maximum symmetric box aspect ratio max(w/h,h/w); match against two manually labeled centers within 60 px, count all other surviving boxes as false positives. No detector inference or camera capture is performed.','limitations':'The clips contain two stationary targets in one room view. The fixed target windows and aspect threshold do not establish field false-positive performance or moving/occluded-ball recall. Aspect gating can reject elongated balls under truncation or motion blur.','confidence_values':CONFIDENCES,'max_symmetric_aspect_values':ASPECT_LIMITS,'results':results}
path=ROOT/'media/frc-fuel-aspect-confidence-sweep.json';path.write_text(json.dumps(out,indent=2)+'\n')
for r in results:
 print('\n'+r['scene'])
 for c in CONFIDENCES:
  rows=[x for x in r['sweep'] if x['confidence']==c]
  best=max(rows,key=lambda x:(x['exact_two_frames'],-x['fp'],-x['fn']))
  base=next(x for x in rows if x['max_symmetric_aspect']==2.0)
  print(f"conf={c:.3f} baseline exact={base['exact_two_frames']}/600 fp={base['fp']} fn={base['fn']}; best aspect={best['max_symmetric_aspect']:.2f} exact={best['exact_two_frames']}/600 fp={best['fp']} fn={best['fn']}")
print('saved',path)
