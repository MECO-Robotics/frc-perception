#!/usr/bin/env python3
"""Rescore saved detector outputs with tighter same-class NMS IoU thresholds."""
import json,math
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'media/frc-fuel-low-confidence-evaluation.json'
TARGETS=((808,589),(902,591)); MATCH=60.0
IOUS=(.05,.10,.20,.30,.45); CONFIDENCES=(.05,.075)
def iou(a,b):
 x1=max(a['x1'],b['x1']);y1=max(a['y1'],b['y1']);x2=min(a['x2'],b['x2']);y2=min(a['y2'],b['y2'])
 inter=max(0,x2-x1)*max(0,y2-y1);area1=(a['x2']-a['x1'])*(a['y2']-a['y1']);area2=(b['x2']-b['x1'])*(b['y2']-b['y1'])
 return inter/(area1+area2-inter) if area1+area2>inter else 0

def nms(rows,confidence,threshold):
 output=[]
 for row in rows:
  boxes=[b for b in row if b['confidence']>=confidence]
  kept=[]
  for b in sorted(boxes,key=lambda z:z['confidence'],reverse=True):
   if all(iou(b,k)<=threshold for k in kept):kept.append(b)
  output.append(kept)
 return output

def score(rows):
 tp=fp=fn=exact=pair_iou_max=0
 for row in rows:
  hits=[None,None];extras=0
  for b in sorted(row,key=lambda z:z['confidence'],reverse=True):
   cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
   d,i=min((math.hypot(cx-x,cy-y),i) for i,(x,y) in enumerate(TARGETS))
   if d<=MATCH and hits[i] is None:hits[i]=b
   else:extras+=1
  count=sum(x is not None for x in hits);tp+=count;fn+=2-count;fp+=extras;exact+=count==2 and extras==0
  if all(hits):pair_iou_max=max(pair_iou_max,iou(*hits))
 return {'tp':tp,'fp':fp,'fn':fn,'exact_two_frames':exact,'precision':round(tp/(tp+fp),6) if tp+fp else 0,'recall':round(tp/(tp+fn),6) if tp+fn else 0,'max_iou_between_two_target_boxes':round(pair_iou_max,4)}

source=json.load(open(SOURCE));out={'method':'Apply same-class greedy NMS to saved TensorRT outputs, which were originally produced using iou=0.45, at lower IoU thresholds. Score confidence .05/.075 on fixed target centers and count duplicates/off-target boxes as false positives. This can test stricter post-suppression but cannot recover candidates already removed by the original .45 pass.','limitations':'Stationary two-target room clips only. NMS may suppress distinct balls when their boxes overlap more under closer spacing or occlusion; these clips show a maximum target-box IoU but cannot validate that regime. Results only cover tightening from the existing .45 threshold.','original_iou':.45,'iou_thresholds':IOUS,'confidence_values':CONFIDENCES,'results':[]}
for clip in source['results']:
 entry={'scene':clip['scene'],'video':clip['video'],'frames':clip['frames'],'configs':[]}
 for conf in CONFIDENCES:
  for threshold in IOUS:
   metrics=score(nms(clip['per_frame'],conf,threshold));entry['configs'].append({'confidence':conf,'iou_threshold':threshold,**metrics})
 out['results'].append(entry)
 for conf in CONFIDENCES:
  print(clip['scene'],'conf',conf,[(r['iou_threshold'],r['exact_two_frames'],r['tp'],r['fp'],r['fn'],r['max_iou_between_two_target_boxes']) for r in entry['configs'] if r['confidence']==conf])
p=ROOT/'media/frc-fuel-nms-iou-sweep.json';p.write_text(json.dumps(out,indent=2)+'\n');print('saved',p)
