#!/usr/bin/env python3
"""Score HSV yellow-purity gating for weak YOLO boxes on public images and stereo clips."""
import json,math,statistics,time,urllib.request
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
PUB=json.loads((ROOT/'media/frc-fuel-tiled-detection-evaluation.json').read_text())['detections']
VID=json.loads((ROOT/'media/frc-fuel-dense-count-nms015.json').read_text())['results']
REPO='https://huggingface.co/datasets/project516/rebuilt-fuel-dataset/resolve/main'
BASE=.05; MAX_ASPECT=2.0; TARGETS=((808.,589.),(902.,591.)); MATCH_PX=60
COLOR_CONFIGS=[{'name':n,'hue':h,'sat':s,'val':v} for n,h,s,v in [
 ('yellow_wide',(10,40),70,40),('yellow_mid',(15,35),70,40),('yellow_strict',(10,40),100,80)]]
GATES=(.20,.25,.30,.40,.50,.60); PURITY=(.05,.10,.20)

def req(url):return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Scout-FRC-evaluation/1.0'}),timeout=90).read()
def iou(a,b):
 x1=max(a['x1'],b['x1']);y1=max(a['y1'],b['y1']);x2=min(a['x2'],b['x2']);y2=min(a['y2'],b['y2']); inter=max(0,x2-x1)*max(0,y2-y1)
 aa=max(0,a['x2']-a['x1'])*max(0,a['y2']-a['y1']);bb=max(0,b['x2']-b['x1'])*max(0,b['y2']-b['y1']);u=aa+bb-inter
 return inter/u if u else 0
def match(pred,gt):
 pairs=sorted(((iou(a,b),i,j) for i,a in enumerate(pred) for j,b in enumerate(gt)),reverse=True);up=set();ug=set()
 for q,i,j in pairs:
  if q<.5:break
  if i not in up and j not in ug:up.add(i);ug.add(j)
 return len(up),len(pred)-len(up),len(gt)-len(ug)
def aspect_ok(b):
 w=max(1,b['x2']-b['x1']);h=max(1,b['y2']-b['y1']);return max(w/h,h/w)<=MAX_ASPECT
def purity(frame,b,cfg):
 h,w=frame.shape[:2];bw=max(1,b['x2']-b['x1']);bh=max(1,b['y2']-b['y1'])
 # Central 70% suppresses floor/background pixels on loose detector boxes.
 padx=round(.15*bw);pady=round(.15*bh);x0=max(0,int(b['x1'])+padx);x1=min(w,int(b['x2'])-padx);y0=max(0,int(b['y1'])+pady);y1=min(h,int(b['y2'])-pady)
 if x1<=x0 or y1<=y0:return 0.
 roi=frame[y0:y1,x0:x1];hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
 m=cv2.inRange(hsv,(cfg['hue'][0],cfg['sat'],cfg['val']),(cfg['hue'][1],255,255))
 return float(np.count_nonzero(m)/m.size)
def select(boxes,cfg=None,gate=None,purity_cut=None):
 out=[]
 for b in boxes:
  if not aspect_ok(b):continue
  if b['confidence']<BASE:continue
  if cfg is not None and b['confidence']<gate and b['_purity'][cfg['name']]<purity_cut:continue
  out.append(b)
 return out
def public_score(rows,selector):
 tp=fp=fn=exact=0;errs=[];per=[]
 for r in rows:
  p=selector(r);g=r['labels'];a,b,c=match(p,g);tp+=a;fp+=b;fn+=c;e=abs(len(p)-len(g));errs.append(e);exact+=e==0
  per.append({'image_id':r['image_id'],'gt':len(g),'pred':len(p),'tp':a,'fp':b,'fn':c,'count_error':e})
 return {'images':len(rows),'tp':tp,'fp':fp,'fn':fn,'precision':round(tp/(tp+fp),5) if tp+fp else None,'recall':round(tp/(tp+fn),5) if tp+fn else None,'exact_count_images':exact,'count_mae':round(statistics.mean(errs),5),'per_image':per}
def video_score(scene,frames,selector):
 pairs=exact=tp=fp=fn=0; per=[]
 for row in frames:
  boxes=selector(row);hits=[None,None];extra=0
  for b in sorted(boxes,key=lambda b:-b['confidence']):
   cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2;d,i=min((math.hypot(cx-x,cy-y),i) for i,(x,y) in enumerate(TARGETS))
   if d<=MATCH_PX and hits[i] is None:hits[i]=b
   else:extra+=1
  n=sum(x is not None for x in hits);tp+=n;fn+=2-n;fp+=extra;pairs+=n==2;exact+=n==2 and extra==0
  per.append({'frame':row['frame'],'hits':n,'extras':extra,'count':n+extra})
 return {'scene':scene,'frames':len(frames),'paired_frames':pairs,'exact_two_frames':exact,'tp':tp,'fp':fp,'fn':fn,'per_frame':per}

def main():
 t0=time.perf_counter(); image_cache={};pubrows=[];feature_cost=[]
 for row in PUB:
  b=req(f"{REPO}/{row['image_id']}.png");image=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
  boxes=[dict(x) for x in row['full_frame'] if aspect_ok(x)]
  for box in boxes:
   box['_purity']={c['name']:purity(image,box,c) for c in COLOR_CONFIGS}
  pubrows.append({'image_id':row['image_id'],'labels':row['labels'],'boxes':boxes})
  image_cache[row['image_id']]=image
 vids=[]
 for clip in VID:
  cap=cv2.VideoCapture(str(ROOT/clip['video']));frames=[]
  for row in clip['per_frame']:
   ok,frame=cap.read()
   if not ok:break
   left=frame[:,:1600];boxes=[dict(x) for x in row['detections'] if aspect_ok(x)]
   for box in boxes:box['_purity']={c['name']:purity(left,box,c) for c in COLOR_CONFIGS}
   frames.append({'frame':row['frame'],'boxes':boxes})
  cap.release();vids.append({'scene':clip['scene'],'frames':frames})
 baseline=lambda r:select(r['boxes'])
 pubbase=public_score(pubrows,baseline)
 vbase=[video_score(v['scene'],v['frames'],lambda r:select(r['boxes'])) for v in vids]
 # The existing candidate-count adaptive rule is included as reference.
 def adaptive(r):
  boxes=[b for b in r['boxes'] if b['confidence']>=BASE and aspect_ok(b)];cut=.2 if len(boxes)>=3 else BASE
  return [b for b in boxes if b['confidence']>=cut]
 pub_ad=public_score(pubrows,adaptive)
 v_ad=[video_score(v['scene'],v['frames'],lambda r:adaptive(r)) for v in vids]
 configs=[]
 for color in COLOR_CONFIGS:
  for gate in GATES:
   for cutoff in PURITY:
    sel=lambda r,c=color,g=gate,p=cutoff:select(r['boxes'],c,g,p)
    configs.append({'color':color,'weak_box_confidence_gate':gate,'minimum_central_yellow_fraction':cutoff,
                    'public':public_score(pubrows,sel),
                    'videos':[video_score(v['scene'],v['frames'],sel) for v in vids]})
 configs.sort(key=lambda x:(-x['public']['exact_count_images'],x['public']['count_mae'],-x['public']['recall']))
 out={'method':'Precompute central-70%-of-box HSV yellow fraction for every FRC YOLO box. Keep all boxes at confidence >= weak-box gate; for lower boxes above the active .05 floor, reject only when yellow fraction is below threshold. Compare against fixed .05 and existing count-triggered .20 rule. Public images scored by IoU >=.5; saved videos scored against two fixed centers within 60 px.','dataset':{'public_source':'https://huggingface.co/datasets/project516/rebuilt-fuel-dataset','public_images':len(pubrows),'video_frames':{v['scene']:len(v['frames']) for v in vids},'split_manifest':False},'configurations':len(configs),'color_configs':COLOR_CONFIGS,'weak_box_confidence_gates':GATES,'yellow_fraction_cutoffs':PURITY,'baseline':{'public':pubbase,'videos':vbase},'adaptive_confidence_reference':{'public':pub_ad,'videos':v_ad},'ranked_color_purity_results':configs,'cached_public_box_features':[{'image_id':r['image_id'],'labels':r['labels'],'boxes':r['boxes']} for r in pubrows],'cached_video_box_features':[{'scene':v['scene'],'frames':v['frames']} for v in vids],'elapsed_s':round(time.perf_counter()-t0,2),'limitations':'Color purity can be confounded by yellow field/robot objects and varies with illumination and camera white balance. The public stills have no official split manifest and no range labels. The two video clips contain only two stationary balls and do not validate color generalization or absolute range.'}
 path=ROOT/'media/frc-fuel-color-purity-gate-evaluation.json';path.write_text(json.dumps(out,indent=2)+'\n')
 print('BASE',json.dumps({'public':{k:v for k,v in pubbase.items() if k!='per_image'},'videos':[{k:v for k,v in r.items() if k!='per_frame'} for r in vbase]},indent=2))
 print('ADAPTIVE',json.dumps({'public':{k:v for k,v in pub_ad.items() if k!='per_image'},'videos':[{k:v for k,v in r.items() if k!='per_frame'} for r in v_ad]},indent=2))
 for c in configs[:15]:print('TOP',c['color']['name'],c['weak_box_confidence_gate'],c['minimum_central_yellow_fraction'],'public',c['public']['exact_count_images'],c['public']['count_mae'],c['public']['tp'],c['public']['fp'],c['public']['fn'],'videos',[(v['scene'],v['paired_frames'],v['exact_two_frames'],v['fp']) for v in c['videos']])
 print('saved',path)
if __name__=='__main__':main()
