#!/usr/bin/env python3
"""Check luma-triggered gamma preprocessing on labeled public 2026 FUEL images."""
import json,math,os,statistics,sys,time,urllib.request,urllib.error
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scout.detectors import TensorRTDetector
REPO='https://huggingface.co/datasets/project516/rebuilt-fuel-dataset/resolve/main';ENGINE=ROOT/'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
GATE=20.;GAMMA=.60;CONF=.05;NMS=.15;IOU=.5;LUT=np.clip((np.arange(256,dtype=np.float32)/255.)**GAMMA*255,0,255).astype(np.uint8)
def get(url):
 req=urllib.request.Request(url,headers={'User-Agent':'Scout-FRC-evaluation/1.0'})
 with urllib.request.urlopen(req,timeout=90) as r:return r.read()
def labels(image_id,w,h):
 text=get(f'{REPO}/{image_id}.txt').decode().strip();out=[]
 for line in text.splitlines():
  cls,cx,cy,bw,bh=map(float,line.split());out.append({'class_id':int(cls),'x1':(cx-bw/2)*w,'y1':(cy-bh/2)*h,'x2':(cx+bw/2)*w,'y2':(cy+bh/2)*h})
 return out
def iou(a,b):
 x1=max(a['x1'],b['x1']);y1=max(a['y1'],b['y1']);x2=min(a['x2'],b['x2']);y2=min(a['y2'],b['y2']);inter=max(0,x2-x1)*max(0,y2-y1);aa=max(0,a['x2']-a['x1'])*max(0,a['y2']-a['y1']);ab=max(0,b['x2']-b['x1'])*max(0,b['y2']-b['y1']);return inter/(aa+ab-inter) if aa+ab>inter else 0.
def match(pred,gt):
 pairs=sorted((iou(p,g),pi,gi) for pi,p in enumerate(pred) for gi,g in enumerate(gt));up=set();ug=set()
 for ov,pi,gi in reversed(pairs):
  if ov<IOU:break
  if pi not in up and gi not in ug:up.add(pi);ug.add(gi)
 return len(up),len(pred)-len(up),len(gt)-len(ug)
def score(rows,key):
 tp=fp=fn=exact=0;errs=[]
 for r in rows:
  preds=[b for b in r[key] if b['confidence']>=CONF];a,b,c=match(preds,r['labels']);tp+=a;fp+=b;fn+=c;err=abs(len(preds)-len(r['labels']));errs.append(err);exact+=err==0
 return {'images':len(rows),'tp':tp,'fp':fp,'fn':fn,'precision':round(tp/(tp+fp),5) if tp+fp else None,'recall':round(tp/(tp+fn),5) if tp+fn else None,'exact_count_images':exact,'mae':round(statistics.mean(errs),5) if errs else None}
os.environ.update(SCOUT_CLASS_NAMES='["ball"]',SCOUT_MAX_BOX_ASPECT_RATIO='2.0',SCOUT_NMS_IOU=str(NMS))
det=TensorRTDetector(str(ENGINE),confidence=.02,iou=NMS);rows=[];lat=[];t0=time.perf_counter()
try:
 for image_id in range(1,55):
  try:data=get(f'{REPO}/{image_id}.png')
  except urllib.error.HTTPError as e:
   if e.code==404:continue
   raise
  image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
  if image is None:raise RuntimeError(f'bad image {image_id}')
  h,w=image.shape[:2];gt=labels(image_id,w,h);mean=float(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).mean());raw,ms=det.infer(image);lat.append(ms)
  pack=lambda bs:[{'x1':float(b.x1),'y1':float(b.y1),'x2':float(b.x2),'y2':float(b.y2),'confidence':float(b.confidence)} for b in bs]
  row={'image_id':image_id,'width':w,'height':h,'mean_luma':mean,'labels':gt,'raw':pack(raw),'gamma':None}
  if mean<GATE:
   g,_=det.infer(cv2.LUT(image,LUT));row['gamma']=pack(g)
  else:row['gamma']=row['raw']
  rows.append(row)
finally:det.close()
out={'method':'Run current FRC 9550 YOLO26n 1280x960 FP16 on all available Project516 labeled public images at candidate confidence .02 and NMS .15. Compare raw confidence .05 scoring with gamma .60 applied only when original image mean grayscale luma <20. Labels matched at IoU >=.50.','limitations':'The public dataset is monocular and has no official split manifest; it contains no stereo/range labels. The evaluated checkpoint is separate from Project516 but may share training data. This tests count only, not low-light sensor noise or motion.','source':REPO,'images':len(rows),'missing_image_ids':[i for i in range(1,55) if not any(r['image_id']==i for r in rows)],'configuration':{'engine':str(ENGINE.relative_to(ROOT)),'candidate_confidence':.02,'operating_confidence':CONF,'nms_iou':NMS,'luma_gate':GATE,'gamma':GAMMA,'iou_match':IOU},'luma_percentiles':[round(float(np.percentile([r['mean_luma'] for r in rows],q)),2) for q in (5,50,95)],'gamma_applied_images':sum(r['mean_luma']<GATE for r in rows),'median_inference_ms':round(statistics.median(lat),2) if lat else None,'elapsed_s':round(time.perf_counter()-t0,2),'raw':score(rows,'raw'),'adaptive_gamma':score(rows,'gamma'),'per_image':[{k:r[k] for k in ('image_id','width','height','mean_luma')} for r in rows]}
p=ROOT/'media/frc-fuel-luma-gamma-public-images.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('images','missing_image_ids','luma_percentiles','gamma_applied_images','raw','adaptive_gamma','elapsed_s')}));print('saved',p)
