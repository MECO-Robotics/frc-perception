#!/usr/bin/env python3
"""Test whether yellow foreground inside YOLO boxes can split merged FUEL detections."""
import json, math, statistics, sys, urllib.request
from pathlib import Path
import cv2, numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'media/frc-fuel-tiled-detection-evaluation.json').read_text())['detections']
REPO='https://huggingface.co/datasets/project516/rebuilt-fuel-dataset/resolve/main'
CONF=.05; IOU_MATCH=.5; NMS_IOU=.15
CONFIGS=[{'name':f'h{h0}-{h1}_s{s}_v{v}_peak{ratio}', 'hue':(h0,h1),'sat':s,'val':v,'peak_ratio':ratio}
 for h0,h1,s,v in [(10,40,40,40),(10,40,70,40),(10,40,70,80),(15,35,70,40)] for ratio in (.45,.60)]

def get_image(i):
 req=urllib.request.Request(f'{REPO}/{i}.png',headers={'User-Agent':'Scout-FRC-evaluation/1.0'})
 return cv2.imdecode(np.frombuffer(urllib.request.urlopen(req,timeout=90).read(),np.uint8),cv2.IMREAD_COLOR)
def iou(a,b):
 x1=max(a['x1'],b['x1']);y1=max(a['y1'],b['y1']);x2=min(a['x2'],b['x2']);y2=min(a['y2'],b['y2']); inter=max(0,x2-x1)*max(0,y2-y1)
 aa=max(0,a['x2']-a['x1'])*max(0,a['y2']-a['y1']);bb=max(0,b['x2']-b['x1'])*max(0,b['y2']-b['y1']);u=aa+bb-inter
 return inter/u if u else 0
def nms(boxes):
 kept=[]
 for b in sorted(boxes,key=lambda x:x['confidence'],reverse=True):
  if all(iou(b,k)<=NMS_IOU for k in kept):kept.append(b)
 return kept
def match(pred,gt):
 pairs=sorted(((iou(a,b),i,j) for i,a in enumerate(pred) for j,b in enumerate(gt)),reverse=True);up=set();ug=set()
 for q,i,j in pairs:
  if q<IOU_MATCH:break
  if i not in up and j not in ug:up.add(i);ug.add(j)
 return len(up),len(pred)-len(up),len(gt)-len(ug)
def find_peaks(component,ratio,min_side):
 dist=cv2.distanceTransform(component.astype(np.uint8),cv2.DIST_L2,5)
 mx=float(dist.max())
 if mx<4:return []
 peak_mask=(dist>=max(4,mx*ratio)).astype(np.uint8)
 n,labs,stats,cents=cv2.connectedComponentsWithStats(peak_mask,8)
 peaks=[]
 for i in range(1,n):
  if stats[i,cv2.CC_STAT_AREA]<1:continue
  yy,xx=np.where(labs==i)
  j=int(np.argmax(dist[yy,xx])); peaks.append((float(xx[j]),float(yy[j]),float(dist[yy[j],xx[j]])))
 peaks.sort(key=lambda x:x[2],reverse=True);kept=[];sep=max(8,min_side*.23)
 for p in peaks:
  if all(math.hypot(p[0]-q[0],p[1]-q[1])>=sep for q in kept):kept.append(p)
 return kept

def split_detection(image,b,cfg):
 h,w=image.shape[:2];bw=b['x2']-b['x1'];bh=b['y2']-b['y1'];padx=round(.04*bw);pady=round(.04*bh)
 x0=max(0,int(b['x1'])-padx);y0=max(0,int(b['y1'])-pady);x1=min(w,int(b['x2'])+padx);y1=min(h,int(b['y2'])+pady)
 crop=image[y0:y1,x0:x1]
 if crop.size==0:return [dict(b)]
 hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV); mask=cv2.inRange(hsv,(cfg['hue'][0],cfg['sat'],cfg['val']),(cfg['hue'][1],255,255))
 mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
 n,labs,stats,_=cv2.connectedComponentsWithStats(mask,8);instances=[]
 for label in range(1,n):
  x,y,cw,ch,area=map(int,stats[label]);
  if area<max(100,.002*bw*bh) or min(cw,ch)<8 or max(cw/ch,ch/cw)>3.5:continue
  comp=labs[y:y+ch,x:x+cw]==label; peaks=find_peaks(comp,cfg['peak_ratio'],min(cw,ch))
  if len(peaks)<2:
   instances.append((x+x0,y+y0,cw,ch,area));continue
  yy,xx=np.where(comp); center_arr=np.array([(p[0],p[1]) for p in peaks],np.float32)
  pts=np.column_stack((xx,yy)).astype(np.float32); nearest=np.argmin(((pts[:,None,:]-center_arr[None,:,:])**2).sum(axis=2),axis=1)
  for pi in range(len(peaks)):
   sel=nearest==pi
   if int(sel.sum())<max(100,.002*bw*bh):continue
   sx,sy=xx[sel],yy[sel]
   instances.append((x+x0+int(sx.min()),y+y0+int(sy.min()),int(sx.max()-sx.min()+1),int(sy.max()-sy.min()+1),int(sel.sum())))
 if len(instances)<2:return [dict(b)]
 out=[]
 for xx,yy,ww,hh,area in instances:
  out.append({'class_id':b.get('class_id',0),'x1':xx,'y1':yy,'x2':xx+ww,'y2':yy+hh,'confidence':b['confidence'],'split_from_box':True})
 return out

def score(preds,gt):
 tp=fp=fn=exact=0;errs=[];per=[]
 for row in preds:
  pred=nms([b for b in row['predictions'] if b['confidence']>=CONF]);g=row['labels'];a,b,c=match(pred,g);tp+=a;fp+=b;fn+=c;e=abs(len(pred)-len(g));exact+=e==0;errs.append(e);per.append({'image_id':row['image_id'],'gt':len(g),'pred':len(pred),'tp':a,'fp':b,'fn':c,'count_error':e})
 return {'images':len(preds),'tp':tp,'fp':fp,'fn':fn,'precision':round(tp/(tp+fp),5) if tp+fp else None,'recall':round(tp/(tp+fn),5) if tp+fn else None,'exact_count_images':exact,'exact_count_fraction':round(exact/len(preds),5),'count_mae':round(statistics.mean(errs),5),'per_image':per}

def main():
 images={}; output_rows={c['name']:[] for c in CONFIGS}; baserows=[]; t0=cv2.getTickCount()
 for ix,row in enumerate(DATA):
  image=get_image(row['image_id']);images[row['image_id']]=image
  baseline=[b for b in row['full_frame'] if b['confidence']>=CONF]
  baserows.append({'image_id':row['image_id'],'labels':row['labels'],'predictions':baseline})
  for cfg in CONFIGS:
   preds=[]
   for b in baseline:preds.extend(split_detection(image,b,cfg))
   output_rows[cfg['name']].append({'image_id':row['image_id'],'labels':row['labels'],'predictions':preds})
  print(f"{ix+1}/{len(DATA)} image {row['image_id']}",flush=True)
 base=score(baserows,[])
 # score() expects predictions in row but does not use the second argument.
 results=[]
 for cfg in CONFIGS:
  sc=score(output_rows[cfg['name']],[]);sc['config']=cfg;results.append(sc)
 results.sort(key=lambda r:(-r['exact_count_images'],r['count_mae'],-r['precision']))
 out={'method':'Apply HSV yellow masking only inside current FRC detector boxes on the 53 labeled Project516 stills. If a box mask forms multiple components or multiple distance-transform peaks, partition foreground by nearest peak and replace the detector box with per-part boxes; otherwise preserve original box. Confidence .05, global NMS .15, labels matched at IoU .50. Sweep hue/S/V cutoffs and peak threshold.','dataset':{'source':'https://huggingface.co/datasets/project516/rebuilt-fuel-dataset','images':len(DATA),'missing_image_ids':[3],'split_manifest':False},'limitations':'Project516 images have no train/validation manifest. They are monocular, so this only evaluates box count/localization, not stereo range. The color split is a post-processing candidate, not used by the running service.','baseline':base,'configurations':len(CONFIGS),'results':results,'detections_by_config':{name:[{'image_id':r['image_id'],'predictions':r['predictions']} for r in rows] for name,rows in output_rows.items()}}
 path=ROOT/'media/frc-fuel-box-segmentation-split-evaluation.json';path.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'baseline':{k:v for k,v in base.items() if k!='per_image'},'results':[{k:v for k,v in r.items() if k not in ('per_image',)} for r in results]},indent=2));print('saved',path)
if __name__=='__main__':main()
