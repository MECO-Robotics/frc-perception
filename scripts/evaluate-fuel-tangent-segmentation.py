#!/usr/bin/env python3
"""Exploratory offline tangent-circle split using cached detections (no detector inference)."""
import cv2, json, math, time, urllib.request, statistics
import numpy as np
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'media/frc-fuel-tiled-detection-evaluation.json').read_text())['detections']
OLD=json.loads((ROOT/'media/frc-fuel-box-segmentation-split-evaluation.json').read_text())
REPO='https://huggingface.co/datasets/project516/rebuilt-fuel-dataset/resolve/main'
CONF=.05

def get_image(i):
    req=urllib.request.Request(f'{REPO}/{i}.png',headers={'User-Agent':'Scout-FRC-evaluation/1.0'})
    return cv2.imdecode(np.frombuffer(urllib.request.urlopen(req,timeout=90).read(),np.uint8),cv2.IMREAD_COLOR)
def iou(a,b):
    x1=max(a['x1'],b['x1']);y1=max(a['y1'],b['y1']);x2=min(a['x2'],b['x2']);y2=min(a['y2'],b['y2'])
    inter=max(0,x2-x1)*max(0,y2-y1); aa=(a['x2']-a['x1'])*(a['y2']-a['y1']);bb=(b['x2']-b['x1'])*(b['y2']-b['y1'])
    return inter/(aa+bb-inter) if aa+bb>inter else 0
def center(b): return ((b['x1']+b['x2'])/2,(b['y1']+b['y2'])/2)
def score(rows):
    counts=[];tp=fp=fn=0;center_errors=[]
    for r in rows:
        pred=r['pred'];gt=r['gt']; counts.append(abs(len(pred)-len(gt)))
        pairs=sorted(((iou(p,g),i,j) for i,p in enumerate(pred) for j,g in enumerate(gt)),reverse=True);up=set();ug=set()
        for q,i,j in pairs:
            if q<.5: break
            if i not in up and j not in ug:up.add(i);ug.add(j)
        tp+=len(up);fp+=len(pred)-len(up);fn+=len(gt)-len(ug)
        pairs=sorted((math.dist(center(p),center(g)),i,j) for i,p in enumerate(pred) for j,g in enumerate(gt));up=set();ug=set()
        for d,i,j in pairs:
            if i not in up and j not in ug:up.add(i);ug.add(j);center_errors.append(d)
    return {'images':len(rows),'count_exact_fraction':sum(x==0 for x in counts)/len(counts),'count_mae':statistics.mean(counts),
            'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn) if tp+fn else 0,
            'matched_center_error_px_median':statistics.median(center_errors) if center_errors else None,
            'matched_center_error_px_p90':float(np.percentile(center_errors,90)) if center_errors else None}

def tangent_split(img,b):
    """HSV foreground distance-transform proposals; split only if two fitted disks are near-tangent."""
    x0,y0,x1,y1=map(int,(b['x1'],b['y1'],b['x2'],b['y2']));w=x1-x0;h=y1-y0
    if w<20 or h<20:return [b]
    pad=max(3,int(.04*min(w,h))); xa=max(0,x0-pad);ya=max(0,y0-pad);xb=min(img.shape[1],x1+pad);yb=min(img.shape[0],y1+pad)
    crop=img[ya:yb,xa:xb];hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,(10,65,45),(42,255,255));mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8));mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    # Retain only foreground overlapping the detector ROI, preventing nearby field markings from seeding a split.
    core=np.zeros_like(mask);core[y0-ya:y1-ya,x0-xa:x1-xa]=mask[y0-ya:y1-ya,x0-xa:x1-xa];mask=core
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if n<=1:return [b]
    j=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));component=(lab==j).astype(np.uint8);area=stats[j,cv2.CC_STAT_AREA]
    if area < .04*w*h:return [b]
    dist=cv2.distanceTransform(component,cv2.DIST_L2,5); mx=float(dist.max())
    if mx<8:return [b]
    # Local maxima via dilation; these are candidate inscribed-circle centers.
    dil=cv2.dilate(dist,np.ones((11,11),np.uint8)); ys,xs=np.where((dist==dil)&(dist>=max(7,mx*.48)))
    cand=sorted(((float(dist[y,x]),int(x),int(y)) for y,x in zip(ys,xs)),reverse=True);peaks=[]
    for rad,x,y in cand:
        if all(math.hypot(x-p[1],y-p[2])>=max(10,.52*min(w,h)) for p in peaks):peaks.append((rad,x,y))
        if len(peaks)>=2:break
    if len(peaks)<2:return [b]
    (r1,cx1,cy1),(r2,cx2,cy2)=peaks;sep=math.hypot(cx2-cx1,cy2-cy1)
    # Radius agreement and approximate tangency screen noisy multi-peak masks.
    if min(r1,r2)<.18*min(w,h) or max(r1,r2)/min(r1,r2)>1.65 or not (.70 <= sep/(r1+r2) <= 1.45):return [b]
    # Require each disk perimeter to be mostly supported by the color mask.
    yy,xx=np.ogrid[:mask.shape[0],:mask.shape[1]]
    for rr,cx,cy in ((r1,cx1,cy1),(r2,cx2,cy2)):
        ring=np.abs(np.sqrt((xx-cx)**2+(yy-cy)**2)-rr)<=max(1.5,.10*rr)
        if ring.sum()==0 or np.count_nonzero(ring & (mask>0))/ring.sum()<.30:return [b]
    out=[]
    for rr,cx,cy in ((r1,cx1,cy1),(r2,cx2,cy2)):
        xx0=max(x0,xa+cx-int(rr));yy0=max(y0,ya+cy-int(rr));xx1=min(x1,xa+cx+int(rr)+1);yy1=min(y1,ya+cy+int(rr)+1)
        if xx1<=xx0 or yy1<=yy0:return [b]
        out.append({'class_id':b.get('class_id',0),'x1':xx0,'y1':yy0,'x2':xx1,'y2':yy1,'confidence':b['confidence'],'tangent_split':True})
    return out

def main():
    rows=[];cpu=[];split_boxes=0;processed=0;start=time.perf_counter()
    for k,r in enumerate(DATA):
        if r['image_id']==3:continue
        img=get_image(r['image_id']);pred=[b for b in r['full_frame'] if b['confidence']>=CONF];t=time.perf_counter();new=[]
        for b in pred:new.extend(tangent_split(img,b))
        elapsed=time.perf_counter()-t;cpu.append(elapsed/max(1,len(pred)));split_boxes+=sum(bool(b.get('tangent_split')) for b in new);processed+=len(pred)
        rows.append({'id':r['image_id'],'gt':r['labels'],'pred':new})
        print(f'{k+1}/{len(DATA)} image {r["image_id"]}, pred {len(pred)} -> {len(new)}, {elapsed*1000:.1f}ms',flush=True)
    # Existing strongest HSV/distance-transform config and cached detector baseline.
    base_rows=[]
    for r in DATA:
        if r['image_id']==3:continue
        base_rows.append({'id':r['image_id'],'gt':r['labels'],'pred':[b for b in r['full_frame'] if b['confidence']>=CONF]})
    oldconf=next(x for x in OLD['results'] if x['config']['name']=='h10-40_s40_v40_peak0.45')
    old_by={x['image_id']:x['predictions'] for x in OLD['detections_by_config'][oldconf['config']['name']]}
    oldrows=[{'id':r['image_id'],'gt':r['labels'],'pred':old_by[r['image_id']]} for r in DATA if r['image_id']!=3]
    out={'method':'Cached YOLO detections in each existing detector box; HSV mask, connected foreground, distance-transform disk centers, approximate tangency/radius/perimeter support checks. Detector inference excluded. Compares full-frame detector baseline, strongest prior HSV distance-transform split configuration, and candidate.',
         'dataset':'52 available Project516 public stills (image 3 missing from prior cache); no train/validation manifest; IoU .5 for localization metrics.',
         'limitations':'Static monocular labeled stills only; cannot evaluate stereo depth, temporal stability, latency under concurrent pipeline load, or generalization. CPU timing measures only local segmentation/split post-processing (image download excluded).',
         'baseline':score(base_rows),'previous_segmentation_split':score(oldrows), 'candidate':score(rows),
         'runtime':{'cpu_seconds_total':time.perf_counter()-start,'per_input_box_ms_median':statistics.median(cpu)*1000,'per_input_box_ms_p90':float(np.percentile(cpu,90))*1000,'split_output_box_fraction':split_boxes/max(1,processed),'input_boxes':processed,'split_outputs':split_boxes,'images_per_second_postprocess':len(rows)/sum([x*max(1,1) for x in []]) if False else None},
         'per_image':[{'image_id':r['id'],'predicted_count':len(r['pred']),'gt_count':len(r['gt']),'count_error':len(r['pred'])-len(r['gt'])} for r in rows]}
    p=ROOT/'media/frc-fuel-tangent-segmentation-evaluation.json';p.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['baseline','previous_segmentation_split','candidate','runtime']},indent=2));print('saved',p)
if __name__=='__main__':main()
