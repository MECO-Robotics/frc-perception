#!/usr/bin/env python3
"""Compare full-frame FRC YOLO26n with overlapping 1600x1200 tile inference."""
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO = 'https://huggingface.co/datasets/project516/rebuilt-fuel-dataset/resolve/main'
ENGINE = ROOT / 'models/rebuilt_fuel_yolo26n_1280x960_fp16.engine'
BASE_CONF = 0.02
SCORES = (0.05, 0.075, 0.10)
NMS_IOU = 0.15
TILE_W, TILE_H = 1600, 1200

os.environ['SCOUT_CLASS_NAMES'] = '["ball"]'
os.environ['SCOUT_MAX_BOX_ASPECT_RATIO'] = '2.0'
from scout.detectors import TensorRTDetector


def get_bytes(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Scout-FRC-evaluation/1.0'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def read_labels(image_id, width, height):
    lines = get_bytes(f'{REPO}/{image_id}.txt').decode().strip().splitlines()
    out = []
    for line in lines:
        cid, cx, cy, bw, bh = map(float, line.split())
        out.append({'class_id': int(cid), 'x1': (cx-bw/2)*width,
                    'y1': (cy-bh/2)*height, 'x2': (cx+bw/2)*width,
                    'y2': (cy+bh/2)*height})
    return out


def iou(a, b):
    x1, y1 = max(a['x1'], b['x1']), max(a['y1'], b['y1'])
    x2, y2 = min(a['x2'], b['x2']), min(a['y2'], b['y2'])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a['x2']-a['x1']) * max(0, a['y2']-a['y1'])
    area_b = max(0, b['x2']-b['x1']) * max(0, b['y2']-b['y1'])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def nms(boxes):
    ordered = sorted(boxes, key=lambda b: b['confidence'], reverse=True)
    keep = []
    for box in ordered:
        if all(iou(box, selected) <= NMS_IOU for selected in keep):
            keep.append(box)
    return keep


def grid_offsets(length, tile):
    if length <= tile:
        return [0]
    # Two columns / four rows for 3024x4032 source images, with overlap to
    # reduce truncation at tile boundaries and include the far image edge.
    count = int(np.ceil((length - tile) / (tile * 0.9))) + 1
    return sorted(set(int(round(i * (length - tile) / (count - 1)))
                      for i in range(count)))


def tile_boxes(detector, image):
    h, w = image.shape[:2]
    xs, ys = grid_offsets(w, TILE_W), grid_offsets(h, TILE_H)
    predictions, latencies = [], []
    for y in ys:
        for x in xs:
            crop = image[y:min(y+TILE_H, h), x:min(x+TILE_W, w)]
            detections, latency = detector.infer(crop)
            latencies.append(latency)
            for d in detections:
                predictions.append({'class_id': d.class_id,
                    'x1': float(d.x1+x), 'y1': float(d.y1+y),
                    'x2': float(d.x2+x), 'y2': float(d.y2+y),
                    'confidence': float(d.confidence),
                    'tile_x': x, 'tile_y': y})
    return nms(predictions), latencies, len(xs)*len(ys)


def match(pred, gt):
    pairs = sorted(((iou(p, g), pi, gi) for pi, p in enumerate(pred)
                    for gi, g in enumerate(gt)), reverse=True)
    used_p, used_g = set(), set()
    for overlap, pi, gi in pairs:
        if overlap < .5:
            break
        if pi not in used_p and gi not in used_g:
            used_p.add(pi); used_g.add(gi)
    return len(used_p), len(pred)-len(used_p), len(gt)-len(used_g)


def summarize(rows, mode, threshold):
    tp=fp=fn=exact=0; errors=[]; records=[]
    for row in rows:
        pred=[b for b in row[mode] if b['confidence'] >= threshold]
        a,b,c=match(pred,row['labels']); tp+=a;fp+=b;fn+=c
        err=abs(len(pred)-len(row['labels'])); errors.append(err); exact += err==0
        records.append({'image_id':row['image_id'],'gt':len(row['labels']),
                        'pred':len(pred),'tp':a,'fp':b,'fn':c,'count_error':err})
    return {'images':len(rows),'tp':tp,'fp':fp,'fn':fn,
            'precision':round(tp/(tp+fp),5) if tp+fp else None,
            'recall':round(tp/(tp+fn),5) if tp+fn else None,
            'exact_count_images':exact,
            'exact_count_fraction':round(exact/len(rows),5) if rows else None,
            'count_mae':round(statistics.mean(errors),5) if errors else None,
            'per_image':records}


def main():
    detector=TensorRTDetector(str(ENGINE),confidence=BASE_CONF,iou=NMS_IOU)
    rows=[]; missing=[]; timings=[];tile_calls=0; started=time.perf_counter()
    try:
        warm=get_bytes(f'{REPO}/1.png')
        warm_img=cv2.imdecode(np.frombuffer(warm,np.uint8),cv2.IMREAD_COLOR)
        detector.infer(warm_img)
        for image_id in range(1,55):
            try: encoded=get_bytes(f'{REPO}/{image_id}.png')
            except urllib.error.HTTPError as exc:
                if exc.code==404: missing.append(image_id); continue
                raise
            image=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
            h,w=image.shape[:2]; labels=read_labels(image_id,w,h)
            t=time.perf_counter(); full,full_ms=detector.infer(image)
            full=[{'class_id':d.class_id,'x1':float(d.x1),'y1':float(d.y1),
                   'x2':float(d.x2),'y2':float(d.y2),'confidence':float(d.confidence)}
                  for d in full]
            tiled,tile_ms,n_tiles=tile_boxes(detector,image); elapsed=time.perf_counter()-t
            tile_calls+=n_tiles; timings.append({'image_id':image_id,
                'full_ms':round(full_ms,2),'tiled_ms':round(sum(tile_ms),2),
                'pipeline_wall_ms':round(elapsed*1000,2),'tiles':n_tiles})
            rows.append({'image_id':image_id,'width':w,'height':h,'labels':labels,
                         'full_frame':full,'tiled':tiled})
            print(f'{image_id:02d}: gt={len(labels)} full={len(full)} tiled={len(tiled)} tiles={n_tiles}',flush=True)
    finally: detector.close()
    summaries={}
    for t in SCORES:
        summaries[str(t)]={'full_frame':summarize(rows,'full_frame',t),
                           'tiled':summarize(rows,'tiled',t)}
    output={'method':'Run the FRC 9550 YOLO26n 1280x960 FP16 engine on each image once at full-frame scale, then on an overlapping grid of native 1600x1200 crops. Tile boxes are offset to source coordinates and merged with global greedy NMS IoU 0.15. Candidate threshold 0.02; score at 0.05/0.075/0.10; match labels at IoU >=0.50.','dataset':{'source':'https://huggingface.co/datasets/project516/rebuilt-fuel-dataset','images':len(rows),'missing_image_ids':missing,'ground_truth_count_histogram':{str(n):sum(len(r['labels'])==n for r in rows) for n in sorted({len(r['labels']) for r in rows})}},'configuration':{'engine':str(ENGINE.relative_to(ROOT)),'engine_input':'1280x960','tile_size':[TILE_W,TILE_H],'tile_grid':'overlapping uniform grid spanning image edges','base_confidence':BASE_CONF,'score_thresholds':SCORES,'nms_iou':NMS_IOU,'shape_limit':2.0,'tile_calls':tile_calls},'limitations':'This Project516 set has no split manifest and may not be an independent held-out sample. It is monocular and cannot score range or 3D positions. Tiled inference is an offline count/localization experiment; its multiple engine calls per input image are not a 60 fps runtime proposal.','timing':{'elapsed_s':round(time.perf_counter()-started,2),'full_frame_median_ms':round(statistics.median(x['full_ms'] for x in timings),2),'tile_engine_calls_per_image_median':round(statistics.median(x['tiles'] for x in timings),1),'tiled_engine_median_ms_per_image':round(statistics.median(x['tiled_ms'] for x in timings),2),'wall_median_ms_per_image':round(statistics.median(x['pipeline_wall_ms'] for x in timings),2),'per_image':timings},'summary_by_score_threshold':summaries,'per_image_counts':{str(t):{'full_frame':summaries[str(t)]['full_frame']['per_image'],'tiled':summaries[str(t)]['tiled']['per_image']} for t in SCORES},'detections':rows}
    path=ROOT/'media/frc-fuel-tiled-detection-evaluation.json';path.write_text(json.dumps(output,indent=2)+'\n')
    compact={t:{mode:{k:v for k,v in pair[mode].items() if k!='per_image'} for mode in ('full_frame','tiled')} for t,pair in summaries.items()}
    print(json.dumps({'images':len(rows),'missing':missing,'tile_calls':tile_calls,'timing':output['timing'],'summary':compact},indent=2));print('saved',path)

if __name__=='__main__': main()
