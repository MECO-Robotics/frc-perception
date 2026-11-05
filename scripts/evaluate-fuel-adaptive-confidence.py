#!/usr/bin/env python3
"""Evaluate candidate-count-triggered confidence cutoffs on saved FUEL data."""
import importlib.util
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = json.loads((ROOT/'media/frc-fuel-project516-box-geometry-evaluation.json').read_text())
VIDEOS = json.loads((ROOT/'media/frc-fuel-dense-count-nms010.json').read_text())
STEREO = json.loads((ROOT/'media/frc-fuel-dense-stereo-nms015-conf005.json').read_text())
BASE_CONF = 0.05
MAX_ASPECT = 2.0
TRIGGERS = (3, 4, 5, 6, 8, 10, 12, 16, 20, 30)
HIGH_CONFS = (0.075, 0.10, 0.15, 0.20, 0.30, 0.40)
TARGETS = ((808,589),(902,591))
spec = importlib.util.spec_from_file_location(
    'project516_eval', ROOT/'scripts/evaluate-fuel-project516-public-dataset.py')
_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_module)


def public_summary(records):
    tp=sum(x['tp'] for x in records); fp=sum(x['fp'] for x in records); fn=sum(x['fn'] for x in records)
    errors=[abs(x['predicted_count']-x['ground_truth_count']) for x in records]
    exact=sum(error==0 for error in errors)
    return {'tp':tp,'fp':fp,'fn':fn,'precision':round(tp/(tp+fp),5) if tp+fp else None,
            'recall':round(tp/(tp+fn),5) if tp+fn else None,'exact_images':exact,
            'exact_fraction':round(exact/len(records),5) if records else None,
            'mae':round(statistics.mean(errors),4) if errors else None}


def video_target_boxes(detections, threshold):
    hits=[None,None]; extras=0
    for b in sorted((b for b in detections if b['confidence']>=threshold),key=lambda b:-b['confidence']):
        cx=(b['x1']+b['x2'])/2;cy=(b['y1']+b['y2'])/2
        distance,i=min((math.hypot(cx-x,cy-y),i) for i,(x,y) in enumerate(TARGETS))
        if distance<=60 and hits[i] is None:hits[i]=b
        else:extras+=1
    return hits,extras


def summarize_range(values):
    if not values:return None
    median=statistics.median(values)
    return {'n':len(values),'median_m':round(median,5),
            'mad_m':round(statistics.median(abs(v-median) for v in values),5)}


def configured_boxes(boxes):
    return [box for box in boxes
            if max((box['x2']-box['x1'])/max(box['y2']-box['y1'],1e-9),
                   (box['y2']-box['y1'])/max(box['x2']-box['x1'],1e-9)) <= MAX_ASPECT]

# Join range frames by scene and frame number. These are from the same confidence-0.05 boxes.
stereo_by_scene={r['scene']:r['frames'] for r in STEREO['results']}
public_summaries={}
video_summaries={}
per_config=[]
for trigger in TRIGGERS:
    for high_conf in HIGH_CONFS:
        key=f'{trigger}/{high_conf}'
        public_rows=[]
        for image in PUBLIC['per_image_predictions']:
            boxes=configured_boxes(image['predictions'])
            candidate_count=sum(b['confidence']>=BASE_CONF for b in boxes)
            chosen_conf=high_conf if candidate_count>=trigger else BASE_CONF
            preds=[b for b in boxes if b['confidence']>=chosen_conf]
            # Greedy label matching uses the same helper as the original dataset evaluation.
            tp,fp,fn=_module.match(preds,image['labels'],0.5)
            public_rows.append({'predicted_count':len(preds),'ground_truth_count':image['ground_truth_count'],
                                'tp':tp,'fp':fp,'fn':fn})
        public_summaries[key]=public_summary(public_rows)

        for clip in VIDEOS['results']:
            counts={'samples':len(clip['per_frame']),'pairs':0,'exact_two':0,'tp':0,'fp':0,'fn':0,
                    'ranges':[[],[]],'valid_range_pairs':0,'triggered_frames':0}
            stereo_frames=stereo_by_scene[clip['scene']]
            for row,srow in zip(clip['per_frame'],stereo_frames):
                boxes=configured_boxes(row['detections'])
                candidate_count=sum(b['confidence']>=BASE_CONF for b in boxes)
                chosen_conf=high_conf if candidate_count>=trigger else BASE_CONF
                hits,extras=video_target_boxes(boxes,chosen_conf)
                hit_count=sum(b is not None for b in hits)
                counts['triggered_frames']+=candidate_count>=trigger
                counts['tp']+=hit_count;counts['fn']+=2-hit_count;counts['fp']+=extras
                counts['pairs']+=hit_count==2
                counts['exact_two']+=hit_count==2 and extras==0
                for i,b in enumerate(hits):
                    if b is not None and srow['ranges_m'][i] is not None:
                        counts['ranges'][i].append(srow['ranges_m'][i])
                if all(b is not None for b in hits) and all(x is not None for x in srow['ranges_m']):
                    counts['valid_range_pairs']+=1
            video_summaries.setdefault(key,{})[clip['scene']]={
                'samples':counts['samples'],'triggered_frames':counts['triggered_frames'],
                'paired_frames':counts['pairs'],'exact_two_frames':counts['exact_two'],
                'tp':counts['tp'],'fp':counts['fp'],'fn':counts['fn'],
                'valid_range_pairs':counts['valid_range_pairs'],
                'ranges_m_by_target':[summarize_range(v) for v in counts['ranges']]}
        per_config.append({'trigger_candidates':trigger,'high_confidence':high_conf,
                           'public':public_summaries[key],'video':video_summaries[key]})

output={'method':('If the number of detections at confidence 0.05 is at least K, raise the frame confidence '
                  'threshold; otherwise retain 0.05. Evaluate image count against Project516 labels and the '
                  'two fixed target windows in both stereo videos. Reuse same-frame SGBM ranges because '
                  'thresholding only removes boxes and does not change retained box coordinates. Apply the '
                  'active maximum box aspect ratio 2.0 before counting candidates.'),
        'base_confidence':BASE_CONF,'candidate_count_triggers':TRIGGERS,'high_confidence_values':HIGH_CONFS,
        'public_dataset':'https://huggingface.co/datasets/project516/rebuilt-fuel-dataset',
        'limitations':'Static image/video scenes only; public set has no official split manifest. A frame-level candidate count trigger may behave differently on motion, occlusion, and field lighting.',
        'results':per_config}
path=ROOT/'media/frc-fuel-adaptive-confidence-evaluation.json'
path.write_text(json.dumps(output,indent=2)+'\n')
ranked=sorted(per_config,key=lambda r:(r['public']['mae'],-r['public']['exact_images'],-r['public']['recall']))
print('best public results')
for r in ranked[:8]:print(r['trigger_candidates'],r['high_confidence'],r['public'])
print('best video-safe candidates')
safe = [r for r in per_config
        if r['video']['earlier']['paired_frames'] >= 575
        and r['video']['corrected-settings']['paired_frames'] >= 600]
for r in sorted(safe,key=lambda r:(r['public']['mae'],-r['public']['exact_images']))[:8]:
    print(r['trigger_candidates'],r['high_confidence'],r['public'],
          {k:(v['paired_frames'],v['exact_two_frames']) for k,v in r['video'].items()})
print('saved',path)
