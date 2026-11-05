#!/usr/bin/env python3
"""Score FRC YOLO26n PyTorch inference at stride-aligned 960x704 on saved clips."""
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models/rebuilt_fuel_yolo26n.pt"
VIDEOS = (("earlier", "media/frc-2026-live-two-fuel-60fps-cfr.mp4"),
          ("corrected-settings", "media/frc-2026-fuel-camera-settings-fixed-60fps.mp4"))
DETECTIONS = json.loads((ROOT / "media/frc-fuel-dense-count-nms015.json").read_text())
SAMPLE_TIMES = (1.0,1.4,1.8,2.2,2.6,3.0,3.4,3.8,4.2,4.6,5.0,5.4,5.8,6.2,6.6)
TARGETS = ((808.0,589.0),(902.0,591.0))
THRESHOLDS = (0.05,0.075,0.10,0.15,0.20,0.25,0.30,0.40)
ASPECT_LIMIT = 2.0
CALIBRATION = ROOT / "media/calibration/stereo_calibration_live_order_candidate.yml"
BASELINE_M = 0.0635

fs=cv2.FileStorage(str(CALIBRATION),cv2.FILE_STORAGE_READ)
kl=fs.getNode("left_camera_matrix").mat();dl=fs.getNode("left_distortion").mat()
kr=fs.getNode("right_camera_matrix").mat();dr=fs.getNode("right_distortion").mat()
rotation=fs.getNode("rotation").mat();translation=fs.getNode("translation").mat();fs.release()
r1,r2,p1,p2,*_=cv2.stereoRectify(kl,dl,kr,dr,(1600,1200),rotation,translation,
                                  flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
map_l=cv2.initUndistortRectifyMap(kl,dl,r1,p1,(1600,1200),cv2.CV_32FC1)
map_r=cv2.initUndistortRectifyMap(kr,dr,r2,p2,(1600,1200),cv2.CV_32FC1)
matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=3,
    P1=8*3*3,P2=32*3*3,uniquenessRatio=8,speckleWindowSize=50,
    speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def geometry(box,disparity):
    corners=np.array([[[box["x1"],box["y1"]]],[[box["x2"],box["y1"]]],
                      [[box["x2"],box["y2"]]],[[box["x1"],box["y2"]]]],np.float32)
    rect=cv2.undistortPoints(corners,kl,dl,R=r1,P=p1).reshape(-1,2)*.5
    x1,y1=np.floor(rect.min(axis=0)).astype(int);x2,y2=np.ceil(rect.max(axis=0)).astype(int)
    x1,x2=max(0,x1),min(disparity.shape[1],x2)
    y1,y2=max(0,y1),min(disparity.shape[0],y2)
    roi=disparity[y1:y2,x1:x2]
    valid=roi[np.isfinite(roi)&(roi>1.5)&(roi<60)]
    if valid.size<8:return None
    z=BASELINE_M*float(p1[0,0])/(2*float(np.median(valid)))
    center=np.array([[[ (box["x1"]+box["x2"])/2,(box["y1"]+box["y2"])/2 ]]],np.float32)
    u,v=cv2.undistortPoints(center,kl,dl,R=r1,P=p1)[0,0]
    xyz=np.asarray(((u-p1[0,2])*z/p1[0,0],(v-p1[1,2])*z/p1[1,1],z))
    return {"range_m":float(z),"xyz_m":xyz}


def range_summary(values):
    if not values:return None
    med=float(statistics.median(values))
    return {"n":len(values),"median_m":round(med,5),
            "mad_m":round(statistics.median(abs(x-med) for x in values),5),
            "p05_m":round(float(np.percentile(values,5)),5),
            "p95_m":round(float(np.percentile(values,95)),5)}


def match(boxes, threshold):
    selected = [b for b in boxes if b["confidence"] >= threshold and
                max(b["x2"]-b["x1"],b["y2"]-b["y1"])/max(1,min(b["x2"]-b["x1"],b["y2"]-b["y1"])) <= ASPECT_LIMIT]
    selected.sort(key=lambda b:b["confidence"],reverse=True)
    hits, errors, extras = [None,None], [], 0
    for box in selected:
        c=((box["x1"]+box["x2"])/2,(box["y1"]+box["y2"])/2)
        distances=[math.dist(c,t) for t in TARGETS]
        d,i=min((v,j) for j,v in enumerate(distances))
        if d <= 60 and hits[i] is None:
            hits[i]=box; errors.append(d)
        else:
            extras+=1
    return hits,extras


def score(rows, threshold):
    tp=fp=fn=paired=exact=0
    errors=[]
    for boxes in rows:
        hits,extras=match(boxes,threshold)
        n=sum(x is not None for x in hits)
        tp+=n;fn+=2-n;fp+=extras
        paired+=n==2;exact+=n==2 and extras==0
        errors.extend(math.dist(((x["x1"]+x["x2"])/2,(x["y1"]+x["y2"])/2),TARGETS[i])
                      for i,x in enumerate(hits) if x is not None)
    return {"confidence":threshold,"tp":tp,"fp":fp,"fn":fn,
            "paired_frames":paired,"exact_two_frames":exact,
            "center_error_px":{"n":len(errors),
                "median":round(statistics.median(errors),3) if errors else None,
                "p95":round(float(np.percentile(errors,95)),3) if errors else None}}


model=YOLO(str(WEIGHTS))
# Warm model and kernels with one sampled image before collecting timing.
cap=cv2.VideoCapture(str(ROOT/VIDEOS[0][1]));cap.set(cv2.CAP_PROP_POS_MSEC,1000)
ok,frame=cap.read();cap.release()
if not ok: raise RuntimeError("Unable to read warmup frame")
model.predict(frame[:,:1600],imgsz=(704,960),conf=.02,iou=.15,device="cpu",verbose=False,max_det=300)

out={"method":"Run the FRC YOLO26n checkpoint on the same 15 timestamps per saved clip used by the existing 640/1280/1280x960/1600 TensorRT resolution comparison. Use stride-aligned 960x704 (width x height), candidate confidence 0.02, NMS IoU 0.15, and the current 2.0 aspect-ratio limit. Rescore at several confidence cutoffs and compare with cached 1280x960 TensorRT detections at identical timestamps.",
     "weights":str(WEIGHTS.relative_to(ROOT)),"input_resolution":"960x704",
     "device":"CPU PyTorch/Ultralytics; no CUDA or TensorRT available in this workspace",
     "sample_times_s":SAMPLE_TIMES,"limitations":"Only 30 sampled frames from two stationary clips. CPU inference time is not comparable to Jetson TensorRT latency. Candidate-specific block-3 stereo repeatability is measured on the same frames, but there is no tape-measured range truth.","results":[]}

for scene,video in VIDEOS:
    cap=cv2.VideoCapture(str(ROOT/video))
    base=next(x for x in DETECTIONS["results"] if x["scene"]==scene)
    model_rows=[];base_rows=[];latencies=[];sample_indices=[]
    candidate_ranges=[[],[]];baseline_ranges=[[],[]]
    candidate_both_ranges=[[],[]];baseline_both_ranges=[[],[]]
    candidate_pair=[];baseline_pair=[];range_deltas=[[],[]]
    for t in SAMPLE_TIMES:
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
        ok,frame=cap.read()
        if not ok: continue
        started=time.perf_counter()
        result=model.predict(frame[:,:1600],imgsz=(704,960),conf=.02,iou=.15,
                             device="cpu",verbose=False,max_det=300)[0]
        latencies.append((time.perf_counter()-started)*1000)
        boxes=[]
        for xyxy,conf in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy()):
            x1,y1,x2,y2=map(float,xyxy)
            boxes.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"confidence":float(conf)})
        idx=min(len(base["per_frame"])-1,max(0,int(round(t*base["fps"]))))
        ref=base["per_frame"][idx]["detections"]
        model_rows.append(boxes);base_rows.append(ref);sample_indices.append(idx)
        left,right=frame[:,:1600],frame[:,1600:]
        left=cv2.resize(cv2.cvtColor(cv2.remap(left,map_l[0],map_l[1],cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
        right=cv2.resize(cv2.cvtColor(cv2.remap(right,map_r[0],map_r[1],cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
        disparity=matcher.compute(left,right).astype(np.float32)/16.0
        for rows,dest_ranges,dest_both in ((boxes,candidate_ranges,candidate_both_ranges),
                                           (ref,baseline_ranges,baseline_both_ranges)):
            hits,_=match(rows,.05)
            frame_points=[]
            for target,box in enumerate(hits):
                value=geometry(box,disparity) if box else None
                frame_points.append(value)
                if value:
                    dest_ranges[target].append(value["range_m"])
            if all(frame_points):
                separation=float(np.linalg.norm(frame_points[0]["xyz_m"]-frame_points[1]["xyz_m"]))
                if rows is boxes:candidate_pair.append(separation)
                else:baseline_pair.append(separation)
                for target in range(2):
                    dest_both[target].append(frame_points[target]["range_m"])
    cap.release()
    out["results"].append({"scene":scene,"sample_indices":sample_indices,"samples":len(model_rows),
      "cpu_latency_ms":{"median":round(statistics.median(latencies),2),
                         "p95":round(float(np.percentile(latencies,95)),2)},
      "stereo_block3_same_frames":{"candidate_960x704_ranges":[range_summary(x) for x in candidate_ranges],
        "baseline_1280x960_ranges":[range_summary(x) for x in baseline_ranges],
        "candidate_valid_pair_range_deltas_vs_baseline_m":[
            range_summary([abs(a-b) for a,b in zip(candidate_both_ranges[t],baseline_both_ranges[t])])
            for t in range(2)],
        "candidate_pair_spacing_m":range_summary(candidate_pair),
        "baseline_pair_spacing_m":range_summary(baseline_pair)},
      "thresholds":{str(th):{"960x704_pt":score(model_rows,th),
                              "1280x960_tensorrt_cached":score(base_rows,th)}
                    for th in THRESHOLDS}})
    print(scene,json.dumps(out["results"][-1]))

path=ROOT/"media/frc-fuel-960x704-checkpoint-evaluation.json"
path.write_text(json.dumps(out,indent=2)+"\n")
print("saved",path)
