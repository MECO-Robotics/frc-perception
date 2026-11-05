#!/usr/bin/env python3
"""Run real saved frames through the 960x704 Xavier TensorRT engine."""
import json
import math
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ENGINE=ROOT/"models/rebuilt_fuel_yolo26n_960x704_fp16.engine"
TRTEXEC=Path("/usr/src/tensorrt/bin/trtexec")
VIDEOS=(("earlier","media/frc-2026-live-two-fuel-60fps-cfr.mp4"),
        ("corrected-settings","media/frc-2026-fuel-camera-settings-fixed-60fps.mp4"))
COUNT=json.loads((ROOT/"media/frc-fuel-dense-count-nms015.json").read_text())
STEREO=json.loads((ROOT/"media/frc-fuel-dense-stereo-block3.json").read_text())
TIMES=(1.0,1.4,1.8,2.2,2.6,3.0,3.4,3.8,4.2,4.6,5.0,5.4,5.8,6.2,6.6)
TARGETS=((808.0,589.0),(902.0,591.0))
THRESHOLDS=(0.02,0.03,0.04,0.05,0.075,0.10,0.15,0.20,0.25,0.30,0.40)
SCALE=704/1200
OFFSET_X=10
BASELINE_M=0.0635

fs=cv2.FileStorage(str(ROOT/"media/calibration/stereo_calibration_live_order_candidate.yml"),cv2.FILE_STORAGE_READ)
kl=fs.getNode("left_camera_matrix").mat();dl=fs.getNode("left_distortion").mat()
kr=fs.getNode("right_camera_matrix").mat();dr=fs.getNode("right_distortion").mat()
rotation=fs.getNode("rotation").mat();translation=fs.getNode("translation").mat();fs.release()
r1,r2,p1,p2,*_=cv2.stereoRectify(kl,dl,kr,dr,(1600,1200),rotation,translation,
                                  flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
map_l=cv2.initUndistortRectifyMap(kl,dl,r1,p1,(1600,1200),cv2.CV_32FC1)
map_r=cv2.initUndistortRectifyMap(kr,dr,r2,p2,(1600,1200),cv2.CV_32FC1)
matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=3,
    P1=72,P2=288,uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def match(boxes,threshold):
    selected=[b for b in boxes if b["confidence"]>=threshold and
      max(b["x2"]-b["x1"],b["y2"]-b["y1"])/max(1,min(b["x2"]-b["x1"],b["y2"]-b["y1"]))<=2.0]
    selected.sort(key=lambda b:b["confidence"],reverse=True)
    hits=[None,None];extras=0
    for box in selected:
        c=((box["x1"]+box["x2"])/2,(box["y1"]+box["y2"])/2)
        d,i=min((math.dist(c,t),j) for j,t in enumerate(TARGETS))
        if d<=60 and hits[i] is None:hits[i]=box
        else:extras+=1
    return hits,extras


def score(rows,threshold):
    tp=fp=fn=paired=exact=0;errors=[]
    for boxes in rows:
        hits,extras=match(boxes,threshold);n=sum(x is not None for x in hits)
        tp+=n;fp+=extras;fn+=2-n;paired+=n==2;exact+=n==2 and extras==0
        errors.extend(math.dist(((b["x1"]+b["x2"])/2,(b["y1"]+b["y2"])/2),TARGETS[i])
          for i,b in enumerate(hits) if b)
    return {"tp":tp,"fp":fp,"fn":fn,"paired_frames":paired,"exact_two_frames":exact,
      "center_error_px":{"n":len(errors),"median":round(statistics.median(errors),3) if errors else None,
        "p95":round(float(np.percentile(errors,95)),3) if errors else None}}


def stereo_geometry(box,disparity):
    corners=np.array([[[box["x1"],box["y1"]]],[[box["x2"],box["y1"]]],
                      [[box["x2"],box["y2"]]],[[box["x1"],box["y2"]]]],np.float32)
    rect=cv2.undistortPoints(corners,kl,dl,R=r1,P=p1).reshape(-1,2)*.5
    x1,y1=np.floor(rect.min(axis=0)).astype(int);x2,y2=np.ceil(rect.max(axis=0)).astype(int)
    roi=disparity[max(0,y1):min(600,y2),max(0,x1):min(800,x2)]
    valid=roi[np.isfinite(roi)&(roi>1.5)&(roi<60)]
    if len(valid)<8:return None
    z=BASELINE_M*float(p1[0,0])/(2*float(np.median(valid)))
    center=np.array([[[ (box["x1"]+box["x2"])/2,(box["y1"]+box["y2"])/2 ]]],np.float32)
    u,v=cv2.undistortPoints(center,kl,dl,R=r1,P=p1)[0,0]
    xyz=np.asarray(((u-p1[0,2])*z/p1[0,0],(v-p1[1,2])*z/p1[1,1],z))
    return float(z),xyz


def summary(values):
    if not values:return None
    med=float(statistics.median(values))
    return {"n":len(values),"median_m":round(med,5),
            "mad_m":round(statistics.median(abs(x-med) for x in values),5),
            "p05_m":round(float(np.percentile(values,5)),5),"p95_m":round(float(np.percentile(values,95)),5)}


env=os.environ.copy()
env["LD_LIBRARY_PATH"]="/usr/local/cuda/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu:"+env.get("LD_LIBRARY_PATH","")
out={"method":"Preprocess 15 timestamps per saved stereo clip with the Scout letterbox geometry for a 960x704 engine, execute the Xavier FP16 TensorRT engine through trtexec on the GPU, decode the real output tensors, score counts/centers against fixed target centers, and compute same-frame block-3 SGBM range and pair spacing. Compare with cached 1280x960 TensorRT detections on the same frames.",
 "engine":str(ENGINE.relative_to(ROOT)),"input_resolution":"960x704","gpu":"Jetson Xavier, Compute Capability 7.2, TensorRT 8.5.2",
 "sample_times_s":TIMES,"limitations":"Thirty samples from two stationary recordings. Fixed target windows and no varied scenes, physical occlusion, or tape-measured range truth. trtexec starts a new process per input, so this evaluates real-frame outputs, not pipeline throughput; use the separate repeated-inference profile for engine throughput.","results":[]}

with tempfile.TemporaryDirectory(prefix="fuel960trt-") as tmp:
  for scene,video in VIDEOS:
    cap=cv2.VideoCapture(str(ROOT/video));clip=next(x for x in COUNT["results"] if x["scene"]==scene)
    model_rows=[];base_rows=[];baseline_ranges=[[],[]]
    candidate_stereo={str(th):{"ranges":[[],[]],"pairs":[]} for th in (0.03,0.05)}
    baseline_pairs=[];indices=[];frame_details=[]
    for sample_i,t in enumerate(TIMES):
      cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read()
      if not ok:continue
      left,right=frame[:,:1600],frame[:,1600:]
      canvas=np.full((704,960,3),114,np.uint8)
      resized=cv2.resize(left,(939,704),interpolation=cv2.INTER_LINEAR)
      canvas[:,OFFSET_X:OFFSET_X+939]=resized
      tensor=np.transpose(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).astype(np.float32)/255.,(2,0,1))[None].copy()
      inp=Path(tmp)/f"{scene}-{sample_i}.bin";raw=Path(tmp)/f"{scene}-{sample_i}.json";log=Path(tmp)/f"{scene}-{sample_i}.log"
      tensor.tofile(inp)
      cmd=[str(TRTEXEC),f"--loadEngine={ENGINE}",f"--loadInputs=images:{inp}",
           "--iterations=1","--duration=0.01","--avgRuns=1",f"--exportOutput={raw}"]
      with log.open("w") as handle:
        completed=subprocess.run(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,timeout=90)
      if completed.returncode!=0:raise RuntimeError(f"trtexec failed for {scene} {t}s: {log.read_text()[-2000:]}")
      result=json.loads(raw.read_text())[0]["values"]
      boxes=[]
      for x1,y1,x2,y2,confidence,class_id in np.asarray(result,dtype=float).reshape(-1,6):
        if class_id!=0:continue
        boxes.append({"x1":(x1-OFFSET_X)/SCALE,"y1":y1/SCALE,
          "x2":(x2-OFFSET_X)/SCALE,"y2":y2/SCALE,"confidence":confidence})
      idx=min(len(clip["per_frame"])-1,max(0,int(round(t*clip["fps"]))))
      baseline_boxes=clip["per_frame"][idx]["detections"]
      model_rows.append(boxes);base_rows.append(baseline_boxes);indices.append(idx)
      frame_details.append({"time_s":t,"frame_index":idx,
        "candidate_boxes_ge_0p01":[b for b in boxes if b["confidence"]>=.01],
        "threshold_metrics":{str(th):score([boxes],th) for th in THRESHOLDS}})
      rect_l=cv2.resize(cv2.cvtColor(cv2.remap(left,map_l[0],map_l[1],cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
      rect_r=cv2.resize(cv2.cvtColor(cv2.remap(right,map_r[0],map_r[1],cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
      disparity=matcher.compute(rect_l,rect_r).astype(np.float32)/16
      for th,rec in candidate_stereo.items():
        hits,_=match(boxes,float(th))
        points=[]
        for target,box in enumerate(hits):
          value=stereo_geometry(box,disparity) if box else None
          points.append(value)
          if value:rec["ranges"][target].append(value[0])
        if all(points):rec["pairs"].append(float(np.linalg.norm(points[0][1]-points[1][1])))
      hits,_=match(baseline_boxes,.05);points=[]
      for target,box in enumerate(hits):
        value=stereo_geometry(box,disparity) if box else None;points.append(value)
        if value:baseline_ranges[target].append(value[0])
      if all(points):baseline_pairs.append(float(np.linalg.norm(points[0][1]-points[1][1])))
      print(scene, t, 'gpu boxes',len(boxes),'sample',len(model_rows))
    cap.release()
    out["results"].append({"scene":scene,"sample_indices":indices,"samples":len(model_rows),
      "thresholds":{str(th):{"960x704_tensorrt_gpu":score(model_rows,th),
        "1280x960_tensorrt_cached":score(base_rows,th)} for th in THRESHOLDS},
      "per_frame":frame_details,
      "stereo_block3_same_frames":{"candidate_by_confidence":{
        th:{"ranges":[summary(v) for v in rec["ranges"]],"pair_spacing_m":summary(rec["pairs"])}
        for th,rec in candidate_stereo.items()},
        "baseline_ranges":[summary(v) for v in baseline_ranges],
        "baseline_pair_spacing_m":summary(baseline_pairs)}})

path=ROOT/"media/frc-fuel-960x704-tensorrt-evaluation.json"
path.write_text(json.dumps(out,indent=2)+"\n")
print("saved",path)
