#!/usr/bin/env python3
"""Compare StereoBM block sizes with active SGBM on sampled FUEL stereo frames."""
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COUNTS = json.loads((ROOT / "media/frc-fuel-dense-count-evaluation.json").read_text())
SGBM = json.loads((ROOT / "media/frc-fuel-dense-stereo-evaluation.json").read_text())
TARGETS = ((808, 589), (902, 591))
STEP, PAD, BASE = 10, 24, .0635
BLOCKS = (9, 15, 21, 31)
fs = cv2.FileStorage(str(ROOT / "media/calibration/stereo_calibration_live_order_candidate.yml"), cv2.FILE_STORAGE_READ)
kl, dl = fs.getNode("left_camera_matrix").mat(), fs.getNode("left_distortion").mat()
kr, dr = fs.getNode("right_camera_matrix").mat(), fs.getNode("right_distortion").mat()
rotation, translation = fs.getNode("rotation").mat(), fs.getNode("translation").mat()
fs.release()
r1, r2, p1, p2, *_ = cv2.stereoRectify(kl, dl, kr, dr, (1600, 1200), rotation, translation,
                                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
map_l = cv2.initUndistortRectifyMap(kl, dl, r1, p1, (1600, 1200), cv2.CV_32FC1)
map_r = cv2.initUndistortRectifyMap(kr, dr, r2, p2, (1600, 1200), cv2.CV_32FC1)
focal = float(p1[0, 0])
MATCHERS = {}
for block in BLOCKS:
    matcher = cv2.StereoBM_create(numDisparities=128, blockSize=block)
    matcher.setMinDisparity(0)
    matcher.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
    matcher.setPreFilterSize(9)
    matcher.setPreFilterCap(31)
    matcher.setTextureThreshold(10)
    matcher.setUniquenessRatio(15)
    matcher.setSpeckleWindowSize(50)
    matcher.setSpeckleRange(2)
    matcher.setDisp12MaxDiff(1)
    MATCHERS[block] = matcher


def matched(dets):
    hits = [None, None]
    for b in sorted((b for b in dets if b["confidence"] >= .05), key=lambda b: -b["confidence"]):
        x, y = (b["x1"]+b["x2"])/2, (b["y1"]+b["y2"])/2
        ds = [math.hypot(x-a, y-c) for a,c in TARGETS]
        i = int(np.argmin(ds))
        if ds[i] <= 60 and hits[i] is None:
            hits[i] = b
    return hits


def rect_box(b):
    corners=np.array([[[b["x1"],b["y1"]]],[[b["x2"],b["y1"]]],[[b["x2"],b["y2"]]],[[b["x1"],b["y2"]]]],np.float32)
    q=cv2.undistortPoints(corners,kl,dl,R=r1,P=p1).reshape(-1,2)*.5
    return (max(0,int(np.floor(q[:,0].min()))),max(0,int(np.floor(q[:,1].min()))),min(800,int(np.ceil(q[:,0].max()))),min(600,int(np.ceil(q[:,1].max()))))


def unproject(box,z):
    center=np.array([[[float(box["x1"]+box["x2"])/2,float(box["y1"]+box["y2"])/2]]],np.float32)
    q=cv2.undistortPoints(center,kl,dl,R=r1,P=p1)[0,0]
    return ((q[0]-p1[0,2])*z/p1[0,0],(q[1]-p1[1,2])*z/p1[1,1],z)


def stats(values):
    vals=[x for x in values if x is not None and math.isfinite(x)]
    if not vals:return {"n":0}
    m=statistics.median(vals)
    return {"n":len(vals),"median":round(m,5),"mad":round(statistics.median(abs(x-m) for x in vals),5),
            "p05":round(float(np.percentile(vals,5)),5),"p95":round(float(np.percentile(vals,95)),5)}


out={"method":"Every tenth frame, compare active SGBM 3WAY and StereoBM block sizes 9/15/21/31 on the active 24 px full-width vertical band. Same 800x600 rectification, detector boxes, 128 disparity range, valid disparity interval (3,120), baseline and focal length. StereoBM uses XSOBEL cap31, texture10, uniqueness15, speckle50/2, disp12MaxDiff1.",
     "limitations":"Two static target balls and one range pose; no measured ground truth. Matcher comparison measures valid coverage/repeatability, not absolute accuracy or dynamic occlusion response.",
     "sample_step_frames":STEP,"vertical_roi_pad_half_px":PAD,"stereobm_blocks":list(BLOCKS),"results":[]}

for ci,clip in enumerate(COUNTS["results"]):
    cap=cv2.VideoCapture(str(ROOT/clip["video"]))
    records={name:{"ranges":[[],[]],"errors_vs_sgbm":[[],[]],"spacing":[],"times":[]} for name in ["SGBM"]+[f"StereoBM_b{b}" for b in BLOCKS]}
    frame_idx=0;start=time.perf_counter()
    while True:
        ok,frame=cap.read()
        if not ok:break
        if frame_idx%STEP==0:
            boxes=matched(clip["per_frame"][frame_idx]["detections"])
            if all(boxes):
                rects=[rect_box(b) for b in boxes]
                y0=max(0,min(b[1] for b in rects)-PAD);y1=min(600,max(b[3] for b in rects)+PAD)
                left,right=frame[:,:1600],frame[:,1600:]
                gl=cv2.resize(cv2.cvtColor(cv2.remap(left,*map_l,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
                gr=cv2.resize(cv2.cvtColor(cv2.remap(right,*map_r,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
                matchers={"SGBM":cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=5,P1=200,P2=800,uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY),**{f"StereoBM_b{b}":m for b,m in MATCHERS.items()}}
                sgbm_z=None
                for name,matcher in matchers.items():
                    t0=time.perf_counter();disp=matcher.compute(gl[y0:y1],gr[y0:y1]).astype(np.float32)/16
                    records[name]["times"].append((time.perf_counter()-t0)*1000)
                    zs=[]
                    for target,(x1,yy1,x2,yy2) in enumerate(rects):
                        roi=disp[max(0,yy1-y0):max(0,yy2-y0),x1:x2]
                        vals=roi[np.isfinite(roi)&(roi>3)&(roi<120)]
                        z=BASE*focal/(2*float(np.median(vals))) if vals.size>=8 else None
                        records[name]["ranges"][target].append(z);zs.append(z)
                    if name=="SGBM":
                        sgbm_z=zs
                    else:
                        for target,z in enumerate(zs):
                            ref=sgbm_z[target] if sgbm_z else None
                            records[name]["errors_vs_sgbm"][target].append(abs(z-ref) if z and ref else None)
                    if all(z is not None for z in zs):
                        records[name]["spacing"].append(math.dist(unproject(boxes[0],zs[0]),unproject(boxes[1],zs[1])))
        frame_idx+=1
    cap.release()
    configs=[]
    for name,data in records.items():
        configs.append({"matcher":name,"ranges_m_by_target":[stats(x) for x in data["ranges"]],
                        "absolute_range_difference_vs_sgbm_m_by_target":[stats(x) for x in data["errors_vs_sgbm"]] if name!="SGBM" else None,
                        "pair_spacing_m":stats(data["spacing"]),
                        "matcher_ms_median":round(statistics.median(data["times"]),2) if data["times"] else None,
                        "matcher_ms_p95":round(float(np.percentile(data["times"],95)),2) if data["times"] else None})
    entry={"scene":clip["scene"],"frames_decoded":frame_idx,"sampled_valid_frames":len(records["SGBM"]["spacing"]),
           "elapsed_s":round(time.perf_counter()-start,2),"configs":configs}
    out["results"].append(entry);print(clip["scene"],json.dumps(configs))

path=ROOT/"media/frc-fuel-stereobm-sweep.json";path.write_text(json.dumps(out,indent=2)+"\n")
print("saved",path)
