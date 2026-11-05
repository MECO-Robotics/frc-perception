#!/usr/bin/env python3
"""Compare OpenCV SGBM path-aggregation modes on sampled saved FUEL stereo frames."""
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "media/frc-fuel-dense-count-evaluation.json").read_text())
TARGETS = ((808, 589), (902, 591))
STEP, PAD, BASE = 5, 24, .0635
MODES = {"SGBM": cv2.STEREO_SGBM_MODE_SGBM,
         "HH": cv2.STEREO_SGBM_MODE_HH,
         "3WAY": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
         "HH4": cv2.STEREO_SGBM_MODE_HH4}
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
MATCHERS = {name: cv2.StereoSGBM_create(minDisparity=0, numDisparities=128,
           blockSize=5, P1=200, P2=800, uniquenessRatio=8,
           speckleWindowSize=50, speckleRange=2, mode=mode)
            for name, mode in MODES.items()}


def matched(dets):
    hits = [None, None]
    for b in sorted((b for b in dets if b["confidence"] >= .05), key=lambda b: -b["confidence"]):
        x, y = (b["x1"]+b["x2"])/2, (b["y1"]+b["y2"])/2
        d = [math.hypot(x-a, y-c) for a, c in TARGETS]
        i = int(np.argmin(d))
        if d[i] <= 60 and hits[i] is None:
            hits[i] = b
    return hits


def rectbox(b):
    corners = np.array([[[b["x1"],b["y1"]]],[[b["x2"],b["y1"]]],[[b["x2"],b["y2"]]],[[b["x1"],b["y2"]]]], np.float32)
    q = cv2.undistortPoints(corners, kl, dl, R=r1, P=p1).reshape(-1,2)*.5
    return (max(0,int(np.floor(q[:,0].min()))), max(0,int(np.floor(q[:,1].min()))),
            min(800,int(np.ceil(q[:,0].max()))), min(600,int(np.ceil(q[:,1].max()))))


def unproject(box, z):
    center = np.array([[[float(box["x1"]+box["x2"])/2,float(box["y1"]+box["y2"])/2]]],np.float32)
    q = cv2.undistortPoints(center,kl,dl,R=r1,P=p1)[0,0]
    return ((q[0]-p1[0,2])*z/p1[0,0],(q[1]-p1[1,2])*z/p1[1,1],z)


def summary(values):
    v=[x for x in values if x is not None]
    if not v:return {"n":0}
    m=statistics.median(v)
    return {"n":len(v),"median":round(m,5),"mad":round(statistics.median(abs(x-m) for x in v),5),
            "p05":round(float(np.percentile(v,5)),5),"p95":round(float(np.percentile(v,95)),5)}


out={"method":"Every fifth frame, reuse confidence>=0.05 FRC9550 boxes. Compute rectified 800x600 SGBM only on active full-width vertical band (24 px pad), comparing OpenCV modes SGBM/HH/3WAY/HH4 with block5, P1=200, P2=800, uniqueness8, speckle50/2 and 128 disparities held fixed.",
     "limitations":"Two stationary balls at one pose; no measured distance truth or motion. Compare valid samples, repeatability and matcher compute only.",
     "sample_step_frames":STEP,"vertical_roi_pad_half_px":PAD,"modes":{k:int(v) for k,v in MODES.items()},"results":[]}

for clip in DATA["results"]:
    cap=cv2.VideoCapture(str(ROOT/clip["video"]))
    acc={m:{"ranges":[[],[]],"spacing":[],"times":[]} for m in MODES}
    frame_index=0; started=time.perf_counter()
    while True:
        ok,frame=cap.read()
        if not ok:break
        if frame_index%STEP==0:
            boxes=matched(clip["per_frame"][frame_index]["detections"])
            if all(boxes):
                rects=[rectbox(b) for b in boxes]
                y0=max(0,min(b[1] for b in rects)-PAD); y1=min(600,max(b[3] for b in rects)+PAD)
                left,right=frame[:,:1600],frame[:,1600:]
                gl=cv2.resize(cv2.cvtColor(cv2.remap(left,*map_l,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
                gr=cv2.resize(cv2.cvtColor(cv2.remap(right,*map_r,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY),(800,600),interpolation=cv2.INTER_AREA)
                for mode,matcher in MATCHERS.items():
                    t0=time.perf_counter();disp=matcher.compute(gl[y0:y1],gr[y0:y1]).astype(np.float32)/16
                    acc[mode]["times"].append((time.perf_counter()-t0)*1000)
                    zs=[]
                    for i,(x1,yy1,x2,yy2) in enumerate(rects):
                        roi=disp[max(0,yy1-y0):max(0,yy2-y0),x1:x2]
                        valid=roi[np.isfinite(roi)&(roi>3)&(roi<120)]
                        z=BASE*focal/(2*float(np.median(valid))) if valid.size>=8 else None
                        acc[mode]["ranges"][i].append(z);zs.append(z)
                    if all(z is not None for z in zs):
                        acc[mode]["spacing"].append(math.dist(unproject(boxes[0],zs[0]),unproject(boxes[1],zs[1])))
        frame_index+=1
    cap.release()
    summaries=[]
    for mode,data in acc.items():
        summaries.append({"mode":mode,"range_m_by_target":[summary(x) for x in data["ranges"]],
                          "pair_spacing_m":summary(data["spacing"]),
                          "matcher_ms_median":round(statistics.median(data["times"]),2) if data["times"] else None,
                          "matcher_ms_p95":round(float(np.percentile(data["times"],95)),2) if data["times"] else None})
    entry={"scene":clip["scene"],"frames_decoded":frame_index,
           "valid_pair_samples":max((x["pair_spacing_m"]["n"] for x in summaries),default=0),
           "elapsed_s":round(time.perf_counter()-started,2),"configs":summaries}
    out["results"].append(entry);print(clip["scene"],json.dumps(summaries))

path=ROOT/"media/frc-fuel-sgbm-mode-sweep.json";path.write_text(json.dumps(out,indent=2)+"\n")
print("saved",path)
