#!/usr/bin/env python3
"""Sweep SGBM speckle-filter settings on sampled saved FUEL stereo frames."""
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
STEP = 10
PAD = 24
BASE = .0635
CONFIGS = ((0, 0), (25, 1), (50, 1), (50, 2), (50, 4), (100, 1), (100, 2), (200, 2), (400, 2))
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
MATCHERS = {cfg: cv2.StereoSGBM_create(minDisparity=0, numDisparities=128,
             blockSize=5, P1=200, P2=800, uniquenessRatio=8,
             speckleWindowSize=cfg[0], speckleRange=cfg[1],
             mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY) for cfg in CONFIGS}


def match(detections):
    hits = [None, None]
    for b in sorted((b for b in detections if b["confidence"] >= .05), key=lambda b: -b["confidence"]):
        cx, cy = (b["x1"]+b["x2"])/2, (b["y1"]+b["y2"])/2
        distances = [math.hypot(cx-x, cy-y) for x, y in TARGETS]
        i = int(np.argmin(distances))
        if distances[i] <= 60 and hits[i] is None:
            hits[i] = b
    return hits


def rect_box(b):
    corners = np.array([[[b["x1"], b["y1"]]], [[b["x2"], b["y1"]]],
                        [[b["x2"], b["y2"]]], [[b["x1"], b["y2"]]]], np.float32)
    q = cv2.undistortPoints(corners, kl, dl, R=r1, P=p1).reshape(-1, 2) * .5
    return (max(0, int(np.floor(q[:, 0].min()))), max(0, int(np.floor(q[:, 1].min()))),
            min(800, int(np.ceil(q[:, 0].max()))), min(600, int(np.ceil(q[:, 1].max()))))


def depth(disp, box, y0):
    x1, y1, x2, y2 = box
    roi = disp[max(0, y1-y0):max(0, y2-y0), x1:x2]
    valid = roi[np.isfinite(roi) & (roi > 3) & (roi < 120)]
    if valid.size < 8:
        return None, int(valid.size)
    med = float(np.median(valid))
    return BASE*focal/(2*med), int(valid.size)


def summary(values):
    values = [x for x in values if x is not None]
    if not values:
        return {"n": 0}
    med = statistics.median(values)
    return {"n": len(values), "median_m": round(med, 5),
            "mad_m": round(statistics.median(abs(x-med) for x in values), 5),
            "p05_m": round(float(np.percentile(values, 5)), 5),
            "p95_m": round(float(np.percentile(values, 95)), 5)}


out = {
    "method": "Every tenth synchronized frame, reuse confidence>=0.05 FRC9550 boxes, rectify/downsample to 800x600, compute SGBM only on the active 24 px full-width vertical band, and compare speckleWindowSize/speckleRange pairs with block5/P1=200/P2=800/uniqueness8 held fixed.",
    "limitations": "Stationary two-ball clips, sparse sample, no measured depth truth. Results measure valid depth and repeatability, not absolute accuracy or moving-scene behavior.",
    "sample_step_frames": STEP,
    "vertical_roi_pad_half_px": PAD,
    "configs_window_range": [{"window": w, "range": r} for w, r in CONFIGS],
    "results": [],
}

for clip in DATA["results"]:
    cap = cv2.VideoCapture(str(ROOT / clip["video"]))
    records = {cfg: {"ranges": [[], []], "pixels": [[], []], "pair_spacing": [], "times_ms": []}
               for cfg in CONFIGS}
    frame_index = 0
    started = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % STEP == 0:
            boxes = match(clip["per_frame"][frame_index]["detections"])
            if all(boxes):
                rects = [rect_box(b) for b in boxes]
                y0 = max(0, min(b[1] for b in rects)-PAD)
                y1 = min(600, max(b[3] for b in rects)+PAD)
                left, right = frame[:, :1600], frame[:, 1600:]
                gl = cv2.resize(cv2.cvtColor(cv2.remap(left, *map_l, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY), (800, 600), interpolation=cv2.INTER_AREA)
                gr = cv2.resize(cv2.cvtColor(cv2.remap(right, *map_r, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY), (800, 600), interpolation=cv2.INTER_AREA)
                for cfg, matcher in MATCHERS.items():
                    t0 = time.perf_counter()
                    disp = matcher.compute(gl[y0:y1], gr[y0:y1]).astype(np.float32)/16
                    records[cfg]["times_ms"].append((time.perf_counter()-t0)*1000)
                    z = []
                    for target, box in enumerate(rects):
                        value, pixels = depth(disp, box, y0)
                        records[cfg]["ranges"][target].append(value)
                        records[cfg]["pixels"][target].append(pixels)
                        z.append(value)
                    if all(x is not None for x in z):
                        # Compute full camera-frame distance using box centers and this matcher Z.
                        points = []
                        for box, zval in zip(boxes, z):
                            center = np.array([[[float(box["x1"]+box["x2"])/2,
                                                 float(box["y1"]+box["y2"])/2]]], np.float32)
                            q = cv2.undistortPoints(center, kl, dl, R=r1, P=p1)[0, 0]
                            points.append(((q[0]-p1[0,2])*zval/p1[0,0],
                                           (q[1]-p1[1,2])*zval/p1[1,1],zval))
                        records[cfg]["pair_spacing"].append(math.dist(*points))
        frame_index += 1
    cap.release()
    result_cfgs = []
    for cfg in CONFIGS:
        rec = records[cfg]
        result_cfgs.append({"window": cfg[0], "range": cfg[1],
                            "ranges_m": [summary(v) for v in rec["ranges"]],
                            "pair_spacing_m": summary(rec["pair_spacing"]),
                            "median_valid_pixels": [round(statistics.median(v), 1) if v else None for v in rec["pixels"]],
                            "matcher_ms_median": round(statistics.median(rec["times_ms"]), 2) if rec["times_ms"] else None,
                            "matcher_ms_p95": round(float(np.percentile(rec["times_ms"],95)), 2) if rec["times_ms"] else None})
    result_cfgs.sort(key=lambda x: (x["window"], x["range"]))
    entry = {"scene": clip["scene"], "frames_decoded": frame_index,
             "valid_sampled_frames": len(records[CONFIGS[0]]["pair_spacing"]),
             "elapsed_s": round(time.perf_counter()-started, 2), "configs": result_cfgs}
    out["results"].append(entry)
    print(clip["scene"], json.dumps(result_cfgs))

path = ROOT / "media/frc-fuel-sgbm-speckle-sweep.json"
path.write_text(json.dumps(out, indent=2)+"\n")
print("saved", path)
