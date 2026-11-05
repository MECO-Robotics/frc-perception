#!/usr/bin/env python3
"""Check whether Hough-refined XY points change the two-FUEL merge-radius requirement."""
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS = json.loads((ROOT / "media/frc-fuel-dense-count-evaluation.json").read_text())
STEREO = json.loads((ROOT / "media/frc-fuel-dense-stereo-evaluation.json").read_text())
TARGETS = ((808.0, 589.0), (902.0, 591.0))
RADIUS_M = (.22, .23, .24, .245, .25, .255, .26)
fs = cv2.FileStorage(str(ROOT / "media/calibration/stereo_calibration_live_order_candidate.yml"), cv2.FILE_STORAGE_READ)
kl, dl = fs.getNode("left_camera_matrix").mat(), fs.getNode("left_distortion").mat()
kr, dr = fs.getNode("right_camera_matrix").mat(), fs.getNode("right_distortion").mat()
rotation, translation = fs.getNode("rotation").mat(), fs.getNode("translation").mat()
fs.release()
r1, r2, p1, p2, *_ = cv2.stereoRectify(kl, dl, kr, dr, (1600, 1200), rotation, translation,
                                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
fx, fy, cx, cy = float(p1[0, 0]), float(p1[1, 1]), float(p1[0, 2]), float(p1[1, 2])


def matched(detections):
    out = [None, None]
    for b in sorted((b for b in detections if b["confidence"] >= .05), key=lambda b: -b["confidence"]):
        center = ((b["x1"]+b["x2"])/2, (b["y1"]+b["y2"])/2)
        ds = [math.dist(center, p) for p in TARGETS]
        i = int(np.argmin(ds))
        if ds[i] <= 60 and out[i] is None:
            out[i] = b
    return out


def hough(gray, box):
    x1, y1, x2, y2 = [int(box[k]) for k in ("x1", "y1", "x2", "y2")]
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(1600, x2), min(1200, y2)
    bx, by = (x1+x2)/2, (y1+y2)/2
    short = min(x2-x1, y2-y1)
    x0, y0 = max(0, int(round(bx-short*.8))), max(0, int(round(by-short*.8)))
    xe, ye = min(1600, int(round(bx+short*.8))), min(1200, int(round(by+short*.8)))
    crop = cv2.GaussianBlur(gray[y0:ye, x0:xe], (9, 9), 2)
    if not crop.size:
        return None
    circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=max(10, int(short*.5)), param1=90, param2=25,
                               minRadius=max(8, int(short*.35)), maxRadius=max(9, int(short*.65)))
    if circles is None:
        return None
    candidates = []
    for x, y, _r in circles[0]:
        q = (float(x+x0), float(y+y0))
        off = math.dist(q, (bx, by))
        if off <= short*.30:
            candidates.append((off, q))
    return min(candidates, key=lambda a: a[0])[1] if candidates else None


def point(center, z):
    q = cv2.undistortPoints(np.array([[[center[0], center[1]]]], np.float32), kl, dl,
                            R=r1, P=p1)[0, 0]
    return np.array(((q[0]-cx)*z/fx, (q[1]-cy)*z/fy, z), np.float64)


def summary(vals):
    if not vals:
        return {"n": 0}
    m = statistics.median(vals)
    return {"n": len(vals), "median_m": round(m, 5),
            "mad_m": round(statistics.median(abs(x-m) for x in vals), 5),
            "p95_m": round(float(np.percentile(vals, 95)), 5),
            "max_m": round(max(vals), 5)}


out = {
    "method": "Use per-frame SGBM Z from the dense stereo sweep and reconstruct pair points with either box centers or Hough param2=25 left-view centers. Sweep Euclidean pair merge cutoffs on all frames where both points are valid.",
    "limitations": "Only two adjacent balls are present. The labels support evaluating their pair distance and merge recall, but cannot measure false merges of separate clusters. SGBM Z has no external metric ground truth.",
    "merge_radii_m": RADIUS_M,
    "results": [],
}

for ci, clip in enumerate(DETECTIONS["results"]):
    cap = cv2.VideoCapture(str(ROOT / clip["video"]))
    distances = {"box_center": [], "hough_center": []}
    hough_valid = 0
    count = 0
    started = time.perf_counter()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = matched(clip["per_frame"][idx]["detections"])
        ranges = STEREO["results"][ci]["frames"][idx]["ranges_m"]
        if all(boxes) and all(z is not None for z in ranges):
            centers = [((b["x1"]+b["x2"])/2, (b["y1"]+b["y2"])/2) for b in boxes]
            distances["box_center"].append(float(np.linalg.norm(point(centers[0], ranges[0])-point(centers[1], ranges[1]))))
            gray = cv2.cvtColor(frame[:, :1600], cv2.COLOR_BGR2GRAY)
            hc = [hough(gray, b) for b in boxes]
            if all(c is not None for c in hc):
                hough_valid += 1
                distances["hough_center"].append(float(np.linalg.norm(point(hc[0], ranges[0])-point(hc[1], ranges[1]))))
            count += 1
        idx += 1
    cap.release()
    methods = []
    for method, vals in distances.items():
        methods.append({"method": method, "pair_separation_m": summary(vals),
                        "merge_counts_by_radius": [{"radius_m": r, "merged_frames": sum(d <= r for d in vals),
                                                    "valid_frames": len(vals),
                                                    "merge_fraction": round(sum(d <= r for d in vals)/len(vals), 5) if vals else None}
                                                   for r in RADIUS_M]})
    entry = {"scene": clip["scene"], "decoded_frames": idx,
             "both_box_and_depth_valid_frames": count,
             "hough_valid_pair_frames": hough_valid,
             "hough_pair_coverage": round(hough_valid/count, 5) if count else None,
             "elapsed_s": round(time.perf_counter()-started, 2), "methods": methods}
    out["results"].append(entry)
    print(clip["scene"], json.dumps(entry))

path = ROOT / "media/frc-fuel-hough-cluster-radius-sweep.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print("saved", path)
