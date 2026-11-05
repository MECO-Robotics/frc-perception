#!/usr/bin/env python3
"""Measure whether Hough-refined centers improve reconstructed FUEL pair geometry."""
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
CAL = ROOT / "media/calibration/stereo_calibration_live_order_candidate.yml"
fs = cv2.FileStorage(str(CAL), cv2.FILE_STORAGE_READ)
kl = fs.getNode("left_camera_matrix").mat()
dl = fs.getNode("left_distortion").mat()
kr = fs.getNode("right_camera_matrix").mat()
dr = fs.getNode("right_distortion").mat()
rotation = fs.getNode("rotation").mat()
translation = fs.getNode("translation").mat()
fs.release()
r1, r2, p1, p2, *_ = cv2.stereoRectify(kl, dl, kr, dr, (1600, 1200), rotation, translation,
                                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
fx, fy = float(p1[0, 0]), float(p1[1, 1])
cx, cy = float(p1[0, 2]), float(p1[1, 2])


def match(detections):
    hits = [None, None]
    for b in sorted((d for d in detections if d["confidence"] >= 0.05),
                    key=lambda d: d["confidence"], reverse=True):
        center = ((b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2)
        ds = [math.dist(center, q) for q in TARGETS]
        i = int(np.argmin(ds))
        if ds[i] <= 60 and hits[i] is None:
            hits[i] = b
    return hits


def xyz(center, z):
    q = cv2.undistortPoints(np.array([[[center[0], center[1]]]], np.float32), kl, dl,
                            R=r1, P=p1)[0, 0]
    return np.array(((q[0] - cx) * z / fx, (q[1] - cy) * z / fy, z), dtype=np.float64)


def stats(values):
    if not values:
        return None
    m = statistics.median(values)
    return {"n": len(values), "median": round(m, 5),
            "mad": round(statistics.median(abs(v - m) for v in values), 5),
            "p05": round(float(np.percentile(values, 5)), 5),
            "p95": round(float(np.percentile(values, 95)), 5)}


out = {
    "method": "Use the per-frame production box-median stereo ranges from the dense stereo evaluation, reconstruct camera-frame xyz with either the YOLO box center or a p25 Hough circle center, and compare pair distance / centroid against positions from the fixed manually labeled image centers at the same measured Z values.",
    "limitations": "The labeled pixel centers are manually fixed and the range values come from box-median SGBM, so this isolates center localization effects only. It does not establish absolute Z accuracy or test motion/occlusion. Hough runs after YOLO localization and uses box-center proximity to choose a circle.",
    "hough_param2": 25,
    "results": [],
}

for clip, stereo_clip in zip(DETECTIONS["results"], STEREO["results"]):
    cap = cv2.VideoCapture(str(ROOT / clip["video"]))
    per_method = {name: {"pair_distance": [], "pair_distance_ref_error": [], "centroid_error": [], "target_xyz_error": [[], []], "hough_valid_pair": 0}
                  for name in ("box_center", "hough_p25")}
    frames = []
    started = time.perf_counter()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = match(clip["per_frame"][idx]["detections"])
        ranges = stereo_clip["frames"][idx]["ranges_m"]
        centers = [None, None]
        gray = cv2.cvtColor(frame[:, :1600], cv2.COLOR_BGR2GRAY)
        for target, box in enumerate(boxes):
            if box is None:
                continue
            bx, by = (box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2
            short = min(box["x2"] - box["x1"], box["y2"] - box["y1"])
            x0, y0 = max(0, int(round(bx - short * .8))), max(0, int(round(by - short * .8)))
            x1, y1 = min(1600, int(round(bx + short * .8))), min(1200, int(round(by + short * .8)))
            crop = cv2.GaussianBlur(gray[y0:y1, x0:x1], (9, 9), 2)
            rmin, rmax = max(8, int(short * .35)), max(9, int(short * .65))
            circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, dp=1.2,
                                       minDist=max(10, int(short * .5)), param1=90,
                                       param2=25, minRadius=rmin, maxRadius=rmax) if crop.size else None
            if circles is not None:
                candidates = []
                for x, y, _r in circles[0]:
                    center = (float(x + x0), float(y + y0))
                    offset = math.dist(center, (bx, by))
                    if offset <= short * .30:
                        candidates.append((offset, center))
                if candidates:
                    centers[target] = min(candidates, key=lambda x: x[0])[1]
            if centers[target] is None:
                centers[target] = (bx, by)
        if all(boxes) and all(z is not None for z in ranges):
            ref = [xyz(TARGETS[t], ranges[t]) for t in range(2)]
            for name in per_method:
                est_centers = [((boxes[t]["x1"] + boxes[t]["x2"]) / 2,
                                (boxes[t]["y1"] + boxes[t]["y2"]) / 2) if name == "box_center" else centers[t]
                               for t in range(2)]
                points = [xyz(est_centers[t], ranges[t]) for t in range(2)]
                pref = np.mean(ref, axis=0)
                pest = np.mean(points, axis=0)
                distance = float(np.linalg.norm(points[0] - points[1]))
                ref_distance = float(np.linalg.norm(ref[0] - ref[1]))
                per_method[name]["pair_distance"].append(distance)
                per_method[name]["pair_distance_ref_error"].append(abs(distance - ref_distance))
                per_method[name]["centroid_error"].append(float(np.linalg.norm(pest - pref)))
                for t in range(2):
                    per_method[name]["target_xyz_error"][t].append(float(np.linalg.norm(points[t] - ref[t])))
            per_method["hough_p25"]["hough_valid_pair"] += int(all(c is not None for c in centers))
            frames.append(idx)
        idx += 1
    cap.release()
    methods = []
    for name, values in per_method.items():
        methods.append({"method": name, "valid_frames": len(values["pair_distance"]),
                        "hough_centers_both_valid_frames": values["hough_valid_pair"] if name == "hough_p25" else None,
                        "pair_distance_m": stats(values["pair_distance"]),
                        "pair_distance_absolute_error_vs_label_xy_m": stats(values["pair_distance_ref_error"]),
                        "centroid_error_xyz_m": stats(values["centroid_error"]),
                        "target_xyz_error_m": [stats(x) for x in values["target_xyz_error"]]})
    out["results"].append({"scene": clip["scene"], "frames": idx,
                           "valid_shared_frames": len(frames),
                           "elapsed_s": round(time.perf_counter() - started, 2),
                           "methods": methods})
    print(clip["scene"], json.dumps(methods))

path = ROOT / "media/frc-fuel-hough-center-3d-evaluation.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print("saved", path)
