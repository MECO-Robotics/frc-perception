#!/usr/bin/env python3
"""Compare HSV-mask centroids with box and Hough centers on saved FUEL clips."""
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS = json.loads((ROOT / "media/frc-fuel-dense-count-nms015.json").read_text())
STEREO = json.loads((ROOT / "media/frc-fuel-dense-stereo-block3.json").read_text())
TARGETS = ((808.0, 589.0), (902.0, 591.0))
CONFIGS = (
    {"name": "h10-40_s40_v40", "h": (10, 40), "s": 40, "v": 40},
    {"name": "h10-40_s70_v40", "h": (10, 40), "s": 70, "v": 40},
    {"name": "h10-40_s100_v40", "h": (10, 40), "s": 100, "v": 40},
    {"name": "h10-40_s70_v80", "h": (10, 40), "s": 70, "v": 80},
    {"name": "h15-35_s70_v40", "h": (15, 35), "s": 70, "v": 40},
    {"name": "h8-45_s70_v40", "h": (8, 45), "s": 70, "v": 40},
)

fs = cv2.FileStorage(str(ROOT / "media/calibration/stereo_calibration_live_order_candidate.yml"),
                     cv2.FILE_STORAGE_READ)
kl, dl = fs.getNode("left_camera_matrix").mat(), fs.getNode("left_distortion").mat()
kr, dr = fs.getNode("right_camera_matrix").mat(), fs.getNode("right_distortion").mat()
rotation, translation = fs.getNode("rotation").mat(), fs.getNode("translation").mat()
fs.release()
r1, _, p1, _, *_ = cv2.stereoRectify(kl, dl, kr, dr, (1600, 1200), rotation, translation,
                                      flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
fx, fy, cx, cy = float(p1[0, 0]), float(p1[1, 1]), float(p1[0, 2]), float(p1[1, 2])


def match(detections):
    hits = [None, None]
    for box in sorted((b for b in detections if b["confidence"] >= .05),
                      key=lambda b: b["confidence"], reverse=True):
        center = ((box["x1"]+box["x2"])/2, (box["y1"]+box["y2"])/2)
        distance, target = min((math.dist(center, t), i) for i, t in enumerate(TARGETS))
        if distance <= 60 and hits[target] is None:
            hits[target] = box
    return hits


def summary(values):
    if not values:
        return None
    med = statistics.median(values)
    return {"n": len(values), "median": round(med, 5),
            "p95": round(float(np.percentile(values, 95)), 5),
            "mad": round(statistics.median(abs(v-med) for v in values), 5)}


def point(center, z):
    q = cv2.undistortPoints(np.array([[[center[0], center[1]]]], np.float32),
                            kl, dl, R=r1, P=p1)[0, 0]
    return np.asarray(((q[0]-cx)*z/fx, (q[1]-cy)*z/fy, z), dtype=float)


def refine(hsv, box, cfg):
    bx, by = (box["x1"]+box["x2"])/2, (box["y1"]+box["y2"])/2
    short = max(1.0, min(box["x2"]-box["x1"], box["y2"]-box["y1"]))
    radius = int(round(short*.65))
    x0, y0 = max(0, int(round(bx-radius))), max(0, int(round(by-radius)))
    x1, y1 = min(1600, int(round(bx+radius+1))), min(1200, int(round(by+radius+1)))
    crop = hsv[y0:y1, x0:x1]
    mask = cv2.inRange(crop, (cfg["h"][0], cfg["s"], cfg["v"]),
                       (cfg["h"][1], 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    choices = []
    for label in range(1, n):
        _, _, w, h, area = map(int, stats[label])
        if area < 30 or area > .65*mask.size or min(w, h) < 3 or max(w, h)/min(w, h) > 2.5:
            continue
        c = (float(centroids[label][0]+x0), float(centroids[label][1]+y0))
        offset = math.dist(c, (bx, by))
        if offset <= short*.30:
            choices.append((offset, c, area))
    if not choices:
        return (bx, by), False
    _, center, _ = min(choices, key=lambda item: item[0])
    return center, True


out = {"method": "Within each cached detector box, threshold yellow HSV pixels in a centered crop, apply 3x3 opening, select the connected component nearest the box center under a 0.30-short-side offset gate, and use its centroid; fall back to the box center when no component qualifies. Compare with raw boxes and the previously evaluated 75% Hough blend. Stereo Z is fixed to cached block-3 output.",
       "thresholds": CONFIGS,
       "limitations": "Two stationary clips with manually fixed center labels and no measured range truth. HSV centroid quality may change with lighting, ball markings, occlusion, or backgrounds. This only refines centers inside detector boxes; it does not test segmentation-only object discovery or detector count recall.",
       "results": []}

for clip, stereo_clip in zip(DETECTIONS["results"], STEREO["results"]):
    cap = cv2.VideoCapture(str(ROOT / clip["video"]))
    names = ["box_raw", "hsv_10_40_70_40", "hsv_10_40_100_40", "hsv_10_40_70_80",
             "hsv_15_35_70_40", "hsv_8_45_70_40", "hough075"]
    data = {name: {"center_errors_px": [], "target_xyz": [[], []],
                   "pair_spacing_m": [], "valid_pairs": 0} for name in names}
    fallback = {cfg["name"]: 0 for cfg in CONFIGS}
    valid_refines = {cfg["name"]: 0 for cfg in CONFIGS}
    elapsed = {cfg["name"]: 0.0 for cfg in CONFIGS}
    started = time.perf_counter()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = match(clip["per_frame"][frame_idx]["detections"])
        ranges = stereo_clip["frames"][frame_idx]["ranges_m"]
        if all(boxes) and all(z is not None for z in ranges):
            hsv = cv2.cvtColor(frame[:, :1600], cv2.COLOR_BGR2HSV)
            centers = {name: [] for name in names}
            for target, box in enumerate(boxes):
                bx, by = (box["x1"]+box["x2"])/2, (box["y1"]+box["y2"])/2
                centers["box_raw"].append((bx, by))
                for cfg in CONFIGS:
                    t0 = time.perf_counter()
                    c, found = refine(hsv, box, cfg)
                    elapsed[cfg["name"]] += time.perf_counter()-t0
                    valid_refines[cfg["name"]] += int(found)
                    fallback[cfg["name"]] += int(not found)
                    name = {"h10-40_s40_v40": "hsv_10_40_70_40",
                            "h10-40_s70_v40": "hsv_10_40_70_40",
                            "h10-40_s100_v40": "hsv_10_40_100_40",
                            "h10-40_s70_v80": "hsv_10_40_70_80",
                            "h15-35_s70_v40": "hsv_15_35_70_40",
                            "h8-45_s70_v40": "hsv_8_45_70_40"}[cfg["name"]]
                    if cfg["name"] in ("h10-40_s70_v40", "h10-40_s100_v40"):
                        centers[name].append(c)
                    elif cfg["name"] != "h10-40_s40_v40":
                        centers[name].append(c)

                short = min(box["x2"]-box["x1"], box["y2"]-box["y1"])
                x0, y0 = max(0, int(round(bx-short*.8))), max(0, int(round(by-short*.8)))
                x1, y1 = min(1600, int(round(bx+short*.8))), min(1200, int(round(by+short*.8)))
                crop = cv2.GaussianBlur(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY),
                                        (9, 9), 2)
                circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, dp=1.2,
                    minDist=max(10, int(short*.5)), param1=90, param2=20,
                    minRadius=max(8, int(short*.35)), maxRadius=max(9, int(short*.65))) if crop.size else None
                hs = []
                if circles is not None:
                    for hx, hy, _ in circles[0]:
                        hc = (float(hx+x0), float(hy+y0))
                        offset = math.dist(hc, (bx, by))
                        if offset <= short*.30:
                            hs.append((offset, hc))
                hough = min(hs, key=lambda x:x[0])[1] if hs else (bx, by)
                centers["hough075"].append((bx+.75*(hough[0]-bx), by+.75*(hough[1]-by)))

            for name, points in centers.items():
                if len(points) != 2:
                    continue
                xyz = []
                for target, center in enumerate(points):
                    data[name]["center_errors_px"].append(math.dist(center, TARGETS[target]))
                    p = point(center, float(ranges[target]))
                    xyz.append(p)
                    data[name]["target_xyz"][target].append(p)
                data[name]["pair_spacing_m"].append(float(np.linalg.norm(xyz[0]-xyz[1])))
                data[name]["valid_pairs"] += 1
        frame_idx += 1
    cap.release()

    configs = {}
    for name, row in data.items():
        configs[name] = {"valid_pairs": row["valid_pairs"],
                         "center_error_px": summary(row["center_errors_px"]),
                         "pair_spacing_m": summary(row["pair_spacing_m"])}
    out["results"].append({"scene": clip["scene"], "frames": frame_idx,
        "whole_clip_elapsed_s": round(time.perf_counter()-started, 2),
        "centers": configs,
        "segmentation_candidates": {cfg["name"]: {
            "refined_centers": valid_refines[cfg["name"]],
            "box_center_fallbacks": fallback[cfg["name"]],
            "measured_refinement_ms_per_box": round(1000*elapsed[cfg["name"]]/max(1, 2*frame_idx), 3)}
            for cfg in CONFIGS}})
    print(clip["scene"], json.dumps(out["results"][-1]))

path = ROOT / "media/frc-fuel-segmentation-center-evaluation.json"
path.write_text(json.dumps(out, indent=2)+"\n")
print("saved", path)
