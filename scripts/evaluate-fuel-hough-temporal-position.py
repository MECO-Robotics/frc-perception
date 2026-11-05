#!/usr/bin/env python3
"""Evaluate Hough/box center blending with causal XYZ median smoothing offline."""
import json
import math
import statistics
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS = json.loads((ROOT / "media/frc-fuel-dense-count-nms015.json").read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stereo", default="media/frc-fuel-dense-stereo-nms015-conf005.json",
                    help="cached per-frame stereo ranges to use")
parser.add_argument("--output", default="media/frc-fuel-hough-temporal-position-evaluation.json")
args = parser.parse_args()
STEREO = json.loads((ROOT / args.stereo).read_text())
TARGETS = ((808.0, 589.0), (902.0, 591.0))
ALPHA = 0.75
WINDOW = 5

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
        center = ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)
        distances = [math.dist(center, target) for target in TARGETS]
        i = int(np.argmin(distances))
        if distances[i] <= 60 and hits[i] is None:
            hits[i] = box
    return hits


def stats(values):
    if not values:
        return None
    center = statistics.median(values)
    return {"n": len(values), "median": round(center, 6),
            "p95": round(float(np.percentile(values, 95)), 6),
            "mad": round(statistics.median(abs(v-center) for v in values), 6)}


def spread(points):
    a = np.asarray(points, dtype=float)
    center = np.median(a, axis=0)
    return {axis: {"median_m": round(float(center[i]), 6),
                   "mad_m": round(float(np.median(np.abs(a[:, i]-center[i]))), 6),
                   "p95_abs_deviation_m": round(float(np.percentile(np.abs(a[:, i]-center[i]), 95)), 6)}
            for i, axis in enumerate(("x", "y", "z"))}


def xyz(center, z):
    q = cv2.undistortPoints(np.array([[[center[0], center[1]]]], np.float32),
                            kl, dl, R=r1, P=p1)[0, 0]
    return np.asarray(((q[0]-cx)*z/fx, (q[1]-cy)*z/fy, z), dtype=float)


out = {"method": "Per saved frame, match the two known balls to cached YOLO boxes, run detector-crop Hough param2=20, blend Hough and box centers at alpha=.75 when a circle is found, and reconstruct XYZ with cached stereo Z. Compare raw box, blended center, and causal coordinate-wise XYZ median5.",
       "alpha": ALPHA, "median_window": WINDOW,
       "stereo_source": args.stereo,
       "stereo_block_size": STEREO.get("block_size"),
       "stereo_disparities": STEREO.get("num_disparities_match_px"),
       "limitations": "Only two stationary saved clips and manually labeled image centers; stereo Z is cached production output, not measured range truth. Temporal smoothing is assessed for repeatability only, not motion lag. Missing detections remain missing and do not create held positions.",
       "results": []}

for clip, stereo_clip in zip(DETECTIONS["results"], STEREO["results"]):
    cap = cv2.VideoCapture(str(ROOT / clip["video"]))
    histories = [[], []]
    rows = {name: {"center_errors_px": [], "target_xyz": [[], []],
                   "pair_separation_m": [], "valid_pairs": 0}
            for name in ("box_raw", "blend075", "blend075_median5")}
    frame_idx = 0
    started = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = match(clip["per_frame"][frame_idx]["detections"])
        ranges = stereo_clip["frames"][frame_idx]["ranges_m"]
        if all(boxes) and all(z is not None for z in ranges):
            gray = cv2.cvtColor(frame[:, :1600], cv2.COLOR_BGR2GRAY)
            centers_by_method = {"box_raw": [], "blend075": [], "blend075_median5": []}
            for target, box in enumerate(boxes):
                bx, by = (box["x1"]+box["x2"])/2, (box["y1"]+box["y2"])/2
                short = min(box["x2"]-box["x1"], box["y2"]-box["y1"])
                x0, y0 = max(0, int(round(bx-short*.8))), max(0, int(round(by-short*.8)))
                x1, y1 = min(1600, int(round(bx+short*.8))), min(1200, int(round(by+short*.8)))
                crop = cv2.GaussianBlur(gray[y0:y1, x0:x1], (9, 9), 2)
                rmin, rmax = max(8, int(short*.35)), max(9, int(short*.65))
                circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, dp=1.2,
                    minDist=max(10, int(short*.5)), param1=90, param2=20,
                    minRadius=rmin, maxRadius=rmax) if crop.size else None
                candidates = []
                if circles is not None:
                    for hx, hy, _radius in circles[0]:
                        center = (float(hx+x0), float(hy+y0))
                        offset = math.dist(center, (bx, by))
                        if offset <= short*.30:
                            candidates.append((offset, center))
                hough = min(candidates, key=lambda v: v[0])[1] if candidates else None
                blend = (bx, by) if hough is None else (bx+ALPHA*(hough[0]-bx), by+ALPHA*(hough[1]-by))
                centers_by_method["box_raw"].append((bx, by))
                centers_by_method["blend075"].append(blend)
                histories[target].append(xyz(blend, float(ranges[target])))
                histories[target] = histories[target][-WINDOW:]
                smooth_xyz = np.median(np.asarray(histories[target]), axis=0)
                centers_by_method["blend075_median5"].append(smooth_xyz)

            for name, centers in centers_by_method.items():
                points = []
                for target, center in enumerate(centers):
                    point = center if name == "blend075_median5" else xyz(center, float(ranges[target]))
                    if name != "blend075_median5":
                        rows[name]["center_errors_px"].append(math.dist(center, TARGETS[target]))
                    points.append(point)
                    rows[name]["target_xyz"][target].append(point)
                rows[name]["pair_separation_m"].append(float(np.linalg.norm(points[0]-points[1])))
                rows[name]["valid_pairs"] += 1
        else:
            histories = [[], []]
        frame_idx += 1
    cap.release()
    configs = {}
    for name, row in rows.items():
        configs[name] = {"valid_pairs": row["valid_pairs"],
                         "pair_separation_m": stats(row["pair_separation_m"]),
                         "target_xyz_spread": [spread(v) for v in row["target_xyz"]]}
        if row["center_errors_px"]:
            configs[name]["center_error_px"] = stats(row["center_errors_px"])
    out["results"].append({"scene": clip["scene"], "video_frames": frame_idx,
                            "elapsed_s": round(time.perf_counter()-started, 2),
                            "configs": configs})
    print(clip["scene"], json.dumps(configs))

path = ROOT / args.output
path.write_text(json.dumps(out, indent=2) + "\n")
print("saved", path)
