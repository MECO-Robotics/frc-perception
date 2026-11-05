#!/usr/bin/env python3
"""Full-clip box/Hough center blend sweep with leave-one-clip-out scoring."""
import json, math, statistics, time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS = json.loads((ROOT / "media/frc-fuel-dense-count-nms015.json").read_text())
STEREO = json.loads((ROOT / "media/frc-fuel-dense-stereo-nms015-conf005.json").read_text())
TARGETS = ((808.0, 589.0), (902.0, 591.0))
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
MERGE_RADII_M = (0.15, 0.22, 0.25, 0.26)

fs = cv2.FileStorage(str(ROOT / "media/calibration/stereo_calibration_live_order_candidate.yml"), cv2.FILE_STORAGE_READ)
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
        distances = [math.dist(center, target) for target in TARGETS]
        i = int(np.argmin(distances))
        if distances[i] <= 60 and hits[i] is None:
            hits[i] = box
    return hits


def stats(values):
    if not values:
        return None
    m = statistics.median(values)
    return {"n": len(values), "median": round(m, 5),
            "p90": round(float(np.percentile(values, 90)), 5),
            "p95": round(float(np.percentile(values, 95)), 5),
            "mad": round(statistics.median(abs(v-m) for v in values), 5)}


out = {
    "method": "Use detector-centered Hough param2=20 inside each matched detector box and blend its center with the YOLO box center at alpha 0/.25/.5/.75/1.0. Score pixel center error and reconstructed target/pair XYZ against manually labeled image centers using the same saved production SGBM Z.",
    "limitations": "Only the same stationary two-ball scene in two camera-setting videos; labels are fixed image centers and Z is not measured ground truth. This isolates XY localization and does not validate motion or physical occlusion.",
    "param2": 20, "alphas": ALPHAS, "cluster_merge_radii_m": MERGE_RADII_M, "results": []
}

for clip, stereo_clip in zip(DETECTIONS["results"], STEREO["results"]):
    cap = cv2.VideoCapture(str(ROOT / clip["video"]))
    buckets = {a: {"center_errors_px": [], "target_xyz_errors_m": [[], []],
                   "pair_distance_errors_m": [], "pair_distances_m": [],
                   "valid_hough_centers": 0, "fallback_centers": 0, "pairs": 0}
               for a in ALPHAS}
    frame_idx, started = 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = match(clip["per_frame"][frame_idx]["detections"])
        ranges = stereo_clip["frames"][frame_idx]["ranges_m"]
        if all(boxes) and all(z is not None for z in ranges):
            gray = cv2.cvtColor(frame[:, :1600], cv2.COLOR_BGR2GRAY)
            box_centers, hough_centers, refs = [], [], []
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
                hough = min(candidates, key=lambda value:value[0])[1] if candidates else None
                box_centers.append((bx, by))
                hough_centers.append(hough)
                q = cv2.undistortPoints(np.array([[[TARGETS[target][0], TARGETS[target][1]]]], np.float32),
                                               kl, dl, R=r1, P=p1)[0, 0]
                z = float(ranges[target])
                refs.append(np.array(((q[0]-cx)*z/fx, (q[1]-cy)*z/fy, z)))
            for alpha, rec in buckets.items():
                estimated = []
                for target, (box_center, hough, ref) in enumerate(zip(box_centers, hough_centers, refs)):
                    rec["valid_hough_centers"] += int(hough is not None)
                    rec["fallback_centers"] += int(hough is None)
                    center = box_center if hough is None else (
                        box_center[0] + alpha*(hough[0]-box_center[0]),
                        box_center[1] + alpha*(hough[1]-box_center[1]))
                    rec["center_errors_px"].append(math.dist(center, TARGETS[target]))
                    q = cv2.undistortPoints(np.array([[[center[0], center[1]]]], np.float32),
                                            kl, dl, R=r1, P=p1)[0, 0]
                    z = float(ranges[target])
                    point = np.array(((q[0]-cx)*z/fx, (q[1]-cy)*z/fy, z))
                    rec["target_xyz_errors_m"][target].append(float(np.linalg.norm(point-ref)))
                    estimated.append(point)
                pair_d = float(np.linalg.norm(estimated[0]-estimated[1]))
                pair_ref = float(np.linalg.norm(refs[0]-refs[1]))
                rec["pair_distances_m"].append(pair_d)
                rec["pair_distance_errors_m"].append(abs(pair_d-pair_ref))
                rec["pairs"] += 1
        frame_idx += 1
    cap.release()
    configs = []
    for alpha, rec in buckets.items():
        configs.append({"alpha": alpha, "valid_pairs": rec["pairs"],
            "valid_hough_centers": rec["valid_hough_centers"], "fallback_centers": rec["fallback_centers"],
            "center_error_px": stats(rec["center_errors_px"]),
            "target_xyz_error_m": [stats(v) for v in rec["target_xyz_errors_m"]],
            "pair_distance_m": stats(rec["pair_distances_m"]),
            "pair_distance_error_vs_manual_xy_m": stats(rec["pair_distance_errors_m"]),
            "cluster_merge_counts": [{"radius_m": radius,
                "merged_pairs": sum(distance <= radius for distance in rec["pair_distances_m"]),
                "valid_pairs": len(rec["pair_distances_m"]),
                "merge_fraction": round(sum(distance <= radius for distance in rec["pair_distances_m"])
                                         / max(1, len(rec["pair_distances_m"])), 5)}
                for radius in MERGE_RADII_M],
            "selection_score_median_plus_p95_px": round(
                stats(rec["center_errors_px"])["median"]+stats(rec["center_errors_px"])["p95"], 5)
                if rec["center_errors_px"] else None})
    out["results"].append({"scene": clip["scene"], "video_frames": frame_idx,
        "elapsed_s": round(time.perf_counter()-started, 2), "configs": configs})
    print(clip["scene"], json.dumps(configs))

cross_validation = []
for train_i, test_i in ((0, 1), (1, 0)):
    train, test = out["results"][train_i], out["results"][test_i]
    selected = min(train["configs"], key=lambda row: row["selection_score_median_plus_p95_px"])
    heldout = next(row for row in test["configs"] if row["alpha"] == selected["alpha"])
    box = next(row for row in test["configs"] if row["alpha"] == 0.0)
    cross_validation.append({"train_scene": train["scene"], "test_scene": test["scene"],
        "selected_alpha": selected["alpha"], "heldout_center_error_px": heldout["center_error_px"],
        "heldout_box_center_error_px": box["center_error_px"],
        "heldout_pair_distance_error_m": heldout["pair_distance_error_vs_manual_xy_m"],
        "heldout_box_pair_distance_error_m": box["pair_distance_error_vs_manual_xy_m"]})
out["leave_one_clip_out"] = cross_validation
path = ROOT / "media/frc-fuel-hough-center-blend-evaluation.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print("cross_validation", json.dumps(cross_validation))
print("saved", path)
