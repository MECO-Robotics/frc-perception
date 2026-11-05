#!/usr/bin/env python3
"""Full-clip Hough center sensitivity check against detector-box centers."""
import json, math, statistics, time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS = json.loads((ROOT / "media/frc-fuel-dense-count-nms015.json").read_text())
STEREO = json.loads((ROOT / "media/frc-fuel-dense-stereo-nms015-conf005.json").read_text())
TARGETS = ((808.0, 589.0), (902.0, 591.0))
PARAM2S = (20, 25, 30, 35)

fs = cv2.FileStorage(str(ROOT / "media/calibration/stereo_calibration_live_order_candidate.yml"), cv2.FILE_STORAGE_READ)
kl, dl = fs.getNode("left_camera_matrix").mat(), fs.getNode("left_distortion").mat()
kr, dr = fs.getNode("right_camera_matrix").mat(), fs.getNode("right_distortion").mat()
rotation = fs.getNode("rotation").mat()
translation = fs.getNode("translation").mat()
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
        target = int(np.argmin(distances))
        if distances[target] <= 60 and hits[target] is None:
            hits[target] = box
    return hits


def summarize(values):
    if not values:
        return None
    median = statistics.median(values)
    return {"n": len(values), "median": round(median, 5),
            "p90": round(float(np.percentile(values, 90)), 5),
            "p95": round(float(np.percentile(values, 95)), 5),
            "mad": round(statistics.median(abs(value - median) for value in values), 5)}


out = {
    "method": "Run detector-centered Hough circles on every frame in both saved clips for param2 20/25/30/35. Compare center localization and the resulting stereo XYZ against manually labeled image centers, using the same saved production SGBM range per target. Missing circles fall back to the YOLO box center.",
    "limitations": "The clips contain only two stationary balls; manual centers and SGBM ranges are the reference, not physical 3D truth. Hough remains a local center-refinement candidate and does not change detection count or stereo Z.",
    "target_centers_px": TARGETS, "param2_values": PARAM2S, "results": []
}

for clip, stereo_clip in zip(DETECTIONS["results"], STEREO["results"]):
    cap = cv2.VideoCapture(str(ROOT / clip["video"]))
    records = {p: {"box_px_error": [], "hough_px_error": [], "center_valid": 0,
                   "xyz_error": [[], []], "pair_distance": [], "pair_distance_ref_error": [],
                   "fallback_count": 0, "matched_pairs": 0} for p in PARAM2S}
    frame_idx = 0
    started = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame[:, :1600], cv2.COLOR_BGR2GRAY)
        boxes = match(clip["per_frame"][frame_idx]["detections"])
        ranges = stereo_clip["frames"][frame_idx]["ranges_m"]
        centers = {p: [None, None] for p in PARAM2S}
        for target, box in enumerate(boxes):
            if box is None:
                continue
            bx, by = (box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2
            short = min(box["x2"] - box["x1"], box["y2"] - box["y1"])
            x0, y0 = max(0, int(round(bx - short * .8))), max(0, int(round(by - short * .8)))
            x1, y1 = min(1600, int(round(bx + short * .8))), min(1200, int(round(by + short * .8)))
            crop = cv2.GaussianBlur(gray[y0:y1, x0:x1], (9, 9), 2)
            rmin, rmax = max(8, int(short * .35)), max(9, int(short * .65))
            for param2 in PARAM2S:
                circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, dp=1.2,
                    minDist=max(10, int(short * .5)), param1=90, param2=param2,
                    minRadius=rmin, maxRadius=rmax) if crop.size else None
                candidates = []
                if circles is not None:
                    for x, y, _radius in circles[0]:
                        center = (float(x + x0), float(y + y0))
                        offset = math.dist(center, (bx, by))
                        if offset <= short * .30:
                            candidates.append((offset, center))
                if candidates:
                    centers[param2][target] = min(candidates, key=lambda item: item[0])[1]
        for param2 in PARAM2S:
            rec = records[param2]
            if all(boxes):
                rec["matched_pairs"] += 1
            if all(boxes) and all(z is not None for z in ranges):
                box_centers = [((b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2) for b in boxes]
                est_centers = []
                reference_xyz = []
                estimated_xyz = []
                for target, z in enumerate(ranges):
                    fallback = centers[param2][target] is None
                    center = box_centers[target] if fallback else centers[param2][target]
                    rec["fallback_count"] += int(fallback)
                    rec["center_valid"] += int(not fallback)
                    rec["box_px_error"].append(math.dist(box_centers[target], TARGETS[target]))
                    rec["hough_px_error"].append(math.dist(center, TARGETS[target]))
                    q = cv2.undistortPoints(np.array([[[center[0], center[1]]]], np.float32),
                                            kl, dl, R=r1, P=p1)[0, 0]
                    qref = cv2.undistortPoints(np.array([[[TARGETS[target][0], TARGETS[target][1]]]], np.float32),
                                               kl, dl, R=r1, P=p1)[0, 0]
                    estimated_xyz.append(np.array(((q[0]-cx)*z/fx, (q[1]-cy)*z/fy, z)))
                    reference_xyz.append(np.array(((qref[0]-cx)*z/fx, (qref[1]-cy)*z/fy, z)))
                for target in range(2):
                    rec["xyz_error"][target].append(float(np.linalg.norm(estimated_xyz[target] - reference_xyz[target])))
                pair_distance = float(np.linalg.norm(estimated_xyz[0] - estimated_xyz[1]))
                reference_distance = float(np.linalg.norm(reference_xyz[0] - reference_xyz[1]))
                rec["pair_distance"].append(pair_distance)
                rec["pair_distance_ref_error"].append(abs(pair_distance-reference_distance))
        frame_idx += 1
    cap.release()
    configs = []
    for param2, rec in records.items():
        configs.append({"param2": param2, "matched_pairs": rec["matched_pairs"],
            "valid_hough_centers": rec["center_valid"], "fallback_centers": rec["fallback_count"],
            "box_center_error_px": summarize(rec["box_px_error"]),
            "refined_center_error_px": summarize(rec["hough_px_error"]),
            "target_xyz_error_m": [summarize(v) for v in rec["xyz_error"]],
            "pair_distance_m": summarize(rec["pair_distance"]),
            "pair_distance_error_vs_manual_xy_m": summarize(rec["pair_distance_ref_error"]),
            "hough_use_fraction": round(rec["center_valid"] / max(1, rec["center_valid"]+rec["fallback_count"]), 5)})
    out["results"].append({"scene": clip["scene"], "video_frames": frame_idx,
                           "elapsed_s": round(time.perf_counter()-started, 2), "configs": configs})
    print(clip["scene"], json.dumps(configs))

path = ROOT / "media/frc-fuel-hough-center-param-sweep.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print("saved", path)
