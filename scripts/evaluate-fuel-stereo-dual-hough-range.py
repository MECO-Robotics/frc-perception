#!/usr/bin/env python3
"""Test dual-view Hough centers as an alternative to box-center stereo disparity."""
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DUAL = json.loads((ROOT / "media/frc-fuel-stereo-dual-view-detector-sweep.json").read_text())
SGBM = json.loads((ROOT / "media/frc-fuel-dense-stereo-evaluation.json").read_text())
BASELINE = .0635
P2 = 25

fs = cv2.FileStorage(str(ROOT / "media/calibration/stereo_calibration_live_order_candidate.yml"), cv2.FILE_STORAGE_READ)
kl, dl = fs.getNode("left_camera_matrix").mat(), fs.getNode("left_distortion").mat()
kr, dr = fs.getNode("right_camera_matrix").mat(), fs.getNode("right_distortion").mat()
rotation, translation = fs.getNode("rotation").mat(), fs.getNode("translation").mat()
fs.release()
r1, r2, p1, p2mat, *_ = cv2.stereoRectify(kl, dl, kr, dr, (1600, 1200), rotation, translation,
                                           flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
map_l = cv2.initUndistortRectifyMap(kl, dl, r1, p1, (1600, 1200), cv2.CV_32FC1)
map_r = cv2.initUndistortRectifyMap(kr, dr, r2, p2mat, (1600, 1200), cv2.CV_32FC1)
focal = float(p1[0, 0])


def hough_center(gray, detection):
    cx, cy = detection["x"], detection["y"]
    short = min(detection["w"], detection["h"])
    x0, y0 = max(0, int(round(cx-short*.8))), max(0, int(round(cy-short*.8)))
    x1, y1 = min(1600, int(round(cx+short*.8))), min(1200, int(round(cy+short*.8)))
    crop = cv2.GaussianBlur(gray[y0:y1, x0:x1], (9, 9), 2)
    if not crop.size:
        return None
    circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=max(10, int(short*.5)), param1=90, param2=P2,
                               minRadius=max(8, int(short*.35)), maxRadius=max(9, int(short*.65)))
    if circles is None:
        return None
    choices = []
    for x, y, _radius in circles[0]:
        point = (float(x+x0), float(y+y0))
        offset = math.dist(point, (cx, cy))
        if offset <= short*.30:
            choices.append((offset, point))
    return min(choices, key=lambda a: a[0])[1] if choices else None


def stats(values):
    vals = [float(x) for x in values if x is not None and math.isfinite(x)]
    if not vals:
        return {"n": 0}
    med = statistics.median(vals)
    return {"n": len(vals), "median": round(med, 5),
            "mad": round(statistics.median(abs(x-med) for x in vals), 5),
            "p05": round(float(np.percentile(vals, 5)), 5),
            "p95": round(float(np.percentile(vals, 95)), 5)}


out = {
    "method": "For every sampled left/right detector association at confidence 0.05, rectify both images, fit Hough param2=25 circles inside each detection box, and compute Z from left-right center disparity. Compare with box-center disparity and SGBM range for the same frame/target.",
    "limitations": "Uses detector associations on stationary two-ball clips; Hough outputs are selected near the detector box centers. SGBM is a repeatability reference, not metric ground truth.",
    "hough_param2": P2,
    "stereo_baseline_m": BASELINE,
    "rectified_focal_px": focal,
    "results": [],
}

for ci, clip in enumerate(DUAL["results"]):
    cap = cv2.VideoCapture(str(ROOT / clip["video"]))
    accum = {method: [[], []] for method in ("box_center", "hough_center")}
    errors = {method: [[], []] for method in accum}
    frame_pair_counts = {method: 0 for method in accum}
    hough_pair_counts = 0
    latency = []
    started = time.perf_counter()
    for row in clip["per_frame"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
        ok, frame = cap.read()
        if not ok:
            continue
        matches = row["thresholds"]["0.05"]["matches"]
        if not any(m and m["right"] is not None for m in matches):
            continue
        left = cv2.remap(frame[:, :1600], *map_l, cv2.INTER_LINEAR)
        right = cv2.remap(frame[:, 1600:], *map_r, cv2.INTER_LINEAR)
        gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        current_hough_valid = [False, False]
        for target, match in enumerate(matches):
            if not match or match["right"] is None:
                continue
            l, r = match["left"], match["right"]
            box_disp = l["x"] - r["x"]
            box_z = BASELINE * focal / box_disp if box_disp > 0 else None
            sgbm_z = row["thresholds"]["0.05"]["sgbm_ranges_m"][target]
            if box_z is not None:
                accum["box_center"][target].append(box_z)
                errors["box_center"][target].append(abs(box_z-sgbm_z) if sgbm_z else None)
            t0 = time.perf_counter()
            lc, rc = hough_center(gl, l), hough_center(gr, r)
            latency.append((time.perf_counter()-t0)*1000)
            if lc and rc:
                disp = lc[0] - rc[0]
                z = BASELINE*focal/disp if disp > 0 else None
                accum["hough_center"][target].append(z)
                errors["hough_center"][target].append(abs(z-sgbm_z) if z and sgbm_z else None)
                current_hough_valid[target] = z is not None
        if all(m and m["right"] is not None for m in matches):
            frame_pair_counts["box_center"] += 1
            if all(current_hough_valid):
                hough_pair_counts += 1
    cap.release()
    summary = []
    for method in accum:
        summary.append({"method": method,
                        "ranges_m": [stats(values) for values in accum[method]],
                        "absolute_difference_vs_sgbm_m": [stats(values) for values in errors[method]],
                        "frames_with_both_pairs": frame_pair_counts[method] if method == "box_center" else hough_pair_counts})
    result = {"scene": clip["scene"], "sampled_frames": len(clip["per_frame"]),
              "summaries": summary,
              "hough_ms_per_pair_median": round(statistics.median(latency), 3) if latency else None,
              "hough_ms_per_pair_p95": round(float(np.percentile(latency, 95)), 3) if latency else None,
              "elapsed_s": round(time.perf_counter()-started, 2)}
    out["results"].append(result)
    print(clip["scene"], json.dumps(result))

path = ROOT / "media/frc-fuel-stereo-dual-hough-range-sweep.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print("saved", path)
