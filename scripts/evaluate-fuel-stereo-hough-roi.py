#!/usr/bin/env python3
"""Compare Hough-circle interior disparity ROIs with full detector-box median depth."""
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS = json.loads((ROOT / "media/frc-fuel-dense-count-evaluation.json").read_text())
TARGETS = ((808, 589), (902, 591))
BASE = 0.0635
STEP = 3
PAD = 24
RADII = (0.55, 0.70, 0.85)
fs = cv2.FileStorage(str(ROOT / "media/calibration/stereo_calibration_live_order_candidate.yml"), cv2.FILE_STORAGE_READ)
kl, dl = fs.getNode("left_camera_matrix").mat(), fs.getNode("left_distortion").mat()
kr, dr = fs.getNode("right_camera_matrix").mat(), fs.getNode("right_distortion").mat()
rotation, translation = fs.getNode("rotation").mat(), fs.getNode("translation").mat()
fs.release()
size = (1600, 1200)
r1, r2, p1, p2, *_ = cv2.stereoRectify(kl, dl, kr, dr, size, rotation, translation,
                                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
map_l = cv2.initUndistortRectifyMap(kl, dl, r1, p1, size, cv2.CV_32FC1)
map_r = cv2.initUndistortRectifyMap(kr, dr, r2, p2, size, cv2.CV_32FC1)
focal = float(p1[0, 0])
matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5,
                                P1=200, P2=800, uniquenessRatio=8,
                                speckleWindowSize=50, speckleRange=2,
                                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def match(dets):
    found = [None, None]
    for box in sorted((b for b in dets if b["confidence"] >= 0.05),
                      key=lambda b: b["confidence"], reverse=True):
        cx, cy = (box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2
        ds = [math.hypot(cx-x, cy-y) for x, y in TARGETS]
        i = int(np.argmin(ds))
        if ds[i] <= 60 and found[i] is None:
            found[i] = box
    return found


def rect_box(box):
    corners = np.array([[[box["x1"], box["y1"]]], [[box["x2"], box["y1"]]],
                        [[box["x2"], box["y2"]]], [[box["x1"], box["y2"]]]], np.float32)
    q = cv2.undistortPoints(corners, kl, dl, R=r1, P=p1).reshape(-1, 2)
    return (max(0, int(np.floor(q[:, 0].min() * .5))),
            max(0, int(np.floor(q[:, 1].min() * .5))),
            min(800, int(np.ceil(q[:, 0].max() * .5))),
            min(600, int(np.ceil(q[:, 1].max() * .5))))


def circle_for(gray, box):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    cx, cy = (x1+x2)/2, (y1+y2)/2
    short = min(x2-x1, y2-y1)
    x0, y0 = max(0, int(round(cx-short*.8))), max(0, int(round(cy-short*.8)))
    x1c, y1c = min(1600, int(round(cx+short*.8))), min(1200, int(round(cy+short*.8)))
    crop = cv2.GaussianBlur(gray[y0:y1c, x0:x1c], (9, 9), 2)
    if not crop.size:
        return None
    rmin, rmax = max(8, int(short*.35)), max(9, int(short*.65))
    circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=max(10, int(short*.5)), param1=90,
                               param2=25, minRadius=rmin, maxRadius=rmax)
    if circles is None:
        return None
    candidates = []
    for x, y, radius in circles[0]:
        c = (float(x+x0), float(y+y0), float(radius))
        off = math.hypot(c[0]-cx, c[1]-cy)
        if off <= short*.30:
            candidates.append((off, c))
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def rectify_circle(circle):
    if circle is None:
        return None
    x, y, radius = circle
    pts = np.array([[[x, y]], [[x + radius, y]], [[x, y + radius]]], np.float32)
    q = cv2.undistortPoints(pts, kl, dl, R=r1, P=p1).reshape(-1, 2)
    rr = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[0])) / 2
    return (float(q[0, 0]), float(q[0, 1]), float(rr))


def estimate(roi, valid):
    vals = roi[valid]
    if vals.size < 8:
        return None
    return BASE * focal / (2 * float(np.median(vals)))


def summary(vals):
    vals = [x for x in vals if x is not None]
    if not vals:
        return {"valid": 0}
    med = statistics.median(vals)
    return {"valid": len(vals), "median_m": round(med, 5),
            "mad_m": round(statistics.median(abs(x-med) for x in vals), 5),
            "p05_m": round(float(np.percentile(vals, 5)), 5),
            "p95_m": round(float(np.percentile(vals, 95)), 5)}


out = {
    "method": "Every third frame: rectify stereo, calculate active 24 px vertical-band SGBM, compare full detector-box median disparity with circular masks centered on Hough param2=25 circles at 0.55/0.70/0.85 of fitted radius. Depth uses active 63.5 mm baseline and rectified focal length.",
    "limitations": "Static clips; no measured distance truth; Hough circles are only fit inside detector boxes. Range medians and MAD measure repeatability, not accuracy.",
    "sample_step_frames": STEP,
    "vertical_roi_pad_half_px": PAD,
    "hough_param2": 25,
    "results": [],
}

for clip in DETECTIONS["results"]:
    cap = cv2.VideoCapture(str(ROOT / clip["video"]))
    accum = {"box_full": [[], []]}
    accum.update({f"hough_r{r:.2f}": [[], []] for r in RADII})
    pair_gaps = {k: [] for k in accum}
    valid_hough_both = 0
    idx = 0
    started = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % STEP == 0:
            boxes = match(clip["per_frame"][idx]["detections"])
            if all(boxes):
                left_raw, right_raw = frame[:, :1600], frame[:, 1600:]
                left_rect = cv2.remap(left_raw, *map_l, cv2.INTER_LINEAR)
                right_rect = cv2.remap(right_raw, *map_r, cv2.INTER_LINEAR)
                gray_left = cv2.resize(cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY), (800, 600), interpolation=cv2.INTER_AREA)
                gray_right = cv2.resize(cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY), (800, 600), interpolation=cv2.INTER_AREA)
                rects = [rect_box(b) for b in boxes]
                y0 = max(0, min(b[1] for b in rects) - PAD)
                y1 = min(600, max(b[3] for b in rects) + PAD)
                disp = matcher.compute(gray_left[y0:y1], gray_right[y0:y1]).astype(np.float32) / 16
                raw_gray = cv2.cvtColor(left_raw, cv2.COLOR_BGR2GRAY)
                circles = [rectify_circle(circle_for(raw_gray, b)) for b in boxes]
                estimates = {k: [None, None] for k in accum}
                for target, (x1, yy1, x2, yy2) in enumerate(rects):
                    roi = disp[max(0, yy1-y0):max(0, yy2-y0), x1:x2]
                    valid = np.isfinite(roi) & (roi > 3) & (roi < 120)
                    estimates["box_full"][target] = estimate(roi, valid)
                    circle = circles[target]
                    if circle is not None:
                        ccx, ccy, radius = circle
                        for r in RADII:
                            yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
                            center_x, center_y = ccx*.5-x1, ccy*.5-(y0+max(0, yy1-y0))
                            mask = (xx+0.5-center_x)**2 + (yy+0.5-center_y)**2 <= (radius*.5*r)**2
                            estimates[f"hough_r{r:.2f}"][target] = estimate(roi, valid & mask)
                valid_hough_both += int(all(c is not None for c in circles))
                for key in accum:
                    for target in range(2):
                        accum[key][target].append(estimates[key][target])
                    a, b = estimates[key]
                    pair_gaps[key].append(abs(a-b) if a is not None and b is not None else None)
        idx += 1
    cap.release()
    configs = []
    for key, by_target in accum.items():
        configs.append({"method": key, "ranges_m": [summary(x) for x in by_target],
                        "absolute_range_gap_m": summary(pair_gaps[key])})
    out["results"].append({"scene": clip["scene"], "frames": idx,
                           "sampled_frames": len(pair_gaps["box_full"]),
                           "both_hough_centers_valid": valid_hough_both,
                           "elapsed_s": round(time.perf_counter()-started, 2),
                           "configs": configs})
    print(clip["scene"], json.dumps(configs))

path = ROOT / "media/frc-fuel-stereo-hough-roi-sweep.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print("saved", path)
