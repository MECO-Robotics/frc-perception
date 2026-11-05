#!/usr/bin/env python3
"""Check candidate stereo-quality warnings on clean full-clip FUEL replay frames."""
import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COUNT_DATA = json.loads((ROOT / "media/frc-fuel-dense-count-evaluation.json").read_text())
TARGETS = ((808, 589), (902, 591))
BASELINE = 0.0635
PAD = 24
SAMPLE_STEP = 3
OUTLIER_M = 0.05
CALIBRATION = ROOT / "media/calibration/stereo_calibration_live_order_candidate.yml"

fs = cv2.FileStorage(str(CALIBRATION), cv2.FILE_STORAGE_READ)
kl = fs.getNode("left_camera_matrix").mat()
dl = fs.getNode("left_distortion").mat()
kr = fs.getNode("right_camera_matrix").mat()
dr = fs.getNode("right_distortion").mat()
rotation = fs.getNode("rotation").mat()
translation = fs.getNode("translation").mat()
fs.release()
size = (1600, 1200)
r1, r2, p1, p2, *_ = cv2.stereoRectify(
    kl, dl, kr, dr, size, rotation, translation,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
map_l = cv2.initUndistortRectifyMap(kl, dl, r1, p1, size, cv2.CV_32FC1)
map_r = cv2.initUndistortRectifyMap(kr, dr, r2, p2, size, cv2.CV_32FC1)
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=128, blockSize=5, P1=200, P2=800,
    uniquenessRatio=8, speckleWindowSize=50, speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
right_matcher = cv2.StereoSGBM_create(
    minDisparity=-128, numDisparities=128, blockSize=5, P1=200, P2=800,
    uniquenessRatio=8, speckleWindowSize=50, speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
focal = float(p1[0, 0])


def match_targets(detections):
    hits = [None, None]
    for box in sorted((d for d in detections if d["confidence"] >= 0.10),
                      key=lambda b: b["confidence"], reverse=True):
        cx = (box["x1"] + box["x2"]) / 2
        cy = (box["y1"] + box["y2"]) / 2
        distances = [math.hypot(cx-x, cy-y) for x, y in TARGETS]
        target = int(np.argmin(distances))
        if distances[target] <= 60 and hits[target] is None:
            hits[target] = box
    return hits


def rect_box(box):
    corners = np.array([[[box["x1"], box["y1"]]], [[box["x2"], box["y1"]]],
                        [[box["x2"], box["y2"]]], [[box["x1"], box["y2"]]]], np.float32)
    q = cv2.undistortPoints(corners, kl, dl, R=r1, P=p1).reshape(-1, 2) * 0.5
    return (max(0, int(np.floor(q[:, 0].min()))),
            max(0, int(np.floor(q[:, 1].min()))),
            min(800, int(np.ceil(q[:, 0].max()))),
            min(600, int(np.ceil(q[:, 1].max()))))


def quality(disparity, reverse, box, band_y0):
    x1, y1, x2, y2 = box
    y1 -= band_y0
    y2 -= band_y0
    roi = disparity[y1:y2, x1:x2]
    valid = np.isfinite(roi) & (roi > 3) & (roi < 120)
    vals = roi[valid]
    if vals.size < 8:
        return {"range_m": None, "mad_px": None,
                "lr_consistency_fraction": None, "valid_fraction": float(vals.size / max(1, roi.size))}
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    yy, xx = np.nonzero(valid)
    ds = roi[yy, xx]
    xr = np.rint(xx + x1 - ds).astype(np.int32)
    yr = yy + y1
    inside = (xr >= 0) & (xr < reverse.shape[1]) & (yr >= 0) & (yr < reverse.shape[0])
    rd = reverse[yr[inside], xr[inside]]
    consistent = np.isfinite(rd) & (rd > -120) & (rd < -3) & (np.abs(ds[inside] + rd) <= 1.0)
    lr_fraction = float(consistent.sum() / vals.size)
    return {"range_m": BASELINE * focal / (2 * median),
            "mad_px": mad, "lr_consistency_fraction": lr_fraction,
            "valid_fraction": float(vals.size / max(1, roi.size))}


def rolling_median(values, radius=5):
    out = []
    for i, value in enumerate(values):
        local = [x for x in values[max(0, i-radius):i+radius+1] if x is not None]
        out.append(float(statistics.median(local)) if local else None)
    return out


out = {
    "method": "Sample every third decoded frame from both clean saved clips. Compute active vertical-band SGBM (24 half-resolution px pad), reverse SGBM, and the candidate synthetic-occlusion warning rule: MAD >1.25 px OR left-right consistency <0.58. Define exploratory range outliers as >5 cm from a centered 11-sample rolling median (~1.65 s at the sampled cadence); this is a temporal repeatability label, not absolute error.",
    "limitations": "Stationary balls and camera; no independent distance truth, motion, or real occlusion. Thresholds are evaluated on the same two recordings used to select the active profile.",
    "sample_step_frames": SAMPLE_STEP,
    "quality_rule": {"mad_px_gt": 1.25, "or_lr_consistency_lt": 0.58},
    "range_outlier_threshold_m": OUTLIER_M,
    "clips": [],
}

for clip in COUNT_DATA["results"]:
    video = ROOT / clip["video"]
    cap = cv2.VideoCapture(str(video))
    rows = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = clip["per_frame"][index]["detections"]
        if index % SAMPLE_STEP == 0:
            boxes = match_targets(detections)
            if all(boxes):
                rects = [rect_box(b) for b in boxes]
                y0 = max(0, min(b[1] for b in rects) - PAD)
                y1 = min(600, max(b[3] for b in rects) + PAD)
                left = frame[:, :1600]
                right = frame[:, 1600:]
                left = cv2.resize(cv2.cvtColor(cv2.remap(left, *map_l, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY), (800, 600), interpolation=cv2.INTER_AREA)
                right = cv2.resize(cv2.cvtColor(cv2.remap(right, *map_r, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY), (800, 600), interpolation=cv2.INTER_AREA)
                disp = left_matcher.compute(left[y0:y1], right[y0:y1]).astype(np.float32) / 16
                rev = right_matcher.compute(right[y0:y1], left[y0:y1]).astype(np.float32) / 16
                target_rows = []
                for target, rect in enumerate(rects):
                    m = quality(disp, rev, rect, y0)
                    m.update({"target": target, "box_confidence": boxes[target]["confidence"]})
                    target_rows.append(m)
                rows.append({"frame": index, "time_s": index / clip["fps"], "targets": target_rows})
        index += 1
    cap.release()

    aggregate = []
    outlier_flags = {}
    for target in range(2):
        values = [row["targets"][target]["range_m"] for row in rows]
        local = rolling_median(values)
        target_rows = []
        for i, row in enumerate(rows):
            item = row["targets"][target]
            deviation = abs(item["range_m"] - local[i]) if item["range_m"] is not None and local[i] is not None else None
            warning = (item["mad_px"] is not None and item["mad_px"] > 1.25) or (
                item["lr_consistency_fraction"] is not None and item["lr_consistency_fraction"] < 0.58)
            target_rows.append({"warning": warning, "deviation_from_local_median_m": deviation,
                                "is_temporal_outlier": deviation is not None and deviation > OUTLIER_M})
        outlier_flags[target] = target_rows
        warnings = [x["warning"] for x in target_rows]
        outliers = [x["is_temporal_outlier"] for x in target_rows]
        tp = sum(w and b for w, b in zip(warnings, outliers))
        fp = sum(w and not b for w, b in zip(warnings, outliers))
        fn = sum(not w and b for w, b in zip(warnings, outliers))
        aggregate.append({
            "target": target,
            "samples": len(rows),
            "warning_count": sum(warnings),
            "warning_fraction": round(sum(warnings) / len(rows), 5) if rows else None,
            "temporal_outlier_count": sum(outliers),
            "warning_outlier_overlap": tp,
            "warning_false_positives": fp,
            "warning_missed_outliers": fn,
            "range_median_m": round(float(statistics.median(v for v in values if v is not None)), 5) if any(v is not None for v in values) else None,
            "range_mad_m": round(float(statistics.median(abs(v - statistics.median(x for x in values if x is not None)) for v in values if v is not None)), 5) if any(v is not None for v in values) else None,
        })
    out["clips"].append({"scene": clip["scene"], "video": clip["video"],
                         "sampled_rows": len(rows), "targets": aggregate, "per_sample": rows})
    print(clip["scene"], json.dumps(aggregate))

path = ROOT / "media/frc-fuel-clean-stereo-quality-evaluation.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print("saved", path)
