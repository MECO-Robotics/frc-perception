#!/usr/bin/env python3
"""Compare FUEL cluster-centroid estimators using detector confidence weights."""
import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COUNTS = json.loads((ROOT / "media/frc-fuel-dense-count-evaluation.json").read_text())
STEREO = json.loads((ROOT / "media/frc-fuel-dense-stereo-evaluation.json").read_text())
TARGETS = ((808.0, 589.0), (902.0, 591.0))
POWERS = (0.0, .5, 1.0, 2.0)
METHODS = ("mean", "coordinate_median") + tuple(f"confidence_power_{p:g}" for p in POWERS)
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
        x, y = (b["x1"]+b["x2"])/2, (b["y1"]+b["y2"])/2
        ds = [math.hypot(x-a, y-c) for a, c in TARGETS]
        i = int(np.argmin(ds))
        if ds[i] <= 60 and out[i] is None:
            out[i] = b
    return out


def xyz(center, z):
    q = cv2.undistortPoints(np.array([[[center[0], center[1]]]], np.float32), kl, dl,
                            R=r1, P=p1)[0, 0]
    return np.array(((q[0]-cx)*z/fx, (q[1]-cy)*z/fy, z), np.float64)


def med(values):
    return statistics.median(values) if values else None


def summarize(points, reference_points, pair_dist, reference_dist):
    if not points:
        return {"n": 0}
    arr = np.asarray(points)
    ref = np.asarray(reference_points)
    errors = np.linalg.norm(arr-ref, axis=1)
    err_pair = [abs(a-b) for a, b in zip(pair_dist, reference_dist)]
    center = np.median(arr, axis=0)
    return {
        "n": len(points),
        "median_centroid_error_vs_labeled_centers_m": round(med(errors.tolist()), 5),
        "p95_centroid_error_vs_labeled_centers_m": round(float(np.percentile(errors, 95)), 5),
        "median_pair_distance_error_vs_labeled_centers_m": round(med(err_pair), 5),
        "p95_pair_distance_error_vs_labeled_centers_m": round(float(np.percentile(err_pair, 95)), 5),
        "centroid_repeatability_mad_xyz": [round(med(np.abs(arr[:, i]-center[i]).tolist()), 5) for i in range(3)],
        "median_centroid_camera_xyz_m": [round(float(x), 5) for x in center],
    }


out = {
    "method": "For each frame with two matched confidence>=0.05 detections and valid SGBM ranges, reconstruct per-ball xyz from box centers. Compare equal mean, coordinate-wise median, and confidence^power weighted means (powers .5/1/2) against the centroid and pair distance from fixed manually labeled image centers using the same per-ball Z values.",
    "limitations": "Only two stationary balls and one camera pose; fixed manual centers; SGBM Z is reused for every method. This isolates XY centroid weighting, not absolute 3D range accuracy or multi-cluster false-merge behavior.",
    "confidence_powers": list(POWERS),
    "results": [],
}

for ci, clip in enumerate(COUNTS["results"]):
    points = {m: [] for m in METHODS}
    reference = {m: [] for m in METHODS}
    pair_distances = {m: [] for m in METHODS}
    reference_distances = {m: [] for m in METHODS}
    confidences = []
    for row, srow in zip(clip["per_frame"], STEREO["results"][ci]["frames"]):
        boxes = matched(row["detections"])
        zs = srow["ranges_m"]
        if not all(boxes) or not all(z is not None for z in zs):
            continue
        estimated = [xyz(((b["x1"]+b["x2"])/2, (b["y1"]+b["y2"])/2), z)
                     for b, z in zip(boxes, zs)]
        labeled = [xyz(TARGETS[i], zs[i]) for i in range(2)]
        conf = np.asarray([max(.001, b["confidence"]) for b in boxes], np.float64)
        confidences.append(conf.tolist())
        dref = float(np.linalg.norm(labeled[0]-labeled[1]))
        for method in METHODS:
            if method == "mean":
                center = np.mean(estimated, axis=0)
            elif method == "coordinate_median":
                center = np.median(np.asarray(estimated), axis=0)
            else:
                power = float(method.rsplit("_", 1)[1])
                weights = conf ** power
                center = np.average(np.asarray(estimated), axis=0, weights=weights)
            ref_center = np.mean(labeled, axis=0)
            points[method].append(center.tolist())
            reference[method].append(ref_center.tolist())
            pair_distances[method].append(float(np.linalg.norm(estimated[0]-estimated[1])))
            reference_distances[method].append(dref)
    results = [{"method": method,
                **summarize(points[method], reference[method], pair_distances[method], reference_distances[method])}
               for method in METHODS]
    conf_medians = [round(med([x[i] for x in confidences]), 4) for i in range(2)] if confidences else [None, None]
    out["results"].append({"scene": clip["scene"], "valid_pair_frames": len(points["mean"]),
                           "median_confidence_by_target": conf_medians,
                           "estimators": results})
    print(clip["scene"], json.dumps({"median_confidence": conf_medians, "estimators": results}))

path = ROOT / "media/frc-fuel-cluster-centroid-sweep.json"
path.write_text(json.dumps(out, indent=2)+"\n")
print("saved", path)
