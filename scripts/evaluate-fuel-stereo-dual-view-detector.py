#!/usr/bin/env python3
"""Evaluate FUEL detections and center-disparity range from both stereo views."""
import itertools
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scout.detectors import TensorRTDetector
VIDEOS = (
    ("earlier", "media/frc-2026-live-two-fuel-60fps-cfr.mp4"),
    ("corrected-settings", "media/frc-2026-fuel-camera-settings-fixed-60fps.mp4"),
)
TARGETS = ((808.0, 589.0), (902.0, 591.0))
SGBM = json.loads((ROOT / "media/frc-fuel-dense-stereo-evaluation.json").read_text())
STEP = 5
BASELINE = .0635
THRESHOLDS = (.05, .10, .20)
os.environ["SCOUT_CLASS_NAMES"] = '["ball"]'
os.environ["SCOUT_MAX_BOX_ASPECT_RATIO"] = "2.0"

fs = cv2.FileStorage(str(ROOT / "media/calibration/stereo_calibration_live_order_candidate.yml"), cv2.FILE_STORAGE_READ)
kl, dl = fs.getNode("left_camera_matrix").mat(), fs.getNode("left_distortion").mat()
kr, dr = fs.getNode("right_camera_matrix").mat(), fs.getNode("right_distortion").mat()
rotation, translation = fs.getNode("rotation").mat(), fs.getNode("translation").mat()
fs.release()
r1, r2, p1, p2, *_ = cv2.stereoRectify(kl, dl, kr, dr, (1600, 1200), rotation, translation,
                                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
focal = float(p1[0, 0])
RECT_TARGETS = [tuple(map(float, cv2.undistortPoints(
    np.array([[[x, y]]], np.float32), kl, dl, R=r1, P=p1)[0, 0])) for x, y in TARGETS]


def rect_center(box, camera):
    matrix, dist, rect = (kl, dl, r1) if camera == "left" else (kr, dr, r2)
    center = np.array([[[float(box.x1 + box.x2) / 2,
                         float(box.y1 + box.y2) / 2]]], np.float32)
    q = cv2.undistortPoints(center, matrix, dist, R=rect, P=p1)[0, 0]
    return float(q[0]), float(q[1])


def rect_dict(box, camera):
    center = rect_center(box, camera)
    return {"x": center[0], "y": center[1],
            "w": float(box.x2 - box.x1), "h": float(box.y2 - box.y1),
            "confidence": float(box.confidence)}


def assign(left, right):
    """Greedy global match with epipolar, disparity, and size checks."""
    candidates = []
    for i, l in enumerate(left):
        for j, r in enumerate(right):
            disparity = l["x"] - r["x"]
            dy = abs(l["y"] - r["y"])
            size_ratio = max(l["w"] / max(1, r["w"]), r["w"] / max(1, l["w"]),
                             l["h"] / max(1, r["h"]), r["h"] / max(1, l["h"]))
            if 10 <= disparity <= 110 and dy <= 12 and size_ratio <= 2.0:
                cost = dy + 0.15 * abs(math.log(size_ratio)) + 0.01 * abs(l["confidence"] - r["confidence"])
                candidates.append((cost, i, j, disparity, dy, size_ratio))
    chosen = []
    used_l, used_r = set(), set()
    for row in sorted(candidates):
        _, i, j, disparity, dy, size_ratio = row
        if i in used_l or j in used_r:
            continue
        used_l.add(i); used_r.add(j)
        chosen.append({"left_index": i, "right_index": j,
                       "disparity_px": disparity, "epipolar_error_px": dy,
                       "size_ratio": size_ratio})
    return chosen


def match_left_target(boxes, threshold, target):
    choices = [b for b in boxes if b["confidence"] >= threshold]
    for box in sorted(choices, key=lambda b: b["confidence"], reverse=True):
        if math.dist((box["x"], box["y"]), RECT_TARGETS[target]) <= 65:
            return box
    return None


def stats(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return {"n": 0}
    med = statistics.median(values)
    return {"n": len(values), "median": round(med, 5),
            "mad": round(statistics.median(abs(x-med) for x in values), 5),
            "p05": round(float(np.percentile(values, 5)), 5),
            "p95": round(float(np.percentile(values, 95)), 5)}


out = {
    "method": "Run the same 1280x960 FP16 FRC9550 TensorRT detector on both halves of sampled synchronized stereo frames. Rectify detection centers, associate candidates using positive disparity 10-110 px, epipolar residual <=12 px and size ratio <=2, then compute Z=baseline*focal/disparity. Compare with production SGBM ranges on the same frame.",
    "limitations": "Sampled static clips with two known targets; right-view associations are not validated against independently labeled right boxes. Center-disparity range depends on two independently estimated box centers and is not absolute range truth.",
    "sample_step_frames": STEP,
    "stereo_baseline_m": BASELINE,
    "rectified_focal_px": focal,
    "thresholds": list(THRESHOLDS),
    "results": [],
}

detector = TensorRTDetector(str(ROOT / "models/rebuilt_fuel_yolo26n_1280x960_fp16.engine"), confidence=.05)
try:
    for ci, (scene, video_name) in enumerate(VIDEOS):
        video = ROOT / video_name
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        records = []
        latencies = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        started = time.perf_counter()
        for idx in range(0, total_frames, STEP):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            left_img, right_img = frame[:, :1600], frame[:, 1600:]
            left_boxes, lt = detector.infer(left_img)
            right_boxes, rt = detector.infer(right_img)
            latencies.extend((lt, rt))
            left_rect = [rect_dict(b, "left") for b in left_boxes]
            right_rect = [rect_dict(b, "right") for b in right_boxes]
            associations = assign(left_rect, right_rect)
            by_threshold = {}
            sgbm_frame = SGBM["results"][ci]["frames"][idx]
            for threshold in THRESHOLDS:
                matched = []
                ranges = []
                for target in range(2):
                    l = match_left_target(left_rect, threshold, target)
                    if l is None:
                        matched.append(None); ranges.append(None); continue
                    li = left_rect.index(l)
                    pair = next((a for a in associations
                                 if a["left_index"] == li and
                                 right_rect[a["right_index"]]["confidence"] >= threshold), None)
                    if pair is None:
                        matched.append({"left": l, "right": None, "disparity_px": None})
                        ranges.append(None)
                    else:
                        r = right_rect[pair["right_index"]]
                        disparity = pair["disparity_px"]
                        z = BASELINE * focal / disparity
                        matched.append({"left": l, "right": r, **pair})
                        ranges.append(z)
                by_threshold[str(threshold)] = {
                    "matched_targets": sum(x is not None and x["right"] is not None for x in matched),
                    "center_ranges_m": ranges,
                    "sgbm_ranges_m": sgbm_frame["ranges_m"],
                    "detections_left": sum(b["confidence"] >= threshold for b in left_rect),
                    "detections_right": sum(b["confidence"] >= threshold for b in right_rect),
                    "matches": matched,
                }
            records.append({"frame": idx, "time_s": round(idx/fps, 4),
                            "left_detections": left_rect, "right_detections": right_rect,
                            "associations": associations, "thresholds": by_threshold})
        cap.release()
        threshold_summaries = []
        for threshold in THRESHOLDS:
            key = str(threshold)
            paired, exact_pair = 0, 0
            zvals = [[], []]
            errors = [[], []]
            disparity_values = []
            epipolar = []
            fp_pair = 0
            for row in records:
                value = row["thresholds"][key]
                valid = [m for m in value["matches"] if m and m["right"] is not None]
                paired += len(valid)
                exact_pair += len(valid) == 2
                fp_pair += max(0, len(row["associations"]) - len(valid))
                for target, match in enumerate(value["matches"]):
                    if match and match["right"] is not None:
                        z = value["center_ranges_m"][target]
                        zvals[target].append(z)
                        ref = value["sgbm_ranges_m"][target]
                        if ref is not None: errors[target].append(abs(z-ref))
                        disparity_values.append(match["disparity_px"])
                        epipolar.append(match["epipolar_error_px"])
            threshold_summaries.append({
                "confidence": threshold,
                "sampled_frames": len(records),
                "paired_target_instances": paired,
                "frames_with_two_stereo_pairs": exact_pair,
                "extra_stereo_associations": fp_pair,
                "center_disparity_ranges_m": [stats(a) for a in zvals],
                "absolute_range_difference_vs_sgbm_m": [stats(a) for a in errors],
                "center_disparity_px": stats(disparity_values),
                "epipolar_error_px": stats(epipolar),
            })
        result = {"scene": scene, "video": video_name,
                  "source_fps": fps, "source_frames": total_frames,
                  "sampled_frames": len(records),
                  "per_view_detector_ms_median": round(statistics.median(latencies), 2),
                  "per_view_detector_ms_p95": round(float(np.percentile(latencies, 95)), 2),
                  "elapsed_s": round(time.perf_counter()-started, 2),
                  "threshold_sweep": threshold_summaries,
                  "per_frame": records}
        out["results"].append(result)
        print(scene, json.dumps(threshold_summaries))
finally:
    detector.close()

path = ROOT / "media/frc-fuel-stereo-dual-view-detector-sweep.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print("saved", path)
