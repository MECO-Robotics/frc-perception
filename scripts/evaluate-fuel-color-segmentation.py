#!/usr/bin/env python3
"""Evaluate yellow-mask connected components and distance-peak splitting on saved FUEL clips."""
import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = (
    ("earlier", "media/frc-2026-live-two-fuel-60fps-cfr.mp4"),
    ("corrected-settings", "media/frc-2026-fuel-camera-settings-fixed-60fps.mp4"),
)
TARGETS = ((808, 589), (902, 591))
MATCH_RADIUS = 60.0
CONFIGS = (
    {"name": "h10-40_s40_v40_raw", "h": (10, 40), "s": 40, "v": 40, "morph": "raw"},
    {"name": "h10-40_s70_v40_raw", "h": (10, 40), "s": 70, "v": 40, "morph": "raw"},
    {"name": "h10-40_s100_v40_raw", "h": (10, 40), "s": 100, "v": 40, "morph": "raw"},
    {"name": "h10-40_s70_v80_open", "h": (10, 40), "s": 70, "v": 80, "morph": "open"},
    {"name": "h10-40_s100_v80_open", "h": (10, 40), "s": 100, "v": 80, "morph": "open"},
    {"name": "h15-35_s70_v40_open", "h": (15, 35), "s": 70, "v": 40, "morph": "open"},
    {"name": "h8-45_s70_v40_open", "h": (8, 45), "s": 70, "v": 40, "morph": "open"},
)


def candidates(mask, split):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    found = []
    for label in range(1, n):
        x, y, w, h, area = map(int, stats[label])
        if area < 500 or area > 50000 or min(w, h) < 15 or max(w, h) / min(w, h) > 1.8:
            continue
        component = labels[y:y+h, x:x+w] == label
        if split:
            dist = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
            peak_mask = (dist >= max(12.0, float(dist.max()) * 0.62)).astype(np.uint8)
            peak_count, peak_labels = cv2.connectedComponents(peak_mask, 8)
            peaks = []
            for peak_id in range(1, peak_count):
                ys, xs = np.where(peak_labels == peak_id)
                if len(xs) >= 2:
                    peaks.append((x + float(xs.mean()), y + float(ys.mean())))
            if 1 < len(peaks) <= 5:
                found.extend(peaks)
                continue
        found.append((float(centroids[label][0]), float(centroids[label][1])))
    return found


def score(points):
    hits = [False, False]
    fp = 0
    for px, py in points:
        choices = [(math.hypot(px - tx, py - ty), i) for i, (tx, ty) in enumerate(TARGETS)]
        distance, i = min(choices)
        if distance <= MATCH_RADIUS and not hits[i]:
            hits[i] = True
        else:
            fp += 1
    return sum(hits), fp


out = {
    "method": "Per-frame HSV yellow segmentation of the rectified left image, followed by connected-component center detection; one variant uses distance-transform local maxima to split joined masks. Match to manually labeled stationary ball centers with a 60 px gate, same gate as the detector benchmark.",
    "limitations": "Two static clips with only two balls; fixed target centers; no unseen objects, motion, occlusion, or other lighting. Component peaks can be affected by background yellow regions. This evaluates count recovery, not metric depth.",
    "target_centers_px": TARGETS,
    "match_radius_px": MATCH_RADIUS,
    "results": [],
}
for scene, video in VIDEOS:
    cap = cv2.VideoCapture(str(ROOT / video))
    frame_results = {cfg["name"]: {"exact": 0, "tp": 0, "fp": 0} for cfg in CONFIGS}
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        left = frame[:, :1600]
        hsv = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
        for cfg in CONFIGS:
            mask = cv2.inRange(hsv, (cfg["h"][0], cfg["s"], cfg["v"]), (cfg["h"][1], 255, 255))
            if cfg["morph"] == "open":
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            pts = candidates(mask, split=False)
            tp, fp = score(pts)
            stat = frame_results[cfg["name"]]
            stat["tp"] += tp
            stat["fp"] += fp
            stat["exact"] += tp == 2 and fp == 0
            split_pts = candidates(mask, split=True)
            tp, fp = score(split_pts)
            stat = frame_results[cfg["name"] + "_distance_split"] = frame_results.get(cfg["name"] + "_distance_split", {"exact": 0, "tp": 0, "fp": 0})
            stat["tp"] += tp
            stat["fp"] += fp
            stat["exact"] += tp == 2 and fp == 0
        frame_count += 1
    cap.release()
    summaries = []
    for name, stat in frame_results.items():
        summaries.append({"config": name, "frames": frame_count, **stat,
                          "exact_two_fraction": round(stat["exact"] / frame_count, 5),
                          "precision": round(stat["tp"] / (stat["tp"] + stat["fp"]), 5) if stat["tp"] + stat["fp"] else 0,
                          "recall": round(stat["tp"] / (2 * frame_count), 5)})
    summaries.sort(key=lambda x: (-x["exact_two_fraction"], x["fp"], -x["recall"]))
    out["results"].append({"scene": scene, "video": video, "frames": frame_count, "configs": summaries})
    print(scene, "frames", frame_count)
    for row in summaries:
        print(row)

output = ROOT / "media/frc-fuel-color-segmentation-sweep.json"
output.write_text(json.dumps(out, indent=2) + "\n")
print("saved", output)
