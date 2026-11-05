#!/usr/bin/env python3
"""Compare FUEL center estimates inside YOLO boxes against fixed labeled centers."""
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "media/frc-fuel-dense-count-evaluation.json").read_text())
TARGETS = ((808.0, 589.0), (902.0, 591.0))
HSV = ((40, 40), (70, 40), (100, 40), (70, 80), (100, 80))  # saturation, value
HOUGH_PARAMS = (20, 25, 30)


def matched(detections):
    out = [None, None]
    for b in sorted((b for b in detections if b["confidence"] >= 0.05),
                    key=lambda b: b["confidence"], reverse=True):
        cx = (b["x1"] + b["x2"]) / 2
        cy = (b["y1"] + b["y2"]) / 2
        distances = [math.hypot(cx - x, cy - y) for x, y in TARGETS]
        i = int(np.argmin(distances))
        if distances[i] <= 60 and out[i] is None:
            out[i] = b
    return out


def summarize(samples):
    if not samples:
        return {"n": 0}
    errors = [x["error_px"] for x in samples]
    mx = statistics.median(x["x"] for x in samples)
    my = statistics.median(x["y"] for x in samples)
    med_error = statistics.median(errors)
    return {
        "n": len(samples),
        "median_error_px": round(med_error, 3),
        "p90_error_px": round(float(np.percentile(errors, 90)), 3),
        "median_dx_px": round(statistics.median(x["dx"] for x in samples), 3),
        "median_dy_px": round(statistics.median(x["dy"] for x in samples), 3),
        "median_center_x": round(mx, 3),
        "median_center_y": round(my, 3),
        "center_mad_radial_px": round(statistics.median(math.hypot(x["x"] - mx, x["y"] - my) for x in samples), 3),
    }


def add_sample(bucket, center, target):
    if center is None:
        return
    x, y = center
    tx, ty = TARGETS[target]
    bucket.append({"x": float(x), "y": float(y), "dx": float(x - tx),
                   "dy": float(y - ty), "error_px": math.hypot(x - tx, y - ty)})


out = {
    "method": "Compare detector-box center, HSV mask centroid (all mask pixels and largest connected component), and Hough circle center inside detector-centered crop. Match detections at confidence >=0.05 to two manually labeled fixed centers with a 60 px gate. Report error against these labels and temporal center spread.",
    "limitations": "Fixed camera and two stationary targets; manual target centers; no tests under motion, occlusion, other distances, or view angles. Hough and HSV centers are calculated only after the learned detector localizes a box, so this is not a standalone detector evaluation.",
    "target_centers_px": TARGETS,
    "hsv_saturation_value_thresholds": [{"s_min": s, "v_min": v} for s, v in HSV],
    "hough_param2": HOUGH_PARAMS,
    "results": [],
}

for clip in DATA["results"]:
    cap = cv2.VideoCapture(str(ROOT / clip["video"]))
    buckets = {}
    for method in ["box_center"] + [f"hsv_all_s{s}_v{v}" for s, v in HSV] + [f"hsv_largest_s{s}_v{v}" for s, v in HSV] + [f"hough_p{p}" for p in HOUGH_PARAMS]:
        buckets[method] = [[], []]
    matched_counts = [0, 0]
    hough_times = []
    start = time.perf_counter()
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame[:, :1600], cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame[:, :1600], cv2.COLOR_BGR2HSV)
        boxes = matched(clip["per_frame"][frame_index]["detections"])
        for target, box in enumerate(boxes):
            if box is None:
                continue
            matched_counts[target] += 1
            x1, y1, x2, y2 = map(int, (box["x1"], box["y1"], box["x2"], box["y2"]))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(1600, x2), min(1200, y2)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            add_sample(buckets["box_center"][target], (cx, cy), target)
            crop_hsv = hsv[y1:y2, x1:x2]
            for s_min, v_min in HSV:
                mask = cv2.inRange(crop_hsv, (10, s_min, v_min), (40, 255, 255))
                yy, xx = np.nonzero(mask)
                name = f"hsv_all_s{s_min}_v{v_min}"
                if len(xx) >= 20:
                    add_sample(buckets[name][target], (x1 + xx.mean(), y1 + yy.mean()), target)
                n, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
                if n > 1:
                    candidates = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 20]
                    if candidates:
                        i = max(candidates, key=lambda k: stats[k, cv2.CC_STAT_AREA])
                        name = f"hsv_largest_s{s_min}_v{v_min}"
                        add_sample(buckets[name][target], (x1 + centers[i, 0], y1 + centers[i, 1]), target)
            short = min(x2 - x1, y2 - y1)
            if short < 10:
                continue
            rx0, ry0 = max(0, int(round(cx - short * .8))), max(0, int(round(cy - short * .8)))
            rx1, ry1 = min(1600, int(round(cx + short * .8))), min(1200, int(round(cy + short * .8)))
            crop = cv2.GaussianBlur(gray[ry0:ry1, rx0:rx1], (9, 9), 2)
            if crop.size == 0:
                continue
            radius_min = max(8, int(short * .35))
            radius_max = max(radius_min + 1, int(short * .65))
            for p2 in HOUGH_PARAMS:
                t0 = time.perf_counter()
                circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, dp=1.2,
                                           minDist=max(10, int(short * .5)),
                                           param1=90, param2=p2,
                                           minRadius=radius_min, maxRadius=radius_max)
                hough_times.append((time.perf_counter() - t0) * 1000)
                if circles is None:
                    continue
                choices = []
                for x, y, _radius in circles[0]:
                    absolute = (float(x + rx0), float(y + ry0))
                    offset = math.hypot(absolute[0] - cx, absolute[1] - cy)
                    if offset <= short * .30:
                        choices.append((offset, absolute))
                if choices:
                    _, center = min(choices, key=lambda item: item[0])
                    add_sample(buckets[f"hough_p{p2}"][target], center, target)
        frame_index += 1
    cap.release()
    methods = []
    for name, targets in buckets.items():
        methods.append({"method": name, "targets": [summarize(samples) for samples in targets]})
    methods.sort(key=lambda x: (sum(t.get("median_error_px", 1e9) for t in x["targets"] if t.get("n")), x["method"]))
    entry = {"scene": clip["scene"], "frames": frame_index,
             "matched_detections_by_target": matched_counts,
             "hough_latency_ms_median": round(statistics.median(hough_times), 3) if hough_times else None,
             "elapsed_s": round(time.perf_counter() - start, 2), "methods": methods}
    out["results"].append(entry)
    print(clip["scene"], "matched", matched_counts)
    for row in methods[:6]:
        print(row)

path = ROOT / "media/frc-fuel-center-localization-sweep.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print("saved", path)
