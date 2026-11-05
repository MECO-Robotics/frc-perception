#!/usr/bin/env python3
"""Stress-test Hough-refined centers with synthetic partial occlusion."""
import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "media/frc-fuel-dense-count-evaluation.json").read_text())
TARGETS = ((808.0, 589.0), (902.0, 591.0))
TIMES = tuple(1.0 + 0.4 * i for i in range(15))
PARAMS = (20, 25, 30)
MASKS = (("none", 0), ("bottom", 0.25), ("bottom", 0.50), ("bottom", 0.75),
         ("left", 0.25), ("left", 0.50), ("left", 0.75))


def match(detections):
    hits = [None, None]
    for box in sorted((b for b in detections if b["confidence"] >= 0.05),
                      key=lambda b: b["confidence"], reverse=True):
        center = ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)
        ds = [math.dist(center, target) for target in TARGETS]
        i = int(np.argmin(ds))
        if ds[i] <= 60 and hits[i] is None:
            hits[i] = box
    return hits


def sample(crop, box, p2):
    x1, y1, x2, y2 = map(int, (box["x1"], box["y1"], box["x2"], box["y2"]))
    cx, cy = (x1+x2)/2, (y1+y2)/2
    short = min(x2-x1, y2-y1)
    x0, y0 = max(0, int(round(cx-short*.8))), max(0, int(round(cy-short*.8)))
    x1c, y1c = min(1600, int(round(cx+short*.8))), min(1200, int(round(cy+short*.8)))
    roi = cv2.GaussianBlur(crop[y0:y1c, x0:x1c], (9, 9), 2)
    if not roi.size:
        return None
    rmin, rmax = max(8, int(short*.35)), max(9, int(short*.65))
    circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=max(10, int(short*.5)), param1=90,
                               param2=p2, minRadius=rmin, maxRadius=rmax)
    if circles is None:
        return None
    candidates = []
    for x, y, _r in circles[0]:
        point = (float(x+x0), float(y+y0))
        offset = math.dist(point, (cx, cy))
        if offset <= short*.30:
            candidates.append((offset, point))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def summary(errors):
    if not errors:
        return {"n": 0}
    return {"n": len(errors), "median_error_px": round(statistics.median(errors), 3),
            "p90_error_px": round(float(np.percentile(errors, 90)), 3)}


out = {
    "method": "At 15 stable timestamps per clip, black out 0/25/50/75% of the bottom or left part of each clean detector box in the left image. Keep the clean YOLO box fixed, run Hough on the masked detector crop, and compare center localization error to fixed manual centers. This isolates center refinement under partial image loss; it does not simulate a physical foreground object's stereo parallax.",
    "limitations": "Synthetic black masks, clean fixed detector boxes, static camera and targets. No moving pieces or real occluders; labels are manually fixed.",
    "sample_times_s": TIMES,
    "hough_param2": PARAMS,
    "results": [],
}

for clip in DATA["results"]:
    cap = cv2.VideoCapture(str(ROOT / clip["video"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    accum = {}
    for orientation, fraction in MASKS:
        key = f"{orientation}_{fraction:.2f}"
        accum[key] = {"box": [[], []], "hough": {p: [[], []] for p in PARAMS},
                      "matched_boxes": [0, 0]}
    rows = []
    for t in TIMES:
        idx = round(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame[:, :1600], cv2.COLOR_BGR2GRAY)
        boxes = match(clip["per_frame"][idx]["detections"])
        for target, box in enumerate(boxes):
            if box is None:
                continue
            bx, by = (box["x1"]+box["x2"])/2, (box["y1"]+box["y2"])/2
            accum["none_0.00"]["matched_boxes"][target] += 1
            accum["none_0.00"]["box"][target].append(math.dist((bx, by), TARGETS[target]))
            x1, y1, x2, y2 = map(int, (box["x1"], box["y1"], box["x2"], box["y2"]))
            for orientation, fraction in MASKS:
                key = f"{orientation}_{fraction:.2f}"
                masked = gray.copy()
                if fraction:
                    if orientation == "bottom":
                        ya = max(0, y2 - max(1, round((y2-y1)*fraction)))
                        masked[ya:y2, x1:x2] = 0
                    else:
                        xa = max(0, x1 + max(1, round((x2-x1)*fraction)))
                        masked[y1:y2, x1:xa] = 0
                for p2 in PARAMS:
                    point = sample(masked, box, p2)
                    if point is not None:
                        accum[key]["hough"][p2][target].append(math.dist(point, TARGETS[target]))
        rows.append({"time_s": t, "frame": idx})
    cap.release()
    configs = []
    baseline_boxes = accum["none_0.00"]["box"]
    matched_counts = accum["none_0.00"]["matched_boxes"]
    for key, values in accum.items():
        orientation, fraction = key.rsplit("_", 1)
        configs.append({"orientation": orientation, "fraction": float(fraction),
                        "box_center_error_px": [summary(x) for x in baseline_boxes],
                        "hough": [{"param2": p,
                                   "center_error_px": [summary(x) for x in values["hough"][p]],
                                   "coverage": [round(len(values["hough"][p][i]) / matched_counts[i], 4) if matched_counts[i] else 0 for i in range(2)]}
                                  for p in PARAMS]})
    out["results"].append({"scene": clip["scene"], "timestamps_evaluated": len(rows), "configs": configs})
    print(clip["scene"])
    for cfg in configs:
        p25 = next(x for x in cfg["hough"] if x["param2"] == 25)
        print(cfg["orientation"], cfg["fraction"], "box", cfg["box_center_error_px"],
              "hough25", p25["center_error_px"], "coverage", p25["coverage"])

path = ROOT / "media/frc-fuel-hough-center-occlusion-sweep.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print("saved", path)
