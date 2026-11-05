#!/usr/bin/env python3
"""Sweep simple confidence gates for Hough-refined FUEL centers."""
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
MASKS = (("none", 0), ("bottom", .25), ("bottom", .50), ("bottom", .75),
         ("left", .25), ("left", .50), ("left", .75))
CONSENSUS_PX = (2.0, 3.0, 4.0, 6.0, 8.0)
RADIUS_RANGES = ((0.35, 0.65), (0.40, 0.60), (0.45, 0.55))


def match(detections):
    hits = [None, None]
    for b in sorted((b for b in detections if b["confidence"] >= .05), key=lambda b: -b["confidence"]):
        center = ((b["x1"]+b["x2"])/2, (b["y1"]+b["y2"])/2)
        ds = [math.dist(center, t) for t in TARGETS]
        i = int(np.argmin(ds))
        if ds[i] <= 60 and hits[i] is None:
            hits[i] = b
    return hits


def circles_for(gray, box, p2):
    x1, y1, x2, y2 = [int(box[k]) for k in ("x1", "y1", "x2", "y2")]
    cx, cy = (x1+x2)/2, (y1+y2)/2
    short = min(x2-x1, y2-y1)
    x0, y0 = max(0, int(round(cx-short*.8))), max(0, int(round(cy-short*.8)))
    xe, ye = min(1600, int(round(cx+short*.8))), min(1200, int(round(cy+short*.8)))
    crop = cv2.GaussianBlur(gray[y0:ye, x0:xe], (9, 9), 2)
    rmin, rmax = max(8, int(short*.35)), max(9, int(short*.65))
    found = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, dp=1.2,
                             minDist=max(10, int(short*.5)), param1=90,
                             param2=p2, minRadius=rmin, maxRadius=rmax) if crop.size else None
    if found is None:
        return None
    choices = []
    for x, y, radius in found[0]:
        center = (float(x+x0), float(y+y0))
        offset = math.dist(center, (cx, cy))
        if offset <= short*.30:
            choices.append((offset, center, float(radius)))
    if not choices:
        return None
    offset, center, radius = min(choices, key=lambda a: a[0])
    return {"center": center, "radius_ratio": radius/short,
            "offset_ratio": offset/short}


def qstats(errors):
    if not errors:
        return {"n": 0}
    return {"n": len(errors), "median_px": round(statistics.median(errors), 3),
            "p90_px": round(float(np.percentile(errors, 90)), 3)}


def evaluate(samples, consensus_px=None, radius_range=None, use_consensus=False, always_box=False):
    errors, selected = [], 0
    per_mask = {}
    for sample in samples:
        use_hough = False
        p25 = sample["fits"].get("25")
        if p25 is not None and not always_box:
            if use_consensus:
                centers = [sample["fits"].get(str(p), {}).get("center") if sample["fits"].get(str(p)) else None for p in PARAMS]
                if all(c is not None for c in centers):
                    spread = max(math.dist(a, b) for i, a in enumerate(centers) for b in centers[i+1:])
                    use_hough = spread <= consensus_px
            else:
                use_hough = True
            if radius_range is not None:
                use_hough = use_hough and radius_range[0] <= p25["radius_ratio"] <= radius_range[1]
        center = p25["center"] if use_hough else sample["box_center"]
        error = math.dist(center, sample["target_center"])
        errors.append(error)
        selected += use_hough
        per_mask.setdefault(sample["mask"], []).append(error)
    return {"median_px": round(statistics.median(errors), 3),
            "p90_px": round(float(np.percentile(errors, 90)), 3),
            "hough_selected_fraction": round(selected/len(samples), 4),
            "by_mask": {k: qstats(v) for k, v in per_mask.items()}}


out = {
    "method": "On the same 15 timestamps per clip and 7 synthetic mask conditions as the Hough-center occlusion sweep, collect circle centers/radius ratios at param2 20/25/30. Compare unconditional param25 against box-center fallback when the three fits disagree or the param25 radius ratio is outside a candidate band. Score center error against manual image centers.",
    "limitations": "Small stationary synthetic-mask sample; manually labeled centers; not physical occlusion and does not include detector misses. Thresholds are exploratory and must be validated on real moving/occluded frames.",
    "radius_ratio_bands": RADIUS_RANGES,
    "consensus_spread_thresholds_px": CONSENSUS_PX,
    "results": [],
}

for clip in DATA["results"]:
    cap = cv2.VideoCapture(str(ROOT / clip["video"]))
    samples = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    for t in TIMES:
        idx = round(t*fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame[:, :1600], cv2.COLOR_BGR2GRAY)
        boxes = match(clip["per_frame"][idx]["detections"])
        for target, box in enumerate(boxes):
            if box is None:
                continue
            x1, y1, x2, y2 = [int(box[k]) for k in ("x1", "y1", "x2", "y2")]
            box_center = ((x1+x2)/2, (y1+y2)/2)
            for orientation, fraction in MASKS:
                masked = gray.copy()
                if fraction:
                    if orientation == "bottom":
                        ya = max(0, y2-max(1, round((y2-y1)*fraction)))
                        masked[ya:y2, x1:x2] = 0
                    else:
                        xe = max(0, x1+max(1, round((x2-x1)*fraction)))
                        masked[y1:y2, x1:xe] = 0
                fits = {str(p): circles_for(masked, box, p) for p in PARAMS}
                samples.append({"mask": f"{orientation}_{fraction:.2f}",
                                "target_center": TARGETS[target],
                                "box_center": box_center, "fits": fits})
    cap.release()
    base = {"box_only": evaluate(samples, always_box=True), "hough_p25_always": evaluate(samples, use_consensus=False)}
    gates = []
    for lo, hi in RADIUS_RANGES:
        gates.append({"type": "radius_ratio", "band": [lo, hi],
                      **evaluate(samples, radius_range=(lo, hi))})
    for threshold in CONSENSUS_PX:
        gates.append({"type": "param2_consensus", "max_spread_px": threshold,
                      **evaluate(samples, consensus_px=threshold, use_consensus=True)})
    for threshold in CONSENSUS_PX:
        for lo, hi in ((.35, .65), (.40, .60)):
            gates.append({"type": "consensus_and_radius", "max_spread_px": threshold,
                          "radius_band": [lo, hi],
                          **evaluate(samples, consensus_px=threshold, radius_range=(lo, hi), use_consensus=True)})
    result = {"scene": clip["scene"], "sample_count": len(samples), "baseline": base, "gates": gates}
    out["results"].append(result)
    print(clip["scene"], "baseline", json.dumps(base))
    for gate in sorted(gates, key=lambda g: (g["median_px"], g["p90_px"]))[:5]:
        print("best_gate", json.dumps({k: v for k, v in gate.items() if k != "by_mask"}))

path = ROOT / "media/frc-fuel-hough-center-fallback-sweep.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print("saved", path)
