#!/usr/bin/env python3
"""Evaluate per-frame cross-resolution detector fusion on saved FUEL clips."""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {s: json.loads((ROOT / f"media/frc-fuel-dense-count-{s}.json").read_text())
         for s in ("640", "1280", "1600")}
TARGETS = ((808, 589), (902, 591))
GATE = 60.0


def iou(a, b):
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0, x2-x1) * max(0, y2-y1)
    aa = (a["x2"]-a["x1"]) * (a["y2"]-a["y1"])
    bb = (b["x2"]-b["x1"]) * (b["y2"]-b["y1"])
    return inter / (aa+bb-inter) if aa+bb > inter else 0.0


def fuse(boxes, mode, threshold):
    ordered = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
    out = []
    while ordered:
        seed = ordered.pop(0)
        group = [b for b in ordered if iou(seed, b) >= threshold]
        ordered = [b for b in ordered if iou(seed, b) < threshold]
        if mode == "nms" or not group:
            out.append(seed)
        else:
            group.append(seed)
            weights = [max(1e-6, b["confidence"]) for b in group]
            total = sum(weights)
            box = {k: sum(b[k]*w for b, w in zip(group, weights))/total
                   for k in ("x1", "y1", "x2", "y2")}
            box["confidence"] = max(b["confidence"] for b in group)
            out.append(box)
    return out


def score(boxes):
    hits = [None, None]
    extras = 0
    for b in sorted(boxes, key=lambda b: b["confidence"], reverse=True):
        x, y = (b["x1"]+b["x2"])/2, (b["y1"]+b["y2"])/2
        dist, idx = min((math.hypot(x-tx, y-ty), i) for i, (tx, ty) in enumerate(TARGETS))
        if dist <= GATE and hits[idx] is None:
            hits[idx] = (b, dist)
        else:
            extras += 1
    return hits, extras


def evaluate(scene, pair, conf, mode, overlap):
    sources = [FILES[s]["results"][[r["scene"] for r in FILES[s]["results"]].index(scene)]
               for s in pair]
    rows = []
    for i in range(600):
        boxes = [b for src in sources for b in src["per_frame"][i]["detections"]
                 if b["confidence"] >= conf]
        rows.append(fuse(boxes, mode, overlap))
    tp = fp = fn = exact = 0
    errors = [[], []]
    for boxes in rows:
        hits, extras = score(boxes)
        count = sum(h is not None for h in hits)
        tp += count; fn += 2-count; fp += extras
        exact += count == 2 and extras == 0
        for j, hit in enumerate(hits):
            if hit:
                errors[j].append(hit[1])
    return {"sources": pair, "confidence": conf, "fusion": mode,
            "iou_threshold": overlap, "exact_two_frames": exact,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp/(tp+fp), 5) if tp+fp else 0,
            "recall": round(tp/(tp+fn), 5) if tp+fn else 0,
            "center_error_px_by_target": [
                {"n": len(v), "median": round(statistics.median(v), 3),
                 "p95": round(sorted(v)[int(.95*(len(v)-1))], 3)} if v else None
                for v in errors]}


out = {"method": "Fuse same-frame detections from separately evaluated 640/1280/1600 YOLO26n engines. Cross-scale NMS or confidence-weighted box fusion; match centers to two fixed labels within 60 px.",
       "limitations": "Only two stationary adjacent balls in two clips. Scale ensembles cost the sum of their detector inference times; this does not test motion, other objects, or absolute depth accuracy.",
       "results": []}
for scene in ("earlier", "corrected-settings"):
    configs = [evaluate(scene, ["1280"], .05, "nms", .5)]
    for pair in (["1280", "640"], ["1280", "1600"]):
        for conf in (.05, .10):
            for mode in ("nms", "wbf"):
                for overlap in (.3, .5, .7):
                    configs.append(evaluate(scene, pair, conf, mode, overlap))
    out["results"].append({"scene": scene, "configs": configs})
    print(scene)
    for row in configs:
        print(row)
path = ROOT / "media/frc-fuel-resolution-ensemble-sweep.json"
path.write_text(json.dumps(out, indent=2)+"\n")
print("saved", path)
