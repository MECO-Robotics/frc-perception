#!/usr/bin/env python3
"""Replay FRC YOLO26n below the existing 0.05 cutoff on both saved clips."""
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scout.detectors import TensorRTDetector
ENGINE = ROOT / "models/rebuilt_fuel_yolo26n_1280x960_fp16.engine"
VIDEOS = (("earlier", "media/frc-2026-live-two-fuel-60fps-cfr.mp4"),
          ("corrected-settings", "media/frc-2026-fuel-camera-settings-fixed-60fps.mp4"))
TARGETS = ((808, 589), (902, 591))
MATCH = 60.0
SWEEP = (.01, .02, .03, .05, .075, .10)


def score(rows, threshold):
    tp = fp = fn = exact = 0
    for row in rows:
        hits = [None, None]
        extras = 0
        for b in sorted((x for x in row if x["confidence"] >= threshold),
                        key=lambda x: x["confidence"], reverse=True):
            x, y = (b["x1"]+b["x2"])/2, (b["y1"]+b["y2"])/2
            dist, idx = min((math.hypot(x-tx, y-ty), i)
                            for i, (tx, ty) in enumerate(TARGETS))
            if dist <= MATCH and hits[idx] is None:
                hits[idx] = b
            else:
                extras += 1
        count = sum(x is not None for x in hits)
        tp += count; fn += 2-count; fp += extras
        exact += count == 2 and extras == 0
    return {"confidence": threshold, "tp": tp, "fp": fp, "fn": fn,
            "exact_two_frames": exact,
            "precision": round(tp/(tp+fp), 5) if tp+fp else 0,
            "recall": round(tp/(tp+fn), 5) if tp+fn else 0}


os.environ["SCOUT_CLASS_NAMES"] = '["ball"]'
os.environ["SCOUT_MAX_BOX_ASPECT_RATIO"] = "2.0"
detector = TensorRTDetector(str(ENGINE), confidence=.005)
out = {"method": "Run the active 1280x960 FP16 TensorRT engine at 0.005 inference cutoff, then score confidence cutoffs 0.01-0.10 against two fixed manual centers.",
       "limitations": "Stationary targets only; detector output is limited to its end-to-end 300-row head. Low-confidence candidates can increase false positives outside this view.",
       "results": []}
try:
    for scene, relative in VIDEOS:
        cap = cv2.VideoCapture(str(ROOT / relative))
        rows, latency = [], []
        start = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes, ms = detector.infer(frame[:, :1600])
            latency.append(ms)
            rows.append([{"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                          "confidence": b.confidence} for b in boxes])
        cap.release()
        result = {"scene": scene, "video": relative, "frames": len(rows),
                  "elapsed_s": round(time.perf_counter()-start, 2),
                  "inference_ms_median": round(statistics.median(latency), 2),
                  "threshold_sweep": [score(rows, c) for c in SWEEP],
                  "per_frame": rows}
        out["results"].append(result)
        print(scene, result["frames"], result["inference_ms_median"], result["threshold_sweep"])
finally:
    detector.close()
path = ROOT / "media/frc-fuel-low-confidence-evaluation.json"
path.write_text(json.dumps(out, indent=2)+"\n")
print("saved", path)
