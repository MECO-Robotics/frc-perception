#!/usr/bin/env python3
"""Evaluate a lower detector confidence floor only on dark saved-video frames."""
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scout.stereo import StereoDepth

VIDEOS = (
    ("earlier", "media/frc-2026-live-two-fuel-60fps-cfr.mp4"),
    ("corrected-settings", "media/frc-2026-fuel-camera-settings-fixed-60fps.mp4"),
)
DETECTIONS = json.loads((ROOT / "media/frc-fuel-dense-count-nms015.json").read_text())
TARGETS = ((808.0, 589.0), (902.0, 591.0))
BASE_CONF = 0.05
LOW_CONF = 0.02
HIGH_CONF = 0.20
TRIGGER = 3
LUMA_GATE = 20.0
os.environ.update({
    "SCOUT_STEREO_ENABLED": "1", "SCOUT_STEREO_RECTIFY": "1",
    "SCOUT_STEREO_CALIBRATION": "media/calibration/stereo_calibration_live_order_candidate.yml",
    "SCOUT_STEREO_BASELINE_M": "0.0635", "SCOUT_STEREO_DISPARITY_SCALE": "1",
    "SCOUT_STEREO_NUM_DISPARITIES": "128", "SCOUT_STEREO_BLOCK_SIZE": "5",
    "SCOUT_STEREO_MATCH_SCALE": "0.5", "SCOUT_STEREO_VERTICAL_ROI_PAD": "24",
    "SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR": "0.55",
    "SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU": "0.90", "SCOUT_FUEL_DIAMETER_M": "0.15",
})
stereo = StereoDepth()


def boxes_to_objects(rows):
    return [SimpleNamespace(
        x1=row["x1"], y1=row["y1"], x2=row["x2"], y2=row["y2"],
        class_id=row.get("class_id", 0), confidence=row["confidence"],
        distance_m=None, disparity_px=None) for row in rows]


def run_stereo(left, right, rows):
    boxes = boxes_to_objects(rows)
    if not boxes:
        return []
    stereo.estimate(left, right, boxes)
    boxes = stereo.filter_fuel_size_inconsistent(boxes)
    return [{"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
             "confidence": float(b.confidence), "distance_m": b.distance_m}
            for b in boxes]


def score(rows):
    hits = [None, None]
    extras = 0
    for box in sorted(rows, key=lambda row: -row["confidence"]):
        center = ((box["x1"] + box["x2"]) / 2,
                  (box["y1"] + box["y2"]) / 2)
        error, index = min((math.dist(center, target), i)
                           for i, target in enumerate(TARGETS))
        if error <= 60 and hits[index] is None:
            hits[index] = box
        else:
            extras += 1
    return hits, extras


def metrics(records, key):
    paired = exact = tp = fp = fn = 0
    ranges = [[], []]
    for row in records:
        hits, extras = score(row[key])
        found = sum(hit is not None for hit in hits)
        tp += found
        fn += 2 - found
        fp += extras
        paired += found == 2
        exact += found == 2 and extras == 0
        for index, hit in enumerate(hits):
            if hit is not None and hit["distance_m"] is not None:
                ranges[index].append(hit["distance_m"])
    return {"frames": len(records), "paired_frames": paired,
            "exact_two_frames": exact, "tp": tp, "fp": fp, "fn": fn,
            "range_by_target_m": [
                {"n": len(values), "median": round(statistics.median(values), 5),
                 "mad": round(statistics.median(abs(v - statistics.median(values))
                                                for v in values), 5)}
                if values else {"n": 0, "median": None, "mad": None}
                for values in ranges]}


output = {
    "method": ("Compare the active detector floor 0.05 plus the 3-candidate/0.20 "
               "adaptive cutoff against a conditional 0.02 floor when left-image "
               "mean luma <20. The adaptive trigger still counts candidates at "
               "0.05. Both paths use rectified SGBM b5/u8, 63.5 mm baseline, "
               "24-pixel vertical ROI, and the active 55% size/stereo gate."),
    "settings": {"base_confidence": BASE_CONF, "dark_confidence": LOW_CONF,
                 "adaptive_confidence": HIGH_CONF, "trigger": TRIGGER,
                 "luma_gate": LUMA_GATE, "nms_iou": 0.15,
                 "stereo_baseline_m": 0.0635, "stereo_size_gate": 0.55},
    "limitations": ("Two stationary indoor recordings; the earlier clip has a "
                    "brief dark opening. No motion, dense unrelated objects, or "
                    "tape-measured distance is available. Low-confidence recovery "
                    "may behave differently under field clutter."),
    "results": [],
}

for clip in DETECTIONS["results"]:
    scene = clip["scene"]
    capture = cv2.VideoCapture(str(ROOT / clip["video"]))
    rows = []
    started = time.perf_counter()
    for index, detrow in enumerate(clip["per_frame"]):
        ok, frame = capture.read()
        if not ok:
            break
        left, right = frame[:, :1600], frame[:, 1600:]
        luma = float(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).mean())
        candidates = detrow["detections"]
        n_base = [b for b in candidates if b["confidence"] >= BASE_CONF]
        adaptive_cutoff = HIGH_CONF if len(n_base) >= TRIGGER else BASE_CONF
        baseline = [b for b in n_base if b["confidence"] >= adaptive_cutoff]
        dark_floor = LOW_CONF if luma < LUMA_GATE else BASE_CONF
        n_dark = [b for b in candidates if b["confidence"] >= dark_floor]
        dark_cutoff = HIGH_CONF if len(n_base) >= TRIGGER else dark_floor
        candidate = [b for b in n_dark if b["confidence"] >= dark_cutoff]
        baseline_output = run_stereo(left, right, baseline)
        candidate_output = (baseline_output if candidate == baseline else
                            run_stereo(left, right, candidate))
        rows.append({"frame": index, "mean_luma": round(luma, 3),
                     "dark_floor_used": luma < LUMA_GATE,
                     "baseline": baseline_output, "dark_floor": candidate_output})
    capture.release()
    result = {"scene": scene, "video": clip["video"], "frames": len(rows),
              "elapsed_s": round(time.perf_counter() - started, 2),
              "dark_floor_frames": sum(row["dark_floor_used"] for row in rows),
              "baseline": metrics(rows, "baseline"),
              "dark_confidence_floor": metrics(rows, "dark_floor"),
              "per_frame": rows}
    output["results"].append(result)
    print(scene, json.dumps({k: v for k, v in result.items() if k != "per_frame"}))

path = ROOT / "media/frc-fuel-dark-confidence-floor.json"
path.write_text(json.dumps(output, indent=2) + "\n")
print("saved", path)
