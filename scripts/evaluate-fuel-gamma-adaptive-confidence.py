#!/usr/bin/env python3
"""Check whether dark-frame gamma helps with the active adaptive filter."""
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scout.detectors import TensorRTDetector
from scout.stereo import StereoDepth

VIDEOS = (
    ("earlier", "media/frc-2026-live-two-fuel-60fps-cfr.mp4"),
    ("corrected-settings", "media/frc-2026-fuel-camera-settings-fixed-60fps.mp4"),
)
BASE = json.loads((ROOT / "media/frc-fuel-dense-count-nms015.json").read_text())
TARGETS = ((808.0, 589.0), (902.0, 591.0))
GAMMA = 0.60
LUMA_GATE = 20.0
BASE_CONF = 0.05
HIGH_CONF = 0.20
TRIGGER = 3
NMS = 0.15
LUT = np.clip((np.arange(256, dtype=np.float32) / 255.0) ** GAMMA * 255,
              0, 255).astype(np.uint8)


def score(boxes, min_conf=BASE_CONF):
    hits = [None, None]
    extras = 0
    for box in sorted((b for b in boxes if b["confidence"] >= min_conf),
                      key=lambda b: -b["confidence"]):
        cx = (box["x1"] + box["x2"]) / 2
        cy = (box["y1"] + box["y2"]) / 2
        distance, index = min((math.dist((cx, cy), target), i)
                              for i, target in enumerate(TARGETS))
        if distance <= 60 and hits[index] is None:
            hits[index] = box
        else:
            extras += 1
    return hits, extras


def apply_adaptive(boxes, floor=BASE_CONF):
    candidates = [b for b in boxes if b["confidence"] >= BASE_CONF]
    cutoff = HIGH_CONF if len(candidates) >= TRIGGER else floor
    return [b for b in boxes if b["confidence"] >= cutoff], cutoff


def summarize(records, key):
    exact = paired = tp = fp = fn = gamma_frames = 0
    changed_ranges = [[], []]
    for row in records:
        boxes = row[key]
        threshold = 0.02 if key == "gamma_low_floor" else BASE_CONF
        hits, extras = score(boxes, threshold)
        n = sum(hit is not None for hit in hits)
        tp += n
        fn += 2 - n
        fp += extras
        paired += n == 2
        exact += n == 2 and extras == 0
        gamma_frames += row["gamma_used"] if key in (
            "gamma_plus_adaptive", "gamma_low_floor") else False
        for target, hit in enumerate(hits):
            if hit and hit.get("distance_m") is not None:
                changed_ranges[target].append(hit["distance_m"])
    return {"frames": len(records), "gamma_frames": gamma_frames,
            "paired_frames": paired, "exact_two_frames": exact,
            "tp": tp, "fp": fp, "fn": fn,
            "range_medians_m": [round(statistics.median(values), 5)
                                if values else None for values in changed_ranges]}


def as_box(item):
    return SimpleNamespace(x1=item["x1"], y1=item["y1"],
                           x2=item["x2"], y2=item["y2"],
                           confidence=item["confidence"],
                           class_id=item.get("class_id", 0),
                           distance_m=None, disparity_px=None)


def serialize(box):
    return {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2,
            "confidence": float(box.confidence),
            "distance_m": getattr(box, "distance_m", None)}


def stereo_gate(left, right, rows):
    boxes = [as_box(item) for item in rows]
    if not boxes:
        return []
    stereo.estimate(left, right, boxes)
    boxes = stereo.filter_fuel_size_inconsistent(boxes)
    return [serialize(box) for box in boxes]


os.environ.update({
    "SCOUT_CLASS_NAMES": '["ball"]', "SCOUT_MAX_BOX_ASPECT_RATIO": "2.0",
    "SCOUT_NMS_IOU": str(NMS),
    "SCOUT_STEREO_ENABLED": "1", "SCOUT_STEREO_RECTIFY": "1",
    "SCOUT_STEREO_CALIBRATION": "media/calibration/stereo_calibration_live_order_candidate.yml",
    "SCOUT_STEREO_BASELINE_M": "0.0635",
    "SCOUT_STEREO_DISPARITY_SCALE": "1", "SCOUT_STEREO_BLOCK_SIZE": "5",
    "SCOUT_STEREO_VERTICAL_ROI_PAD": "24", "SCOUT_STEREO_MATCH_SCALE": "0.5",
    "SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR": "0.55",
    "SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU": "0.90",
    "SCOUT_FUEL_DIAMETER_M": "0.15",
})
detector = TensorRTDetector(
    str(ROOT / "models/rebuilt_fuel_yolo26n_1280x960_fp16.engine"),
    confidence=0.02, iou=NMS)
stereo = StereoDepth()
output = {
    "method": ("Compare raw detections with the deployed candidate-count adaptive "
               "confidence rule (3 candidates at 0.05 => cutoff 0.20), then "
               "apply gamma 0.60 only when full left-image mean luma <20 before "
               "the same adaptive rule. On gamma frames, recompute SGBM and apply "
               "the active 55% apparent-size gate and duplicate suppression."),
    "settings": {"gamma": GAMMA, "luma_gate": LUMA_GATE,
                 "base_confidence": BASE_CONF, "adaptive_trigger": TRIGGER,
                 "adaptive_confidence": HIGH_CONF, "nms_iou": NMS},
    "limitations": ("Two stationary indoor recordings; the earlier clip has a "
                    "brief dark opening. Low-light synthetic correction cannot "
                    "recreate exposure, noise, motion blur, or field lighting. "
                    "Stereo distances lack tape-measured ground truth."),
    "results": [],
}

try:
    for clip_index, (scene, video_rel) in enumerate(VIDEOS):
        base_clip = BASE["results"][clip_index]
        cap = cv2.VideoCapture(str(ROOT / video_rel))
        records = []
        index = 0
        started = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            left, right = frame[:, :1600], frame[:, 1600:]
            raw = base_clip["per_frame"][index]["detections"]
            luma = float(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).mean())
            use_gamma = luma < LUMA_GATE
            selected = raw
            if use_gamma:
                candidates, _ = detector.infer(cv2.LUT(left, LUT))
                selected = [{"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                             "confidence": float(b.confidence), "class_id": b.class_id}
                            for b in candidates]
            adaptive, cutoff = apply_adaptive(selected)
            adaptive_low_floor, low_cutoff = apply_adaptive(selected, floor=0.02)
            raw_fixed = [b for b in raw if b["confidence"] >= BASE_CONF]
            raw_adaptive, _ = apply_adaptive(raw)
            raw_fixed_gated = stereo_gate(left, right, raw_fixed)
            raw_adaptive_gated = (raw_fixed_gated if raw_fixed == raw_adaptive else
                                  stereo_gate(left, right, raw_adaptive))
            combined_gated = (raw_adaptive_gated if not use_gamma else
                              stereo_gate(left, right, adaptive))
            combined_low_gated = (raw_adaptive_gated if not use_gamma else
                                  stereo_gate(left, right, adaptive_low_floor))
            records.append({"frame": index, "mean_luma": round(luma, 3),
                            "gamma_used": use_gamma, "adaptive_cutoff": cutoff,
                            "gamma_low_floor_cutoff": low_cutoff,
                            "raw_conf005": raw_fixed_gated,
                            "adaptive_only": raw_adaptive_gated,
                            "gamma_plus_adaptive": combined_gated,
                            "gamma_low_floor": combined_low_gated})
            index += 1
        cap.release()
        result = {"scene": scene, "video": video_rel, "frames": index,
                  "elapsed_s": round(time.perf_counter() - started, 2),
                  "raw_conf005": summarize(records, "raw_conf005"),
                  "adaptive_only": summarize(records, "adaptive_only"),
                  "gamma_plus_adaptive": summarize(records, "gamma_plus_adaptive"),
                  "gamma_plus_low_floor": summarize(records, "gamma_low_floor"),
                  "gamma_used_frames": sum(row["gamma_used"] for row in records),
                  "per_frame": records}
        output["results"].append(result)
        print(scene, json.dumps({k: v for k, v in result.items() if k != "per_frame"}))
finally:
    detector.close()

path = ROOT / "media/frc-fuel-gamma-adaptive-confidence.json"
path.write_text(json.dumps(output, indent=2) + "\n")
print("saved", path)
