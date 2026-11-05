#!/usr/bin/env python3
"""Sweep low-light preprocessing, confidence floor, and trigger on both FUEL clips."""
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
GAMMAS = (0.55, 0.60, 0.65)
LUMA_GATES = (15.0, 20.0, 25.0, 30.0)
FLOORS = (0.02, 0.05)
BASE_CONF, HIGH_CONF, TRIGGER, NMS = 0.05, 0.20, 3, 0.15
OUTPUT_CAPABILITY_LUMA = max(LUMA_GATES)

os.environ.update({
    "SCOUT_CLASS_NAMES": '["ball"]', "SCOUT_MAX_BOX_ASPECT_RATIO": "2.0",
    "SCOUT_NMS_IOU": str(NMS),
    "SCOUT_STEREO_ENABLED": "1", "SCOUT_STEREO_RECTIFY": "1",
    "SCOUT_STEREO_CALIBRATION": "media/calibration/stereo_calibration_live_order_candidate.yml",
    "SCOUT_STEREO_BASELINE_M": "0.0635", "SCOUT_STEREO_DISPARITY_SCALE": "1",
    "SCOUT_STEREO_NUM_DISPARITIES": "128", "SCOUT_STEREO_BLOCK_SIZE": "5",
    "SCOUT_STEREO_MATCH_SCALE": "0.5", "SCOUT_STEREO_VERTICAL_ROI_PAD": "24",
    "SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR": "0.55",
    "SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU": "0.90", "SCOUT_FUEL_DIAMETER_M": "0.15",
})
detector = TensorRTDetector(
    str(ROOT / "models/rebuilt_fuel_yolo26n_1280x960_fp16.engine"),
    confidence=0.02, iou=NMS)
stereo = StereoDepth()


def adapt(boxes, floor):
    at_base = sum(box["confidence"] >= BASE_CONF for box in boxes)
    cutoff = HIGH_CONF if at_base >= TRIGGER else floor
    return [box for box in boxes if box["confidence"] >= cutoff]


def run_stereo(left, right, boxes):
    items = [SimpleNamespace(x1=int(b["x1"]), y1=int(b["y1"]),
                             x2=int(b["x2"]), y2=int(b["y2"]),
                             class_id=int(b.get("class_id", 0)),
                             confidence=float(b["confidence"]),
                             distance_m=None, disparity_px=None)
             for b in boxes]
    if not items:
        return []
    stereo.estimate(left, right, items)
    items = stereo.filter_fuel_size_inconsistent(items)
    return [{"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
             "confidence": float(b.confidence), "distance_m": b.distance_m}
            for b in items]


def score(boxes, threshold):
    hits, extras = [None, None], 0
    for box in sorted((b for b in boxes if b["confidence"] >= threshold),
                      key=lambda b: -b["confidence"]):
        center = ((box["x1"] + box["x2"]) / 2,
                  (box["y1"] + box["y2"]) / 2)
        error, index = min((math.dist(center, point), i)
                           for i, point in enumerate(TARGETS))
        if error <= 60 and hits[index] is None:
            hits[index] = box
        else:
            extras += 1
    return hits, extras


def summarize(frames, threshold):
    tp = fp = fn = paired = exact = 0
    ranges = [[], []]
    pair_distances = []
    for item in frames:
        hits, extras = score(item["boxes"], threshold)
        found = sum(hit is not None for hit in hits)
        tp += found
        fn += 2 - found
        fp += extras
        paired += found == 2
        exact += found == 2 and extras == 0
        for i, hit in enumerate(hits):
            if hit and hit.get("distance_m") is not None:
                ranges[i].append(hit["distance_m"])
        if all(hit and hit.get("distance_m") for hit in hits):
            pair_distances.append(item["pair_distance_m"])
    range_summaries = []
    for values in ranges:
        med = statistics.median(values) if values else None
        range_summaries.append({
            "n": len(values), "median_m": round(med, 5) if med is not None else None,
            "mad_m": round(statistics.median(abs(v - med) for v in values), 5)
            if med is not None else None})
    if pair_distances:
        med = statistics.median(pair_distances)
        pair_summary = {"n": len(pair_distances), "median_m": round(med, 5),
                        "mad_m": round(statistics.median(abs(v - med) for v in pair_distances), 5),
                        "p95_m": round(float(np.percentile(pair_distances, 95)), 5),
                        "max_m": round(max(pair_distances), 5)}
    else:
        pair_summary = {"n": 0}
    return {"paired_frames": paired, "exact_two_frames": exact,
            "tp": tp, "fp": fp, "fn": fn,
            "ranges_by_target": range_summaries,
            "pair_separation_m": pair_summary}


def box_point(box):
    if box is None or box.get("distance_m") is None:
        return None
    fs = stereo.maps
    _, _, _, kl, r1, p1 = fs
    center = np.array([[[ (box["x1"] + box["x2"]) / 2,
                           (box["y1"] + box["y2"]) / 2 ]]], np.float32)
    uv = cv2.undistortPoints(center, kl, stereo.calibration["left_distortion"],
                             R=r1, P=p1)[0, 0]
    z = float(box["distance_m"])
    return np.array(((uv[0] - p1[0, 2]) * z / p1[0, 0],
                     (uv[1] - p1[1, 2]) * z / p1[1, 1], z))


out = {
    "method": ("Sweep gamma 0.55/0.60/0.65, luma trigger 15/20/25/30, and "
               "confidence floor 0.02/0.05. The existing adaptive rule raises "
               "the cutoff to 0.20 when at least three candidates score 0.05. "
               "Every changed detection set is rerun through active rectified "
               "SGBM and the 55% size/stereo gate; unchanged frames reuse baseline."),
    "settings": {"gammas": GAMMAS, "luma_gates": LUMA_GATES, "confidence_floors": FLOORS,
                 "adaptive_trigger": TRIGGER, "adaptive_cutoff": HIGH_CONF,
                 "stereo_baseline_m": 0.0635, "stereo_size_gate": 0.55},
    "limitations": ("Two stationary indoor videos with the same two adjacent FUEL. "
                    "Selection can characterize the dark opening but cannot show "
                    "field clutter, moving targets, or absolute range error."),
    "results": [],
}

try:
    for clip in BASE["results"]:
        cap = cv2.VideoCapture(str(ROOT / clip["video"]))
        baseline = []
        gamma_boxes = {gamma: {floor: [] for floor in FLOORS} for gamma in GAMMAS}
        gamma_latency_ms = {gamma: [] for gamma in GAMMAS}
        dark_frames = {}
        lumas = []
        processed = 0
        started = time.perf_counter()
        for index, detrow in enumerate(clip["per_frame"]):
            ok, frame = cap.read()
            if not ok:
                break
            left, right = frame[:, :1600], frame[:, 1600:]
            luma = float(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).mean())
            lumas.append(luma)
            raw = detrow["detections"]
            base_boxes = run_stereo(left, right, adapt(raw, BASE_CONF))
            baseline.append({"boxes": base_boxes, "pair_distance_m": None})
            if luma < OUTPUT_CAPABILITY_LUMA:
                dark_frames[index] = (left.copy(), right.copy())
                for gamma in GAMMAS:
                    lut = np.clip((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255,
                                  0, 255).astype(np.uint8)
                    candidates, latency = detector.infer(cv2.LUT(left, lut))
                    gamma_latency_ms[gamma].append(float(latency))
                    dets = [{"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                             "confidence": float(b.confidence), "class_id": b.class_id}
                            for b in candidates]
                    for floor in FLOORS:
                        selected = adapt(dets, floor)
                        gamma_boxes[gamma][floor].append((index, selected))
            processed += 1
        cap.release()

        # Baseline stereo geometry can be computed from already gated boxes.
        for row in baseline:
            hits, _ = score(row["boxes"], BASE_CONF)
            if all(hits):
                points = [box_point(hit) for hit in hits]
                if all(point is not None for point in points):
                    row["pair_distance_m"] = float(np.linalg.norm(points[0] - points[1]))

        policies = []
        baseline_summary = summarize(baseline, BASE_CONF)
        for gamma in GAMMAS:
            for floor in FLOORS:
                # Candidate results indexed by decoded frame, so each luma gate
                # selects a prefix of the corresponding gamma inference frames.
                candidate_by_frame = {}
                stereo_ms = []
                for index, selected in gamma_boxes[gamma][floor]:
                    left, right = dark_frames[index]
                    tic = time.perf_counter()
                    processed_boxes = run_stereo(left, right, selected)
                    stereo_ms.append((time.perf_counter() - tic) * 1000)
                    candidate_by_frame[index] = processed_boxes
                for gate in LUMA_GATES:
                    frames = []
                    used = 0
                    for i, base_row in enumerate(baseline):
                        if lumas[i] < gate:
                            boxes = candidate_by_frame[i]
                            used += 1
                        else:
                            boxes = base_row["boxes"]
                        row = {"boxes": boxes, "pair_distance_m": None}
                        hits, _ = score(boxes, min(floor, BASE_CONF))
                        if all(hits):
                            points = [box_point(hit) for hit in hits]
                            if all(point is not None for point in points):
                                row["pair_distance_m"] = float(np.linalg.norm(points[0] - points[1]))
                        frames.append(row)
                    metrics = summarize(frames, min(floor, BASE_CONF))
                    policies.append({"gamma": gamma, "luma_gate": gate,
                                     "confidence_floor": floor,
                                     "gamma_frames": used,
                                     "metrics": metrics,
                                     "median_gamma_inference_ms": round(statistics.median(
                                         gamma_latency_ms[gamma]), 2)
                                         if gamma_latency_ms[gamma] else None,
                                     "median_changed_frame_stereo_ms": round(statistics.median(stereo_ms), 2)
                                         if stereo_ms else None})
        output_clip = {
            "scene": clip["scene"], "video": clip["video"],
            "frames": processed, "elapsed_s": round(time.perf_counter() - started, 2),
            "luma_percentiles": [round(float(np.percentile(lumas, q)), 2) for q in (5, 50, 95)],
            "baseline": baseline_summary, "policies": policies,
        }
        out["results"].append(output_clip)
        print(clip["scene"], "baseline", baseline_summary)
        for item in sorted(policies, key=lambda x: (-x["metrics"]["exact_two_frames"],
                                                     -x["metrics"]["tp"], x["gamma_frames"]))[:8]:
            print(item["gamma"], item["luma_gate"], item["confidence_floor"],
                  item["gamma_frames"], item["metrics"]["exact_two_frames"],
                  item["metrics"]["tp"], item["metrics"]["fp"],
                  item["metrics"]["ranges_by_target"])
finally:
    detector.close()

path = ROOT / "media/frc-fuel-dark-policy-sweep.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print("saved", path)
