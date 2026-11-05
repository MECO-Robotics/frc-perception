#!/usr/bin/env python3
"""Compare causal EMA and rolling-median stereo range smoothing at replay cadence."""
import argparse
import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COUNTS = json.loads((ROOT / "media/frc-fuel-dense-count-evaluation.json").read_text())
STEREO = json.loads((ROOT / "media/frc-fuel-dense-stereo-evaluation.json").read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stride", type=int, default=7, help="decoded frames between simulated inference outputs")
args = parser.parse_args()
STRIDE, FPS = args.stride, 59.900166
PHASES = range(STRIDE)
TARGETS = ((808, 589), (902, 591))
METHODS = {"raw": None, "median3": ("median", 3), "median5": ("median", 5),
           "ema025": ("ema", .25), "ema050": ("ema", .50), "ema075": ("ema", .75)}
fs = cv2.FileStorage(str(ROOT / "media/calibration/stereo_calibration_live_order_candidate.yml"), cv2.FILE_STORAGE_READ)
kl, dl = fs.getNode("left_camera_matrix").mat(), fs.getNode("left_distortion").mat()
kr, dr = fs.getNode("right_camera_matrix").mat(), fs.getNode("right_distortion").mat()
rotation, translation = fs.getNode("rotation").mat(), fs.getNode("translation").mat()
fs.release()
r1, r2, p1, p2, *_ = cv2.stereoRectify(kl, dl, kr, dr, (1600, 1200), rotation, translation,
                                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)


def matched(dets):
    out = [None, None]
    for b in sorted((b for b in dets if b["confidence"] >= .10), key=lambda b: -b["confidence"]):
        x, y = (b["x1"]+b["x2"])/2, (b["y1"]+b["y2"])/2
        ds = [math.hypot(x-a, y-c) for a, c in TARGETS]
        i = int(np.argmin(ds))
        if ds[i] <= 60 and out[i] is None:
            out[i] = b
    return out


def smooth(events, target, method):
    config = METHODS[method]
    result = []
    history = []
    ema = None
    for event in events:
        value = event["ranges"][target]
        if config is None:
            result.append(value)
            continue
        if value is None:
            result.append(None)
            continue
        if config[0] == "median":
            history.append(value)
            result.append(float(statistics.median(history[-config[1]:])))
        else:
            alpha = config[1]
            ema = value if ema is None else alpha*value + (1-alpha)*ema
            result.append(ema)
    return result


def dispersion(values, reference):
    vals = [x for x in values if x is not None]
    if not vals:
        return {"n": 0}
    center = statistics.median(reference)
    absdev = [abs(x-center) for x in vals]
    med = statistics.median(vals)
    return {"n": len(vals), "median_m": round(med, 5),
            "mad_m": round(statistics.median(abs(x-med) for x in vals), 5),
            "p95_abs_dev_from_raw_median_m": round(float(np.percentile(absdev, 95)), 5)}


def xyz(box, z):
    center = np.array([[[float(box["x1"]+box["x2"])/2,
                         float(box["y1"]+box["y2"])/2]]], np.float32)
    q = cv2.undistortPoints(center, kl, dl, R=r1, P=p1)[0, 0]
    return ((q[0]-p1[0,2])*z/p1[0,0], (q[1]-p1[1,2])*z/p1[1,1], z)


def summary(vals, raw_center=None):
    vals = [x for x in vals if x is not None]
    if not vals:
        return {"n": 0}
    m = statistics.median(vals)
    center = statistics.median(raw_center) if raw_center else m
    return {"n": len(vals), "median_m": round(m, 5),
            "mad_m": round(statistics.median(abs(x-m) for x in vals), 5),
            "p95_abs_dev_m": round(float(np.percentile([abs(x-center) for x in vals],95)),5)}


out = {
    "method": "Subsample dense per-frame detector and stereo results at stride 7 across all phases (about 8.6 Hz). Compare raw ranges, causal rolling medians over the last 3/5 outputs, and causal EMA alpha=.25/.5/.75. Missing current ranges remain missing; no hold-last output is counted.",
    "limitations": "Two stationary targets, fixed camera, no moving-object response or absolute range truth. This measures jitter and valid-sample retention only; smoothing response time is represented by the sampling window/EMA alpha and needs motion validation.",
    "stride_frames": STRIDE,
    "approx_output_hz": FPS/STRIDE,
    "phase_results": [],
}

for count_clip, stereo_clip in zip(COUNTS["results"], STEREO["results"]):
    frame_rows = count_clip["per_frame"]
    depth_rows = stereo_clip["frames"]
    phases = []
    for phase in PHASES:
        events = []
        for index in range(phase, len(frame_rows), STRIDE):
            boxes = matched(frame_rows[index]["detections"])
            ranges = depth_rows[index]["ranges_m"]
            events.append({"frame": index, "boxes": boxes, "ranges": ranges})
        method_rows = []
        raw_series = [smooth(events, t, "raw") for t in range(2)]
        raw_centers = [[x for x in series if x is not None] for series in raw_series]
        for method in METHODS:
            series = [smooth(events, t, method) for t in range(2)]
            targets = [summary(series[t], raw_centers[t]) for t in range(2)]
            spacings = []
            for i, event in enumerate(events):
                z0, z1 = series[0][i], series[1][i]
                if z0 is None or z1 is None or not all(event["boxes"]):
                    continue
                spacings.append(math.dist(xyz(event["boxes"][0], z0), xyz(event["boxes"][1], z1)))
            raw_spacing = []
            for i, event in enumerate(events):
                a, b = raw_series[0][i], raw_series[1][i]
                if a is not None and b is not None and all(event["boxes"]):
                    raw_spacing.append(math.dist(xyz(event["boxes"][0], a), xyz(event["boxes"][1], b)))
            method_rows.append({"method": method, "targets": targets,
                                "pair_spacing_m": summary(spacings, raw_spacing)})
        phases.append({"phase": phase, "sampled_outputs": len(events), "methods": method_rows})
    aggregate = []
    for method in METHODS:
        rows = [next(x for x in p["methods"] if x["method"] == method) for p in phases]
        aggregate.append({"method": method,
                          "mean_target_mad_m": [round(statistics.mean(r["targets"][t]["mad_m"] for r in rows),5) for t in range(2)],
                          "mean_target_p95_abs_dev_m": [round(statistics.mean(r["targets"][t]["p95_abs_dev_m"] for r in rows),5) for t in range(2)],
                          "mean_pair_spacing_mad_m": round(statistics.mean(r["pair_spacing_m"]["mad_m"] for r in rows),5),
                          "mean_pair_spacing_p95_abs_dev_m": round(statistics.mean(r["pair_spacing_m"]["p95_abs_dev_m"] for r in rows),5),
                          "mean_valid_target_samples": [round(statistics.mean(r["targets"][t]["n"] for r in rows),2) for t in range(2)],
                          "mean_valid_pair_samples": round(statistics.mean(r["pair_spacing_m"]["n"] for r in rows),2)})
    result = {"scene": count_clip["scene"], "phase_average": aggregate, "phases": phases}
    out["phase_results"].append(result)
    print(count_clip["scene"], json.dumps(aggregate))

path = ROOT / f"media/frc-fuel-range-smoother-stride-{STRIDE}.json"
path.write_text(json.dumps(out, indent=2)+"\n")
print("saved", path)
