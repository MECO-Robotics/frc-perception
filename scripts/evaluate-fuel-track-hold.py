#!/usr/bin/env python3
"""Evaluate causal short track holds on unique live replay outputs."""
import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "media/frc-fuel-api-unique-block3-position.json"
source = json.loads(SOURCE_PATH.read_text())
stats = source["stats_before"]
width, height = map(int, stats["inference_resolution"].split("x"))
fx, fy = float(stats["stereo_focal_px"]), float(stats["stereo_focal_px"])
cx, cy = width/2, height/2
baseline = float(stats["stereo_baseline_m"])
scale = float(stats.get("stereo_disparity_scale", 1.0))
rows = source["unique_frames"]


def median(values):
    return float(statistics.median(values)) if values else None


def summarize(values):
    if not values:
        return None
    center = median(values)
    return {"n": len(values), "median": round(center, 6),
            "p95": round(float(np.percentile(values, 95)), 6),
            "mad": round(median([abs(v-center) for v in values]), 6)}


def decode(row):
    if row["count"] != 2 or len(row.get("centers_normalized", [])) != 2:
        return None
    targets = []
    for uv, disparity in zip(row["centers_normalized"], row["disparities_px"]):
        if disparity is None or disparity <= 0:
            return None
        u, v = uv[0]*width, uv[1]*height
        z = baseline*fx/(float(disparity)*scale)
        targets.append((u, v, z, np.asarray(((u-cx)*z/fx, (v-cy)*z/fy, z))))
    # The balls remain side by side in this recording; stable left/right ordering
    # gives a simple association for this fixed-scene persistence experiment.
    return sorted(targets, key=lambda item: item[0])


decoded = [decode(row) for row in rows]
time_s = [float(row["rtp_timestamp"])/90000.0 for row in rows]
cadence_ms = [1000*(b-a) for a,b in zip(time_s,time_s[1:])]


def evaluate(max_gap=None, max_age_ms=None):
    last = None
    last_seen_index = None
    emitted = []
    stale_ages_ms = []
    for index, value in enumerate(decoded):
        if value is not None:
            last, last_seen_index = value, index
            emitted.append((value, False, 0.0))
        elif (last is not None and
              ((max_gap is not None and index-last_seen_index <= max_gap) or
               (max_age_ms is not None and 1000*(time_s[index]-time_s[last_seen_index]) <= max_age_ms))):
            age = 1000*(time_s[index]-time_s[last_seen_index])
            emitted.append((last, True, age))
            stale_ages_ms.append(age)
        else:
            emitted.append((None, False, None))
    exact2 = sum(value is not None for value,_,_ in emitted)
    hold_rows = [(i,v,age) for i,(v,stale,age) in enumerate(emitted) if stale]
    reacquisition = []
    for i, (held, stale, age) in enumerate(emitted):
        if not stale:
            continue
        next_fresh = next((j for j in range(i+1, len(emitted))
                           if emitted[j][0] is not None and not emitted[j][1]), None)
        if next_fresh is not None:
            fresh = emitted[next_fresh][0]
            reacquisition.extend(float(np.linalg.norm(held[t][3]-fresh[t][3])) for t in range(2))
    target_xyz = [[], []]
    for values, _, _ in emitted:
        if values is not None:
            for target in range(2):
                target_xyz[target].append(values[target][3])
    spreads = []
    for series in target_xyz:
        a=np.asarray(series)
        med=np.median(a,axis=0)
        spreads.append({axis: {"mad_m": round(float(np.median(abs(a[:,k]-med[k]))),6),
                               "p95_abs_deviation_m": round(float(np.percentile(abs(a[:,k]-med[k]),95)),6)}
                        for k,axis in enumerate(("x","y","z"))})
    return {"max_missed_updates": max_gap,
            "max_position_age_ms": max_age_ms,
            "two_target_outputs": exact2,
            "filled_zero_box_updates": len(hold_rows),
            "remaining_zero_box_updates": len(rows)-exact2,
            "stale_position_age_ms": summarize(stale_ages_ms),
            "held_to_next_fresh_position_delta_m": summarize(reacquisition),
            "output_position_spread": spreads}


out = {
    "method": "Use one sample per unique RTP timestamp from the live block-3 replay. If current output lacks exactly two valid 3D targets, compare holds limited by 0/1/2/3 missing updates and independent wall-clock age limits of 100/150/250/350/450 ms. Association is left/right by image x for this fixed side-by-side scene. Never refresh a held timestamp from a held output.",
    "input": str(SOURCE_PATH.relative_to(ROOT)),
    "unique_outputs": len(rows),
    "fresh_two_target_outputs": sum(v is not None for v in decoded),
    "zero_box_indices": [i for i,row in enumerate(rows) if row["count"] == 0],
    "inference_cadence_ms": summarize(cadence_ms),
    "hold_results": [evaluate(max_gap=n) for n in (0,1,2,3)] +
                    [evaluate(max_age_ms=n) for n in (100,150,250,350,450)],
    "limitations": "The source is one stationary two-ball replay; persistence can make cluster count steadier here but cannot distinguish a detector miss from a real target departure. Position repeatability on held outputs is artificially reduced, and moving-target lag/false persistence are not measured. No camera was opened for this analysis."
}
path = ROOT / "media/frc-fuel-track-hold-evaluation.json"
path.write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out, indent=2))
print("saved", path)
