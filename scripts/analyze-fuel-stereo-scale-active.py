#!/usr/bin/env python3
"""Compare vertical-ROI stereo scales against box-size proxy and pair stability."""
import json, math, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETECTIONS = json.loads((ROOT / "media/frc-fuel-dense-count-nms015.json").read_text())
SCALES = (0.50, 0.33, 0.25)
FOCAL_RAW_PX = 1046.16
FUEL_DIAMETER_M = 0.15
TARGETS = ((808, 589), (902, 591))


def matched(detections):
    out = [None, None]
    for box in sorted((b for b in detections if b["confidence"] >= .05),
                      key=lambda b: b["confidence"], reverse=True):
        center = ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)
        distances = [math.dist(center, target) for target in TARGETS]
        index = min(range(2), key=distances.__getitem__)
        if distances[index] <= 60 and out[index] is None:
            out[index] = box
    return out


def stats(values):
    if not values:
        return None
    center = statistics.median(values)
    return {"n": len(values), "median": round(center, 5),
            "mad": round(statistics.median(abs(x-center) for x in values), 5),
            "p05": round(float(sorted(values)[int(.05*(len(values)-1))]), 5),
            "p95": round(float(sorted(values)[int(.95*(len(values)-1))]), 5)}


out = {
    "method": "For every matched target in each saved clip, compare stereo scales .50/.33/.25 with a FUEL nominal-diameter range proxy from unrectified YOLO box width/height. Also summarize target ranges and 3D pair spacing. All matcher runs use SGBM 3WAY b5/u8, 128 disparities, and a 24-pixel vertical ROI pad.",
    "limitations": "The size proxy assumes tight boxes around an undistorted 15 cm sphere and is not measured range truth. This comparison can expose consistency shifts but cannot identify absolute accuracy.",
    "scales": [], "results": []
}
for scale in SCALES:
    suffix = {0.50: "0p50", 0.33: "0p33", 0.25: "0p25"}[scale]
    file = ROOT / f"media/frc-fuel-stereo-scale-active-{suffix}.json"
    data = json.loads(file.read_text())
    out["scales"].append({"scale": scale, "artifact": str(file.relative_to(ROOT)),
                            "matcher_ms_median": [r["stereo_compute_ms_median"] for r in data["results"]],
                            "matcher_ms_p95": [r["stereo_compute_ms_p95"] for r in data["results"]]})
    for clip_index, (clip, result) in enumerate(zip(DETECTIONS["results"], data["results"])):
        by_target = [[], []]
        abs_delta = [[], []]
        signed_delta = [[], []]
        pair_error = []
        paired = 0
        for index, row in enumerate(clip["per_frame"]):
            boxes = matched(row["detections"])
            ranges = result["frames"][index]["ranges_m"]
            for target, (box, z) in enumerate(zip(boxes, ranges)):
                if box is None or z is None:
                    continue
                width = max(1.0, box["x2"] - box["x1"])
                height = max(1.0, box["y2"] - box["y1"])
                size_range = FOCAL_RAW_PX * FUEL_DIAMETER_M / math.sqrt(width * height)
                by_target[target].append(z)
                signed_delta[target].append(z-size_range)
                abs_delta[target].append(abs(z-size_range))
            if all(boxes) and all(z is not None for z in ranges):
                pair_error.append(result["frames"][index]["pair_spacing_m"])
                paired += 1
        out["results"].append({"scale": scale, "scene": clip["scene"],
            "valid_pairs": paired, "range_m": [stats(x) for x in by_target],
            "signed_stereo_minus_size_proxy_m": [stats(x) for x in signed_delta],
            "absolute_stereo_to_size_proxy_delta_m": [stats(x) for x in abs_delta],
            "pair_spacing_m": stats(pair_error)})

path = ROOT / "media/frc-fuel-stereo-scale-active-analysis.json"
path.write_text(json.dumps(out, indent=2) + "\n")
for row in out["results"]:
    print(row["scale"], row["scene"], "pairs", row["valid_pairs"],
          "ranges", [x["median"] if x else None for x in row["range_m"]],
          "size abs delta", [x["median"] if x else None for x in row["absolute_stereo_to_size_proxy_delta_m"]],
          "spacing", row["pair_spacing_m"])
print("saved", path)
