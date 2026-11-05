#!/usr/bin/env python3
"""Score 3D cluster counts for the best dark-frame detector policy."""
import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "media/frc-fuel-gamma-adaptive-confidence.json").read_text())
CAL = ROOT / "media/calibration/stereo_calibration_live_order_candidate.yml"
TARGETS = ((808.0, 589.0), (902.0, 591.0))
RADII = (0.15, 0.22, 0.25, 0.26)
fs = cv2.FileStorage(str(CAL), cv2.FILE_STORAGE_READ)
kl = fs.getNode("left_camera_matrix").mat()
dl = fs.getNode("left_distortion").mat()
kr = fs.getNode("right_camera_matrix").mat()
dr = fs.getNode("right_distortion").mat()
rotation = fs.getNode("rotation").mat()
translation = fs.getNode("translation").mat()
fs.release()
r1, _, p1, _, *_ = cv2.stereoRectify(
    kl, dl, kr, dr, (1600, 1200), rotation, translation,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)


def object_points(boxes, min_conf):
    matched = [None, None]
    extras = 0
    for box in sorted((b for b in boxes if b["confidence"] >= min_conf),
                      key=lambda b: -b["confidence"]):
        center = ((box["x1"] + box["x2"]) / 2,
                  (box["y1"] + box["y2"]) / 2)
        error, index = min((math.dist(center, target), i)
                           for i, target in enumerate(TARGETS))
        if error <= 60 and matched[index] is None:
            matched[index] = box
        else:
            extras += 1
    points = []
    for box in matched:
        if box is None or box.get("distance_m") is None:
            points.append(None)
            continue
        cx = (box["x1"] + box["x2"]) / 2
        cy = (box["y1"] + box["y2"]) / 2
        uv = cv2.undistortPoints(np.array([[[cx, cy]]], np.float32), kl, dl,
                                 R=r1, P=p1)[0, 0]
        z = float(box["distance_m"])
        points.append(np.array(((uv[0] - p1[0, 2]) * z / p1[0, 0],
                                (uv[1] - p1[1, 2]) * z / p1[1, 1], z)))
    return points, extras


def components(points, radius):
    valid = [point for point in points if point is not None]
    parent = list(range(len(valid)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            if np.linalg.norm(valid[i] - valid[j]) <= radius:
                parent[find(j)] = find(i)
    return len({find(i) for i in range(len(valid))})


output = {
    "method": ("Use saved post-stereo-size-gate detections from raw adaptive "
               "confidence and gamma 0.60 plus a 0.02 dim-frame confidence floor. "
               "Unproject centers with the active rectified calibration and "
               "63.5 mm runtime baseline; count 3D connected components at several "
               "cluster radii."),
    "limitations": ("Only two stationary neighboring FUEL balls are present. "
                    "This measures cluster repeatability for this pair, not false "
                    "merges between unrelated clusters or absolute metric accuracy."),
    "cluster_radii_m": list(RADII),
    "results": [],
}

for clip in DATA["results"]:
    summary = {"baseline": {r: [] for r in RADII},
               "gamma_low_floor": {r: [] for r in RADII}}
    pair_distances = {key: [] for key in summary}
    rows = {key: 0 for key in summary}
    paired = {key: 0 for key in summary}
    extras = {key: 0 for key in summary}
    for row in clip["per_frame"]:
        for key, source_key, minimum in (("baseline", "adaptive_only", 0.05),
                                         ("gamma_low_floor", "gamma_low_floor", 0.02)):
            points, fp = object_points(row[source_key], minimum)
            extras[key] += fp
            rows[key] += 1
            if all(point is not None for point in points):
                paired[key] += 1
                pair_distances[key].append(float(np.linalg.norm(points[0] - points[1])))
            for radius in RADII:
                summary[key][radius].append(components(points, radius))
    configs = {}
    for key in summary:
        configs[key] = {
            "paired_valid_frames": paired[key], "off_target_detections": extras[key],
            "pair_distance_m": {
                "n": len(pair_distances[key]),
                "median": round(statistics.median(pair_distances[key]), 5),
                "mad": round(statistics.median(abs(x - statistics.median(pair_distances[key]))
                                               for x in pair_distances[key]), 5),
                "p95": round(float(np.percentile(pair_distances[key], 95)), 5),
                "max": round(max(pair_distances[key]), 5),
            } if pair_distances[key] else {"n": 0},
            "cluster_count_frames_by_radius": {
                str(radius): {str(count): values.count(count) for count in sorted(set(values))}
                for radius, values in summary[key].items()},
        }
    result = {"scene": clip["scene"], "frames": rows["baseline"], "configs": configs}
    output["results"].append(result)
    print(clip["scene"], json.dumps(configs))

path = ROOT / "media/frc-fuel-dark-floor-cluster-evaluation.json"
path.write_text(json.dumps(output, indent=2) + "\n")
print("saved", path)
