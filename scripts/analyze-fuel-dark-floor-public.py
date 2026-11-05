#!/usr/bin/env python3
"""Check dark-frame confidence-floor behavior on labeled Project516 stills."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "media/frc-fuel-project516-box-geometry-evaluation.json").read_text())
BASE = 0.05
LOW = 0.02
HIGH = 0.20
TRIGGER = 3
MAX_ASPECT = 2.0
IOU_THRESHOLD = 0.5


def iou(a, b):
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"])
    area_b = max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def match(predictions, labels):
    pairs = sorted(((iou(p, g), pi, gi)
                    for pi, p in enumerate(predictions)
                    for gi, g in enumerate(labels)), reverse=True)
    used_p, used_g = set(), set()
    for overlap, pi, gi in pairs:
        if overlap < IOU_THRESHOLD:
            break
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    return len(used_p), len(predictions) - len(used_p), len(labels) - len(used_g)


def configured(predictions):
    result = []
    for box in predictions:
        width = max(box["x2"] - box["x1"], 1e-9)
        height = max(box["y2"] - box["y1"], 1e-9)
        aspect = max(width / height, height / width)
        if aspect <= MAX_ASPECT:
            result.append(box)
    return result


POLICIES = {
    "fixed_0.05": lambda boxes: [b for b in boxes if b["confidence"] >= BASE],
    "adaptive_0.05": lambda boxes: [b for b in boxes if b["confidence"] >=
                                    (HIGH if sum(x["confidence"] >= BASE for x in boxes) >= TRIGGER else BASE)],
    "fixed_0.02": lambda boxes: [b for b in boxes if b["confidence"] >= LOW],
    "adaptive_floor_0.02": lambda boxes: [b for b in boxes if b["confidence"] >=
                                          (HIGH if sum(x["confidence"] >= BASE for x in boxes) >= TRIGGER else LOW)],
}

records = {name: [] for name in POLICIES}
for image in DATA["per_image_predictions"]:
    boxes = configured(image["predictions"])
    labels = image["labels"]
    for name, policy in POLICIES.items():
        predictions = policy(boxes)
        tp, fp, fn = match(predictions, labels)
        records[name].append({"image_id": image["image_id"],
                              "ground_truth_count": len(labels),
                              "predicted_count": len(predictions),
                              "tp": tp, "fp": fp, "fn": fn,
                              "absolute_count_error": abs(len(predictions) - len(labels))})

summaries = {}
for name, rows in records.items():
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    exact = sum(r["absolute_count_error"] == 0 for r in rows)
    summaries[name] = {
        "images": len(rows), "exact_count_images": exact,
        "count_mae": round(statistics.mean(r["absolute_count_error"] for r in rows), 4),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(tp / (tp + fp), 5) if tp + fp else None,
        "recall": round(tp / (tp + fn), 5) if tp + fn else None,
    }

output = {
    "method": ("Rescore cached FRC9550 candidate detections on the 53 labeled "
               "Project516 images at confidence floor 0.02. Compare fixed and "
               "adaptive thresholds, with and without a low floor; the adaptive "
               "rule counts candidates at 0.05 and uses cutoff 0.20 at 3+ candidates."),
    "settings": {"base_confidence": BASE, "low_floor": LOW, "high_cutoff": HIGH,
                 "trigger": TRIGGER, "max_box_aspect_ratio": MAX_ASPECT,
                 "match_iou": IOU_THRESHOLD,
                 "candidate_nms_iou": DATA["nms_iou"]},
    "limitations": ("These are monocular still images with no official train/test "
                    "membership and no stereo range labels. Cached candidates use "
                    "NMS 0.10, not the current video profile's 0.15. This global "
                    "rescore does not apply a luma trigger, gamma, or stereo size "
                    "gate; it is a clutter-risk check for the 0.02 floor only."),
    "summaries": summaries,
    "per_policy_per_image": records,
}
path = ROOT / "media/frc-fuel-dark-floor-public-evaluation.json"
path.write_text(json.dumps(output, indent=2) + "\n")
for name, summary in summaries.items():
    print(name, summary)
print("saved", path)
