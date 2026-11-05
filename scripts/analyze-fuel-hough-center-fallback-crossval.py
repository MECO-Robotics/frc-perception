#!/usr/bin/env python3
"""Leave-one-clip-out check of exploratory Hough-center fallback gates."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "media/frc-fuel-hough-center-fallback-sweep.json").read_text())


def score(row):
    # Equal weight on typical and tail localization error; lower is better.
    return row["median_px"] + row["p90_px"]


def describe_gate(gate):
    return {k: gate[k] for k in ("type", "max_spread_px", "band", "radius_band") if k in gate}


results = []
for train_i, test_i in ((0, 1), (1, 0)):
    train, test = DATA["results"][train_i], DATA["results"][test_i]
    selected = min(train["gates"], key=score)
    heldout = next(g for g in test["gates"] if describe_gate(g) == describe_gate(selected))
    train_box = train["baseline"]["box_only"]
    test_box = test["baseline"]["box_only"]
    results.append({
        "train_scene": train["scene"], "test_scene": test["scene"],
        "selection_metric": "median_center_error_px + p90_center_error_px",
        "selected_gate": describe_gate(selected),
        "train_gate": {"median_px": selected["median_px"], "p90_px": selected["p90_px"],
                       "hough_selected_fraction": selected["hough_selected_fraction"]},
        "train_box_center": {"median_px": train_box["median_px"], "p90_px": train_box["p90_px"]},
        "heldout_gate": {"median_px": heldout["median_px"], "p90_px": heldout["p90_px"],
                         "hough_selected_fraction": heldout["hough_selected_fraction"]},
        "heldout_box_center": {"median_px": test_box["median_px"], "p90_px": test_box["p90_px"]},
        "heldout_composite_score_gain_px": round(score(test_box) - score(heldout), 3),
        "heldout_by_mask": heldout["by_mask"],
    })
out = {
    "method": "For each direction, select the gate with lowest median+p90 center error on one clip, then evaluate that exact gate on the other clip. This tests transfer across the two camera-settings recordings but retains synthetic masks and fixed targets.",
    "limitations": "Only two stationary recordings, synthetic masks, and manually fixed centers. No motion, physical occlusion, or field cluster diversity. Selection thresholds come from the previous exploratory sweep.",
    "results": results,
}
path = ROOT / "media/frc-fuel-hough-center-fallback-crossval.json"
path.write_text(json.dumps(out, indent=2) + "\n")
for result in results:
    print(result["train_scene"], "->", result["test_scene"], result["selected_gate"],
          "heldout gain", result["heldout_composite_score_gain_px"],
          "gate", result["heldout_gate"], "box", result["heldout_box_center"])
print("saved", path)
