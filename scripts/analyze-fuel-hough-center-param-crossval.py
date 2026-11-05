#!/usr/bin/env python3
"""Leave-one-recording-out check for Hough center param2 selection."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "media/frc-fuel-hough-center-param-sweep.json").read_text())


def score(config):
    # Balance typical and tail localization; lower is better.
    center = config["refined_center_error_px"]
    return center["median"] + center["p95"]


results = []
for train_i, test_i in ((0, 1), (1, 0)):
    train, test = DATA["results"][train_i], DATA["results"][test_i]
    selected = min(train["configs"], key=score)
    heldout = next(c for c in test["configs"] if c["param2"] == selected["param2"])
    box = heldout["box_center_error_px"]
    refined = heldout["refined_center_error_px"]
    results.append({
        "train_scene": train["scene"], "test_scene": test["scene"],
        "selection_metric": "median_center_error_px + p95_center_error_px",
        "selected_param2": selected["param2"],
        "training_score_px": round(score(selected), 5),
        "heldout_box_center": {"median_px": box["median"], "p95_px": box["p95"]},
        "heldout_hough_center": {"median_px": refined["median"], "p95_px": refined["p95"]},
        "heldout_score_gain_px": round(box["median"] + box["p95"] - score(heldout), 5),
        "heldout_hough_valid_centers": heldout["valid_hough_centers"],
        "heldout_fallback_centers": heldout["fallback_centers"],
    })

out = {
    "method": "Select param2 using median+p95 pixel center error on one saved clip and score that fixed setting on the other.",
    "limitations": "Only two recordings of the same stationary two-ball arrangement, fixed manual image centers, and no motion or physical occlusion. The split checks camera-setting transfer, not independent-scene generalization.",
    "results": results,
}
path = ROOT / "media/frc-fuel-hough-center-param-crossval.json"
path.write_text(json.dumps(out, indent=2) + "\n")
for row in results:
    print(json.dumps(row))
print("saved", path)
