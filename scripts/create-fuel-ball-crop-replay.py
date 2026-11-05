#!/usr/bin/env python3
"""Create a 60 Hz left-camera crop around a detected FUEL ball pair."""
import argparse
import json
import statistics
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "media/frc-2026-fuel-camera-settings-fixed-60fps.mp4"
DEFAULT_EVALUATION = ROOT / "media/frc-fuel-dense-count-evaluation.json"
DEFAULT_OUTPUT = ROOT / "media/frc-fuel-ball-crop-60fps.mp4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scene", default="corrected-settings")
    parser.add_argument("--crop-size", type=int, default=256)
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation.read_text())
    scene = next(item for item in evaluation["results"]
                 if item["scene"] == args.scene)
    # Select the consistently detected left-hand ball, then center the crop
    # at its median detector center to avoid adding detector jitter to playback.
    leftmost = [min(frame["detections"], key=lambda d: (d["x1"] + d["x2"]) / 2)
                for frame in scene["per_frame"] if frame["detections"]]
    if not leftmost:
        raise RuntimeError(f"No detections found for scene {args.scene}")
    center_x = statistics.median((d["x1"] + d["x2"]) / 2 for d in leftmost)
    center_y = statistics.median((d["y1"] + d["y2"]) / 2 for d in leftmost)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if width < 2 * args.crop_size:
        raise RuntimeError("Source is not a side-by-side stereo frame")
    half = args.crop_size // 2
    x0 = max(0, min(width // 2 - args.crop_size,
                    round(center_x) - half))
    y0 = max(0, min(height - args.crop_size, round(center_y) - half))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (args.crop_size, args.crop_size))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {args.output}")

    frame_count = 0
    while True:
        ok, stereo = capture.read()
        if not ok:
            break
        # The source is 3200x1200 side-by-side; detections are in left-view
        # coordinates (1600x1200), so crop only that calibrated camera view.
        writer.write(stereo[y0:y0 + args.crop_size,
                            x0:x0 + args.crop_size])
        frame_count += 1
    capture.release()
    writer.release()

    def display_path(path):
        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError:
            return str(path)

    metadata = {
        "source_video": display_path(args.video),
        "detector_evaluation": display_path(args.evaluation),
        "scene": args.scene,
        "crop_video": display_path(args.output),
        "source_frame_size": [width, height],
        "source_fps": fps,
        "crop_frame_size": [args.crop_size, args.crop_size],
        "crop_origin_in_left_camera": [x0, y0],
        "selected_ball_center_px": [center_x, center_y],
        "selected_detection_frames": len(leftmost),
        "frames_written": frame_count,
        "duration_s": frame_count / fps if fps else None,
        "note": "Detector-centered fixed crop from the left view. The nearby second ball remains visible; source playback remains at its recorded frame rate.",
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
