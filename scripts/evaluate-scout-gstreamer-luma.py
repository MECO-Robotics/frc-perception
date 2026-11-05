#!/usr/bin/env python3
"""Compare Scout's GStreamer left-view luma with OpenCV on saved stereo MP4s."""
import json, os, statistics, time
from pathlib import Path

import cv2
import gi
import numpy as np
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = (("earlier", "media/frc-2026-live-two-fuel-60fps-cfr.mp4"),
          ("corrected-settings", "media/frc-2026-fuel-camera-settings-fixed-60fps.mp4"))
Gst.init(None)


def summary(values):
    return {"n": len(values), "min": round(min(values), 3),
            "p05": round(float(np.percentile(values, 5)), 3),
            "median": round(statistics.median(values), 3),
            "p95": round(float(np.percentile(values, 95)), 3),
            "max": round(max(values), 3)} if values else None


out = {
    "method": "Decode each saved stereo MP4 through the same Jetson path as Scout (nvv4l2decoder, nvvidconv to BGRx at 3200x1200, crop left 1600-pixel eye, videoconvert to BGR). Compare per-frame grayscale means with OpenCV VideoCapture on the same encoded frame index.",
    "limitations": "This measures pixel-conversion differences and whether an offline luma cutoff transfers to the runtime decoder. It does not evaluate detector recall or range accuracy.",
    "thresholds": [20, 30, 40, 50, 60, 70], "results": []
}

for scene, relative in VIDEOS:
    video = ROOT / relative
    opencv = cv2.VideoCapture(str(video))
    pipeline_description = (
        f'filesrc location="{video}" ! qtdemux name=demux '
        'demux.video_0 ! queue ! h264parse ! nvv4l2decoder ! nvvidconv ! '
        'video/x-raw,format=BGRx,width=3200,height=1200 ! '
        'videocrop right=1600 ! videoconvert ! video/x-raw,format=BGR ! '
        'appsink name=sink sync=false max-buffers=8 drop=false enable-last-sample=false'
    )
    pipeline = Gst.parse_launch(pipeline_description)
    sink = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)
    rows, started = [], time.perf_counter()
    bus = pipeline.get_bus()
    while True:
        sample = sink.emit("try-pull-sample", 2 * Gst.SECOND)
        if sample is None:
            message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if message and message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                raise RuntimeError(f"GStreamer decode failed for {scene}: {err}; {debug}")
            break
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        info = GstVideo.VideoInfo()
        if not info.from_caps(caps):
            raise RuntimeError(f"Unable to read video caps: {caps.to_string()}")
        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("Unable to map GStreamer BGR sample")
        try:
            width, height = info.width, info.height
            stride = info.stride[0]
            bgr = np.ndarray((height, width, 3), dtype=np.uint8,
                             buffer=mapped.data, strides=(stride, 3, 1))
            luma_gst = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())
            valid, cv_frame = opencv.read()
            luma_cv = float(cv2.cvtColor(cv_frame[:, :1600], cv2.COLOR_BGR2GRAY).mean()) if valid else None
            rows.append({"frame": len(rows), "luma_gstreamer": luma_gst, "luma_opencv": luma_cv,
                         "difference": luma_gst-luma_cv if luma_cv is not None else None})
        finally:
            buffer.unmap(mapped)
    pipeline.set_state(Gst.State.NULL)
    opencv.release()
    gst_values = [r["luma_gstreamer"] for r in rows]
    cv_values = [r["luma_opencv"] for r in rows if r["luma_opencv"] is not None]
    differences = [r["difference"] for r in rows if r["difference"] is not None]
    result = {"scene": scene, "video": relative, "frames": len(rows),
              "elapsed_s": round(time.perf_counter()-started, 2),
              "gstreamer_luma": summary(gst_values), "opencv_luma": summary(cv_values),
              "gstreamer_minus_opencv": summary(differences),
              "frames_below_threshold": {str(t): sum(x < t for x in gst_values) for t in out["thresholds"]},
              "first_40_frames": rows[:40]}
    out["results"].append(result)
    print(scene, json.dumps({k:v for k,v in result.items() if k != "first_40_frames"}))

path = ROOT / "media/frc-scout-gstreamer-luma-comparison.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print("saved", path)
