#!/usr/bin/env bash
set -euo pipefail

cd /home/xavier/scout-edge
exec 9>/tmp/scout-edge.lock
flock -n 9 || exit 0

export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0
export LD_LIBRARY_PATH=/usr/local/cuda/targets/aarch64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export GST_PLUGIN_PATH=/home/xavier/scout-edge/.gst/usr/lib/aarch64-linux-gnu/gstreamer-1.0
export SCOUT_DETECTOR=tensorrt
export SCOUT_ENGINE="${SCOUT_ENGINE:-models/rebuilt_fuel_yolo26n_1280x960_fp16.engine}"
export SCOUT_CLASS_NAMES="${SCOUT_CLASS_NAMES:-[\"ball\"]}"
export SCOUT_MAX_BOX_ASPECT_RATIO="${SCOUT_MAX_BOX_ASPECT_RATIO:-2.0}"
export SCOUT_CAMERA_FORMAT="${SCOUT_CAMERA_FORMAT:-MJPG}"
export SCOUT_CAMERA_WIDTH="${SCOUT_CAMERA_WIDTH:-3200}"
export SCOUT_CAMERA_HEIGHT="${SCOUT_CAMERA_HEIGHT:-1200}"
export SCOUT_CAMERA_FPS="${SCOUT_CAMERA_FPS:-60}"
export SCOUT_STEREO_ENABLED="${SCOUT_STEREO_ENABLED:-1}"
export SCOUT_STEREO_CALIBRATION="${SCOUT_STEREO_CALIBRATION:-media/calibration/stereo_calibration_live_order_candidate.yml}"
export SCOUT_STEREO_RECTIFY="${SCOUT_STEREO_RECTIFY:-1}"
export SCOUT_STEREO_DENSE="${SCOUT_STEREO_DENSE:-0}"
export SCOUT_STEREO_LOCAL_RECTIFY="${SCOUT_STEREO_LOCAL_RECTIFY:-1}"
export SCOUT_STEREO_BASELINE_M="${SCOUT_STEREO_BASELINE_M:-0.0635}"
export SCOUT_STEREO_DISPARITY_SCALE="${SCOUT_STEREO_DISPARITY_SCALE:-1.0}"
export SCOUT_STEREO_MATCH_SCALE="${SCOUT_STEREO_MATCH_SCALE:-0.25}"
export SCOUT_STEREO_NUM_DISPARITIES="${SCOUT_STEREO_NUM_DISPARITIES:-32}"
export SCOUT_STEREO_VERTICAL_ROI_PAD="${SCOUT_STEREO_VERTICAL_ROI_PAD:-24}"
export SCOUT_FUEL_DIAMETER_M="${SCOUT_FUEL_DIAMETER_M:-0.15}"
export SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR="${SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR:-0.55}"
export SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU="${SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU:-0.90}"
export SCOUT_TRACK_DIRECT_FLOW="${SCOUT_TRACK_DIRECT_FLOW:-1}"
export SCOUT_TRACK_FLOW_GRACE_MS="${SCOUT_TRACK_FLOW_GRACE_MS:-250}"
export SCOUT_STREAM_ENABLED="${SCOUT_STREAM_ENABLED:-0}"
export SCOUT_OUTPUT_WIDTH=640
export SCOUT_OUTPUT_HEIGHT=480
export SCOUT_INFERENCE_WIDTH="${SCOUT_INFERENCE_WIDTH:-1280}"
export SCOUT_INFERENCE_MAX_FPS="${SCOUT_INFERENCE_MAX_FPS:-5}"
export SCOUT_INFERENCE_ALWAYS="${SCOUT_INFERENCE_ALWAYS:-1}"
export SCOUT_FLOW_WIDTH="${SCOUT_FLOW_WIDTH:-640}"
export SCOUT_OPENCV_THREADS="${SCOUT_OPENCV_THREADS:-1}"
export SCOUT_PLAYBACK_RATE="${SCOUT_PLAYBACK_RATE:-1.0}"
export SCOUT_TARGET_FPS="${SCOUT_TARGET_FPS:-60}"
export SCOUT_CONFIDENCE="${SCOUT_CONFIDENCE:-0.05}"
export SCOUT_NMS_IOU="${SCOUT_NMS_IOU:-0.10}"
export SCOUT_ADAPTIVE_CONFIDENCE_TRIGGER="${SCOUT_ADAPTIVE_CONFIDENCE_TRIGGER:-3}"
export SCOUT_ADAPTIVE_CONFIDENCE_HIGH="${SCOUT_ADAPTIVE_CONFIDENCE_HIGH:-0.20}"
export SCOUT_MAX_CANDIDATES="${SCOUT_MAX_CANDIDATES:-0}"
export SCOUT_REUSE_PREPROCESS="${SCOUT_REUSE_PREPROCESS:-1}"
if command -v tailscale >/dev/null 2>&1; then
  export SCOUT_ICE_ADDRESS="${SCOUT_ICE_ADDRESS:-$(tailscale ip -4 2>/dev/null | head -n1)}"
fi

if [[ -n "${SCOUT_SOURCE:-}" ]]; then
  SOURCE="$SCOUT_SOURCE"
else
  SOURCE="media/frc-2026-fuel-camera-settings-fixed-60fps.mp4"
fi
CAMERA_DEVICE="$SOURCE"
if [[ "$SOURCE" =~ ^[0-9]+$ ]]; then
  CAMERA_DEVICE="/dev/video$SOURCE"
fi
if [[ "$SOURCE" == /dev/v4l/* ]]; then
  RESOLVED_CAMERA="$(readlink -f "$SOURCE" 2>/dev/null || true)"
  if [[ -n "$RESOLVED_CAMERA" ]]; then
    CAMERA_DEVICE="$RESOLVED_CAMERA"
    SOURCE="$RESOLVED_CAMERA"
  fi
fi
if [[ "$SOURCE" == /dev/video* || "$SOURCE" == /dev/v4l/* || "$SOURCE" =~ ^[0-9]+$ ]] && [[ ! -e "$CAMERA_DEVICE" ]]; then
  echo "Camera $CAMERA_DEVICE is unavailable; refusing to fall back to a different source" >&2
  exit 1
fi

SOURCE_2="${SCOUT_SOURCE_2:-}"
if [[ -n "$SOURCE_2" ]]; then
  CAMERA_DEVICE_2="$SOURCE_2"
  if [[ "$SOURCE_2" =~ ^[0-9]+$ ]]; then
    CAMERA_DEVICE_2="/dev/video$SOURCE_2"
  fi
  if [[ "$SOURCE_2" == /dev/v4l/* ]]; then
    RESOLVED_CAMERA_2="$(readlink -f "$SOURCE_2" 2>/dev/null || true)"
    if [[ -n "$RESOLVED_CAMERA_2" ]]; then
      CAMERA_DEVICE_2="$RESOLVED_CAMERA_2"
      SOURCE_2="$RESOLVED_CAMERA_2"
    fi
  fi
  if [[ "$SOURCE_2" == /dev/video* || "$SOURCE_2" == /dev/v4l/* || "$SOURCE_2" =~ ^[0-9]+$ ]] && [[ ! -e "$CAMERA_DEVICE_2" ]]; then
    echo "Camera $CAMERA_DEVICE_2 is unavailable; refusing to start the second pipeline" >&2
    exit 1
  fi
fi

APP_ARGS=(--source "$SOURCE" --host 0.0.0.0 --port 8080)
if [[ -n "$SOURCE_2" ]]; then
  APP_ARGS+=(--source-2 "$SOURCE_2")
fi

exec .venv/bin/python -m scout \
  "${APP_ARGS[@]}"
