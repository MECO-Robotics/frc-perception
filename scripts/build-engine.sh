#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/src/tensorrt/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/targets/aarch64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ONNX="${1:-models/yolo11s_1088.onnx}"
ENGINE="${2:-models/yolo11s_fp16.engine}"
command -v trtexec >/dev/null || { echo 'trtexec is required on a compatible Jetson build host.' >&2; exit 1; }
trtexec --onnx="$ONNX" --saveEngine="$ENGINE" --fp16 --workspace=2048 --timingCacheFile="${ENGINE%.engine}.cache" --buildOnly
trtexec --loadEngine="$ENGINE" --iterations=100 --duration=60
trtexec --loadEngine="$ENGINE" --dumpLayerInfo --exportTimes="${ENGINE%.engine}.timings.json" --iterations=100
echo "TensorRT engine written to $ENGINE"
