#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/src/tensorrt/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/targets/aarch64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
LOG_DIR="${SCOUT_LOG_DIR:-$(pwd)/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/provision-tensorrt-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1
on_exit() {
  status=$?
  echo "provision exit status: $status"
  echo "provision log: $LOG_FILE"
  exit "$status"
}
trap on_exit EXIT
echo "provision start: $(date -u --iso-8601=seconds)"
echo "host: $(hostname)"
echo "kernel: $(uname -a)"
echo "python: $(command -v python3 || true)"
python3 --version || true
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-nvcc-11-4 cuda-cudart-dev-11-4 cuda-profiler-api-11-4 libcurand-dev-11-4 libcudla-11-4 python3-libnvinfer python3-libnvinfer-dev libnvinfer-bin python3-dev libboost-python-dev
echo "CUDA headers:"
ls -l /usr/local/cuda/include/cuda.h
export CUDA_ROOT=/usr/local/cuda
export CUDA_INC_DIR=/usr/local/cuda/targets/aarch64-linux/include
export CUDA_LIB_DIR=/usr/local/cuda/targets/aarch64-linux/lib
export CXXFLAGS="-I${CUDA_INC_DIR} ${CXXFLAGS:-}"
export LDFLAGS="-L${CUDA_LIB_DIR} ${LDFLAGS:-}"
SCOUT_PYTHON="${SCOUT_PYTHON:-$(pwd)/.venv/bin/python}"
if [[ ! -x "$SCOUT_PYTHON" ]]; then
  echo "Create the SCOUT virtual environment with scripts/install-jetson.sh first." >&2
  exit 1
fi
# TensorRT 8.5 uses NumPy aliases removed in 1.24; PyCUDA needs a newer
# NumPy C API than the Jetson system package provides.
"$SCOUT_PYTHON" -m pip install 'numpy==1.23.5'
CUDA_ROOT="$CUDA_ROOT" "$SCOUT_PYTHON" -m pip install --verbose --upgrade pycuda
"$SCOUT_PYTHON" - <<'PY'
import tensorrt as trt
print('TensorRT Python', trt.__version__)
import pycuda.driver as cuda
print('PyCUDA import OK')
PY
command -v trtexec
trtexec --help
