#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-yolo11s}"
SIZE="${2:-1088}"
EXPORT_PYTHON="${SCOUT_EXPORT_PYTHON:-python3}"
export YOLO_AUTOINSTALL=false
mkdir -p models
# Use a separate export environment on Jetson to preserve runtime dependencies.
"$EXPORT_PYTHON" -c 'import ultralytics, onnx, onnxslim, onnxruntime' || {
  echo "Install ultralytics, onnx, onnxslim, and onnxruntime in the export environment first." >&2
  exit 1
}
"$EXPORT_PYTHON" - "$MODEL" "$SIZE" <<'PYTHON'
import sys
from pathlib import Path
from ultralytics import YOLO
model, size_spec = sys.argv[1], sys.argv[2].lower()
if 'x' in size_spec:
    width, height = (int(value) for value in size_spec.split('x', 1))
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError('Input width and height must be positive multiples of 32')
    imgsz = (height, width)
    size_name = f'{width}x{height}'
else:
    size = int(size_spec)
    if size <= 0 or size % 32:
        raise ValueError('Input size must be a positive multiple of 32')
    imgsz = size
    size_name = str(size)
checkpoint = Path(model).with_suffix('.pt')
output = YOLO(str(checkpoint)).export(format='onnx', imgsz=imgsz, opset=17,
                                      simplify=True, dynamic=False, nms=False,
                                      device='cpu')
destination = Path('models') / f'{checkpoint.stem}_{size_name}.onnx'
Path(output).replace(destination)
print(f'ONNX model written to {destination}')
PYTHON
