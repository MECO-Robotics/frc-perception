#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
.venv/bin/python -m pip install \
  --find-links=https://tortall.net/~robotpy/wheels/2023/raspbian/ \
  'pyntcore==2023.4.3.0'
