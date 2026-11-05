#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip python3-opencv
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install 'fastapi>=0.115,<1' 'uvicorn[standard]>=0.30,<1'
mkdir -p "$HOME/.config/systemd/user"
cp scripts/scout-edge.service "$HOME/.config/systemd/user/scout-edge.service"
systemctl --user daemon-reload
if ! systemctl --user enable --now scout-edge.service 2>/dev/null; then
  # Headless SSH sessions often have no user D-Bus. Keep the service usable
  # without requiring a desktop login or system-wide unit installation.
  if [ -f scout.pid ] && kill -0 "$(cat scout.pid)" 2>/dev/null; then kill "$(cat scout.pid)"; fi
  nohup .venv/bin/python -m scout --source "${SCOUT_SOURCE:-0}" --host 0.0.0.0 --port 8080 > scout.log 2>&1 &
  echo $! > scout.pid
  sleep 2
fi
echo "SCOUT is running at http://$(hostname -I | awk '{print $1}'):8080"
