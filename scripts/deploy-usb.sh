#!/usr/bin/env bash
set -euo pipefail
TARGET="${1:-xavier@192.168.55.1}"
REMOTE=/home/xavier/scout-edge
ssh "$TARGET" "mkdir -p $REMOTE"
rsync -az --delete --exclude .git --exclude .venv ./ "$TARGET:$REMOTE/"
echo "SCOUT copied to $TARGET:$REMOTE"
echo "Run remotely: ssh -t $TARGET 'cd $REMOTE && sudo ./scripts/install-jetson.sh'"
