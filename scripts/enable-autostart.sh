#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/xavier/scout-edge
CRON_LINE="@reboot sleep 20 && cd ${ROOT} && nohup ./scripts/start-scout.sh >> ${ROOT}/scout-boot.log 2>&1 < /dev/null &"
TEMP_CRONTAB=$(mktemp)
trap 'rm -f "$TEMP_CRONTAB"' EXIT

crontab -l 2>/dev/null | grep -vE '# SCOUT boot startup|start-scout\.sh' > "$TEMP_CRONTAB" || true
printf '%s\n' '# SCOUT boot startup' "$CRON_LINE" >> "$TEMP_CRONTAB"
crontab "$TEMP_CRONTAB"
echo 'SCOUT boot startup enabled through cron.'
