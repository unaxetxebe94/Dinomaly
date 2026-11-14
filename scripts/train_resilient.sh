#!/usr/bin/env bash
# Train, and if the process dies resume automatically from last.pt.
#
#   ./scripts/train_resilient.sh [config] [max_attempts]
#
# Useful for long runs on this machine: `last.pt` is written every
# train.save_every iterations, so an unexpected exit costs at most that many
# iterations. The final held-out test evaluation runs once training reaches
# train.total_iters.
set -uo pipefail

CONFIG="${1:-configs/top.yaml}"
ATTEMPTS="${2:-5}"

RUN_DIR=$(uv run python - "$CONFIG" <<'PY'
import sys
import yaml

print(yaml.safe_load(open(sys.argv[1]))["train"]["checkpoint_dir"])
PY
)
LOG="$RUN_DIR/train_driver.log"
mkdir -p "$RUN_DIR"

for attempt in $(seq 1 "$ATTEMPTS"); do
    if [ -f "$RUN_DIR/last.pt" ]; then
        echo "[$(date +%H:%M:%S)] attempt $attempt/$ATTEMPTS: resuming from $RUN_DIR/last.pt" | tee -a "$LOG"
        uv run python scripts/train.py --config "$CONFIG" --resume "$RUN_DIR/last.pt" --eval-test
    else
        echo "[$(date +%H:%M:%S)] attempt $attempt/$ATTEMPTS: fresh run" | tee -a "$LOG"
        uv run python scripts/train.py --config "$CONFIG" --eval-test
    fi
    status=$?
    if [ "$status" -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] finished successfully" | tee -a "$LOG"
        exit 0
    fi
    echo "[$(date +%H:%M:%S)] exited with status $status; retrying in 30 s" | tee -a "$LOG"
    sleep 30
done

echo "[$(date +%H:%M:%S)] giving up after $ATTEMPTS attempts" | tee -a "$LOG"
exit 1
