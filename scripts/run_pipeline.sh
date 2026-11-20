#!/usr/bin/env bash
# Build tiles, train, and evaluate in one go.
#   ./scripts/run_pipeline.sh [config] [workers]
set -euo pipefail

CONFIG="${1:-configs/top-bottom.yaml}"
WORKERS="${2:-6}"

read -r TILES_DIR RUN_DIR < <(uv run python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["data"]["tiles_dir"], cfg["train"]["checkpoint_dir"])
PY
)

if [ -f "$TILES_DIR/manifest.parquet" ]; then
    echo "Tiles already exist in $TILES_DIR; filling in any missing images (use --overwrite to rebuild)."
    uv run python scripts/build_tiles.py --config "$CONFIG" --workers "$WORKERS" --resume
else
    uv run python scripts/build_tiles.py --config "$CONFIG" --workers "$WORKERS"
fi

uv run python scripts/train.py --config "$CONFIG" --workers "$WORKERS" --eval-test

uv run python scripts/evaluate.py --config "$CONFIG" \
    --checkpoint "$RUN_DIR/best.pt" --split test --max-tiles 8000
