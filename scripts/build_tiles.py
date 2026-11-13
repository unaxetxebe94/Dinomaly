#!/usr/bin/env python
"""Step 1: build tiles + masks from the COCO-annotated large images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dinomaly.tiling import build_tiles
from dinomaly.utils import apply_overrides, load_config, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML config")
    parser.add_argument("--tiles-dir", default=None, help="Override data.tiles_dir")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes (one process per image)")
    parser.add_argument("--limit-images", type=int, default=None, help="Debug: only tile N images per source")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing tiles directory")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep the existing manifest and only tile images that are missing (recommended after a partial run)",
    )
    parser.add_argument("--set", action="append", default=[], help="Override config values, e.g. data.tile_size=512")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.set)
    tiles_dir = Path(args.tiles_dir or config["data"]["tiles_dir"])
    if (tiles_dir / "manifest.parquet").exists() and not (args.overwrite or args.resume):
        raise SystemExit(
            f"{tiles_dir} already contains a manifest. Pass --resume to fill in missing images "
            "or --overwrite to rebuild everything."
        )

    logger = setup_logging(tiles_dir / "build_tiles.log")
    logger.info(f"Config: {json.dumps(config, indent=2, default=str)}")
    build_tiles(
        config,
        tiles_dir,
        workers=args.workers,
        limit_images=args.limit_images,
        overwrite=args.overwrite,
        resume=args.resume,
        log=logger.info,
    )


if __name__ == "__main__":
    main()
