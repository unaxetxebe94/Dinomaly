#!/usr/bin/env python
"""Preview the ROI crop (``data.roi``) before tiling hours of TIFFs.

For a sample of images it binarises a thumbnail on brightness, keeps the
largest connected bright zone and draws its bounding box.  Tiling only
produces tiles inside that box.

Example:
    uv run python scripts/inspect_roi.py --config configs/top.yaml --limit 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

from dinomaly.coco import load_coco
from dinomaly.roi import (
    RoiSettings,
    brightness_thumbnail,
    compute_roi,
    load_thumbnail_rgb,
)
from dinomaly.tiling import positions_in_roi, tile_positions
from dinomaly.utils import apply_overrides, load_config, seed_everything, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", default=None, help="Only inspect this source name")
    parser.add_argument("--limit", type=int, default=8, help="Images per source (0 = all)")
    parser.add_argument("--out-dir", default="roi_preview", help="Where to write the preview JPGs")
    parser.add_argument("--threshold", type=float, default=None, help="Override data.roi.threshold")
    parser.add_argument("--select", default=None, choices=["area", "width", "height"])
    parser.add_argument("--margin", type=int, default=None)
    parser.add_argument("--preview", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.set)
    data_cfg = config["data"]
    roi_cfg = RoiSettings.from_config(data_cfg.get("roi"))
    if args.threshold is not None:
        roi_cfg.threshold = float(args.threshold)
    if args.select is not None:
        roi_cfg.select = args.select
    if args.margin is not None:
        roi_cfg.margin = int(args.margin)
    if args.preview is not None:
        roi_cfg.preview = int(args.preview)
    roi_cfg.enabled = True
    roi_cfg.validate()

    tile_size = int(args.tile_size or data_cfg.get("tile_size", 1024))
    overlap = int(args.overlap if args.overlap is not None else data_cfg.get("tile_overlap", 128))
    seed = int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))
    seed_everything(seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()

    logger.info(
        f"ROI: threshold={roi_cfg.threshold} select={roi_cfg.select} margin={roi_cfg.margin} "
        f"preview={roi_cfg.preview} | tiles {tile_size}px / overlap {overlap}px"
    )
    rows: list[str] = []
    for source in data_cfg["sources"]:
        name = str(source.get("name") or Path(source["root"]).name)
        if args.source and name != args.source:
            continue
        dataset = load_coco(source["root"], source.get("annotations", "_annotations.coco.json"))
        images = [im for im in dataset.images if im.exists]
        if not images:
            logger.warning(f"[{name}] no images found under {source['root']}")
            continue
        if args.limit > 0 and args.limit < len(images):
            rng = np.random.default_rng(seed)
            chosen = sorted(rng.choice(len(images), size=args.limit, replace=False))
            sample = [images[i] for i in chosen]
        else:
            sample = images

        for image in sample:
            thumb, width, height = load_thumbnail_rgb(image.path, roi_cfg.preview)
            gray, _, _ = brightness_thumbnail(image.path, roi_cfg.preview)
            canvas = Image.fromarray(thumb)
            draw = ImageDraw.Draw(canvas)
            stem = f"{name}_{Path(image.file_name).stem}"
            full_tiles = len(tile_positions(width, tile_size, overlap)) * len(
                tile_positions(height, tile_size, overlap)
            )

            roi = compute_roi(image.path, roi_cfg)
            roi_tiles = full_tiles
            text = f"FULL IMAGE ({full_tiles} tiles)"
            if roi is not None:
                nx = len(positions_in_roi(width, tile_size, overlap, roi.x0, roi.x1))
                ny = len(positions_in_roi(height, tile_size, overlap, roi.y0, roi.y1))
                roi_tiles = nx * ny
                sx, sy = canvas.width / width, canvas.height / height
                draw.rectangle(
                    [round(roi.x0 * sx), round(roi.y0 * sy), round(roi.x1 * sx) - 1, round(roi.y1 * sy) - 1],
                    outline=(255, 40, 40),
                    width=3,
                )
                text = f"ROI {roi_tiles} tiles (full {full_tiles}) cover={roi.coverage:.0%}"
            else:
                logger.warning(f"[{name}] {image.file_name}: nothing above threshold -> full image")
            draw.text((8, 8), text, fill=(255, 40, 40))

            canvas.save(out_dir / f"{stem}.jpg", quality=88)
            Image.fromarray(((gray > roi_cfg.threshold) * 255).astype(np.uint8)).save(
                out_dir / f"{stem}_mask.png"
            )

            if roi is not None:
                line = (
                    f"{image.file_name}: {width}x{height} | comps={roi.num_components} "
                    f"main={roi.component_frac:.1%} | box=({roi.x0},{roi.y0})-({roi.x1},{roi.y1}) "
                    f"| cover={roi.coverage:.1%} | tiles {roi_tiles} (full {full_tiles})"
                )
            else:
                line = f"{image.file_name}: {width}x{height} | no ROI -> full image ({full_tiles} tiles)"
            logger.info(line)
            rows.append(line)

    if rows:
        summary = out_dir / "summary.txt"
        summary.write_text("\n".join(rows) + "\n")
        logger.info(f"Previews: {out_dir}  ({len(rows)} images, summary in {summary})")
    else:
        logger.warning("No images inspected")


if __name__ == "__main__":
    main()
