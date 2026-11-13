#!/usr/bin/env python
"""Step 3: evaluate a trained checkpoint on the val/test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from dinomaly.data import TileDataset, make_loader, sample_eval_tiles, split_frames
from dinomaly.engine import load_checkpoint
from dinomaly.metrics import EvalConfig, evaluate_loader, format_metrics
from dinomaly.model import build_model
from dinomaly.utils import (
    apply_overrides,
    load_config,
    resolve_device,
    save_json,
    seed_everything,
    setup_logging,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tiles-dir", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--max-tiles", type=int, default=8000, help="0 means every tile (slow)")
    parser.add_argument("--threshold", type=float, default=None, help="Pixel threshold for F1 (default: checkpoint val threshold)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.set)
    tiles_dir = Path(args.tiles_dir or config["data"]["tiles_dir"])
    logger = setup_logging()
    seed = int(config.get("project", {}).get("seed", 42))
    seed_everything(seed)
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes_entries = json.loads((tiles_dir / "classes.json").read_text())
    classes = {int(e["mask_value"]): str(e["name"]) for e in classes_entries}

    model = build_model(config["model"]).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    logger.info(f"Loaded {args.checkpoint} (iter {checkpoint.get('iteration')}, best={checkpoint.get('best_score')})")

    threshold = args.threshold
    if threshold is None:
        threshold = checkpoint.get("pixel_threshold")
    if threshold is not None and not np.isfinite(threshold):
        threshold = None
    if threshold is not None:
        logger.info(f"Using pixel threshold {threshold:.4f} for F1@threshold")

    manifest = pd.read_parquet(tiles_dir / "manifest.parquet")
    frames = split_frames(manifest)
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    crop_size = int(data_cfg.get("crop_size", 392))
    workers = int(args.workers if args.workers is not None else data_cfg.get("num_workers", 8))
    batch_size = int(train_cfg.get("batch_size", 16))
    eval_cfg = EvalConfig(
        score_ratio=float(train_cfg.get("score_ratio", 0.01)),
        image_topk=int(train_cfg.get("image_topk", 1)),
        pro_max_tiles=int(train_cfg.get("pro_max_tiles", 500)),
        pro_size=int(train_cfg.get("pro_size", 128)),
        pro_thresholds=int(train_cfg.get("pro_thresholds", 100)),
    )

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    all_results = {}
    for split in splits:
        frame = frames[split]
        if frame.empty:
            logger.warning(f"Split {split} is empty, skipping")
            continue
        subset = sample_eval_tiles(frame, args.max_tiles if args.max_tiles > 0 else None, seed=seed)
        logger.info(
            f"Evaluating {split}: {len(subset)} tiles ({int(subset['has_defect'].sum())} with defects) "
            f"from {subset['image_uid'].nunique()} images"
        )
        dataset = TileDataset(subset, tiles_dir, crop_size=crop_size, train=False, return_mask=True)
        loader = make_loader(dataset, batch_size, shuffle=False, num_workers=workers)
        result = evaluate_loader(
            model,
            loader,
            device,
            classes=classes,
            cfg=eval_cfg,
            pixel_threshold=threshold,
            amp=bool(train_cfg.get("amp", True)),
            gaussian_sigma=float(train_cfg.get("gaussian_sigma", 4.0)),
        )
        result["split"] = split
        result["checkpoint"] = args.checkpoint
        result["max_tiles"] = args.max_tiles
        result["pixel_threshold"] = threshold
        all_results[split] = result
        logger.info(f"{split.upper()}: {format_metrics(result)}")
        if result.get("pixel", {}).get("per_category"):
            logger.info("Per-category pixel AP (AP / AUROC):")
            for name, values in sorted(result["pixel"]["per_category"].items(), key=lambda kv: kv[1].get("ap") or -1):
                logger.info(
                    f"  {name:24s} AP={values['ap']:.4f} AUROC={values['auroc']:.4f} "
                    f"pixels={values['pixels']}"
                )

    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "eval_metrics.json"
    payload = all_results if len(all_results) > 1 else next(iter(all_results.values()))
    save_json(payload, out)
    logger.info(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
