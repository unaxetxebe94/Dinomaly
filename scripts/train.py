#!/usr/bin/env python
"""Step 2: train Dinomaly on defect-free tiles."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from dinomaly.data import TileDataset, make_loader, sample_eval_tiles, split_frames
from dinomaly.engine import Trainer, load_checkpoint
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


def class_mapping(tiles_dir: Path) -> dict[int, str]:
    path = tiles_dir / "classes.json"
    if not path.exists():
        return {}
    entries = json.loads(path.read_text())
    return {int(e["mask_value"]): str(e["name"]) for e in entries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tiles-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from (e.g. runs/top/last.pt)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--eval-test", action="store_true", help="Evaluate best.pt on the test split after training")
    parser.add_argument("--test-max-tiles", type=int, default=8000)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.set)
    tiles_dir = Path(args.tiles_dir or config["data"]["tiles_dir"])
    run_dir = Path(args.run_dir or config["train"]["checkpoint_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(run_dir / "train.log")
    save_json(config, run_dir / "config.json")

    seed = int(config.get("project", {}).get("seed", 42))
    seed_everything(seed)
    device = resolve_device(args.device)
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB)")

    manifest = pd.read_parquet(tiles_dir / "manifest.parquet")
    frames = split_frames(manifest)
    for name, frame in frames.items():
        logger.info(
            f"split {name:5s}: {len(frame):7d} tiles | defect={int(frame['has_defect'].sum()):5d} "
            f"| images={frame['image_uid'].nunique()}"
        )
    if frames["train"].empty:
        raise SystemExit("The train split is empty")

    train_df = frames["train"]
    train_df = train_df[~train_df["has_defect"]]
    if train_df.empty:
        raise SystemExit("No defect-free training tiles: adjust the tiling/min_std or split")
    logger.info(f"Training on {len(train_df)} defect-free tiles from {train_df['image_uid'].nunique()} images")

    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    crop_size = int(data_cfg.get("crop_size", 392))
    workers = int(args.workers if args.workers is not None else data_cfg.get("num_workers", 8))
    batch_size = int(train_cfg.get("batch_size", 16))

    train_dataset = TileDataset(
        train_df,
        tiles_dir,
        crop_size=crop_size,
        train=True,
        return_mask=False,
        min_scale=float(data_cfg.get("train_min_scale", 0.6)),
        rotation=bool(data_cfg.get("rotation", False)),
    )
    train_loader = make_loader(train_dataset, batch_size, shuffle=True, num_workers=workers, drop_last=True)

    val_max = train_cfg.get("max_eval_tiles", 2500)
    val_df = sample_eval_tiles(frames["val"], val_max, seed=seed)
    logger.info(f"Validation subset: {len(val_df)} tiles ({int(val_df['has_defect'].sum())} with defects)")
    val_dataset = TileDataset(val_df, tiles_dir, crop_size=crop_size, train=False, return_mask=True)
    val_loader = make_loader(val_dataset, batch_size, shuffle=False, num_workers=workers)

    model = build_model(config["model"]).to(device)
    n_trainable = sum(p.numel() for p in model.trainable_parameters)
    logger.info(f"Encoder: {model.encoder.arch} | trainable params: {n_trainable/1e6:.1f}M")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        run_dir=run_dir,
        classes=class_mapping(tiles_dir),
        logger=logger,
    )
    if args.resume:
        trainer.resume(args.resume)
    metrics = trainer.fit()
    save_json(metrics, run_dir / "val_metrics.json")
    logger.info(f"Validation: {format_metrics(metrics)}")

    if args.eval_test and not frames["test"].empty:
        logger.info("Loading best.pt and evaluating the held-out test split ...")
        checkpoint = load_checkpoint(run_dir / "best.pt", model, device=device)
        model.to(device).eval()
        test_df = sample_eval_tiles(frames["test"], args.test_max_tiles, seed=seed)
        logger.info(f"Test subset: {len(test_df)} tiles ({int(test_df['has_defect'].sum())} with defects)")
        test_dataset = TileDataset(test_df, tiles_dir, crop_size=crop_size, train=False, return_mask=True)
        test_loader = make_loader(test_dataset, batch_size, shuffle=False, num_workers=workers)
        eval_cfg = EvalConfig(
            score_ratio=float(train_cfg.get("score_ratio", 0.01)),
            image_topk=int(train_cfg.get("image_topk", 1)),
            pro_max_tiles=int(train_cfg.get("pro_max_tiles", 500)),
            pro_size=int(train_cfg.get("pro_size", 128)),
            pro_thresholds=int(train_cfg.get("pro_thresholds", 100)),
        )
        started = time.time()
        error_tiles = int(train_cfg.get("save_error_tiles", 50))
        test_metrics = evaluate_loader(
            model,
            test_loader,
            device,
            classes=class_mapping(tiles_dir),
            cfg=eval_cfg,
            pixel_threshold=checkpoint.get("pixel_threshold"),
            amp=bool(train_cfg.get("amp", True)),
            gaussian_sigma=float(train_cfg.get("gaussian_sigma", 4.0)),
            error_dir=run_dir / "test_errors" if error_tiles > 0 else None,
            error_tiles=error_tiles,
        )
        test_metrics["elapsed_seconds"] = round(time.time() - started, 1)
        test_metrics["checkpoint"] = str(run_dir / "best.pt")
        test_metrics["pixel_threshold"] = checkpoint.get("pixel_threshold")
        save_json(test_metrics, run_dir / "test_metrics.json")
        logger.info(f"Test: {format_metrics(test_metrics)}")
        errors = test_metrics.get("errors")
        if errors and errors.get("dir"):
            logger.info(
                f"Test error tiles: FP={errors.get('saved_false_positives', 0)}/"
                f"{errors.get('false_positive_tiles_total', 0)} "
                f"FN={errors.get('saved_false_negatives', 0)}/"
                f"{errors.get('false_negative_tiles_total', 0)} -> {errors['dir']}"
            )

    logger.info(f"Artifacts written to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
