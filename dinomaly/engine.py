"""Training loop and checkpoint helpers."""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from .losses import global_cosine_hm_percent
from .metrics import EvalConfig, evaluate_loader, format_metrics, to_tracking_metrics
from .model import Dinomaly
from .utils import AverageMeter, save_json


class WarmCosineScheduler(_LRScheduler):
    """Linear warmup followed by a cosine decay, as in the Dinomaly paper."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_value: float,
        final_value: float,
        total_iters: int,
        warmup_iters: int = 0,
        start_warmup_value: float = 0.0,
    ) -> None:
        self.final_value = final_value
        self.total_iters = total_iters
        warmup = np.linspace(start_warmup_value, base_value, max(1, warmup_iters))
        iters = np.arange(max(1, total_iters - warmup_iters))
        cosine = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / max(1, len(iters))))
        self.schedule = np.concatenate((warmup, cosine)) if warmup_iters > 0 else cosine
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        if self.last_epoch >= len(self.schedule):
            return [self.final_value for _ in self.base_lrs]
        return [float(self.schedule[self.last_epoch]) for _ in self.base_lrs]


def infinite(loader: Iterable) -> Iterable:
    while True:
        yield from loader


def trainable_state_dict(model: Dinomaly) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("encoder.")}


def save_checkpoint(
    path: str | Path,
    model: Dinomaly,
    optimizer: torch.optim.Optimizer | None,
    scheduler: _LRScheduler | None,
    iteration: int,
    best_score: float,
    pixel_threshold: float | None,
    config: dict[str, Any],
    classes: dict[int, str] | None,
    metrics: dict[str, Any] | None = None,
    image_threshold: float | None = None,
    score_name: str | None = None,
    tile_threshold: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = trainable_state_dict(model)
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise RuntimeError(
            f"Refusing to save {path}: model weights contain NaN/Inf (training diverged). "
            "Resume from the last good checkpoint."
        )
    torch.save(
        {
            "model": state,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "iteration": iteration,
            "best_score": best_score,
            "pixel_threshold": pixel_threshold,
            "image_threshold": image_threshold,
            "tile_threshold": tile_threshold,
            "score_name": score_name,
            "config": config,
            "classes": classes,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: Dinomaly,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: _LRScheduler | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}")
    missing = [k for k in missing if not k.startswith("encoder.")]
    if missing:
        raise RuntimeError(f"Checkpoint is missing model weights (architecture mismatch?): {missing[:5]}")
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def _get_metric(metrics: dict[str, Any], dotted: str) -> float:
    node: Any = metrics
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return float("nan")
        node = node[part]
    try:
        return float(node)
    except (TypeError, ValueError):
        return float("nan")


class Trainer:
    def __init__(
        self,
        model: Dinomaly,
        train_loader: Iterable,
        val_loader: Iterable,
        device: torch.device,
        config: dict[str, Any],
        run_dir: str | Path,
        classes: dict[int, str] | None = None,
        logger: Any = None,
        tracker: Any = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.classes = classes
        self.log = logger.info if logger is not None else print
        self.tracker = tracker

        train_cfg = config.get("train", {})
        self.total_iters = int(train_cfg.get("total_iters", 10000))
        self.batch_size = int(train_cfg.get("batch_size", 16))
        self.eval_every = int(train_cfg.get("eval_every", 1000))
        self.log_every = int(train_cfg.get("log_every", 50))
        self.amp = bool(train_cfg.get("amp", True))
        self.grad_clip = float(train_cfg.get("grad_clip", 0.1))
        self.save_every = int(train_cfg.get("save_every", 500))
        self.hard_mining_p = float(train_cfg.get("hard_mining_p", 0.9))
        self.hard_mining_factor = float(train_cfg.get("hard_mining_factor", 0.1))
        self.select_metric = str(train_cfg.get("select_metric", "image.auroc"))
        self.max_eval_tiles = train_cfg.get("max_eval_tiles")
        self.sigma = float(train_cfg.get("gaussian_sigma", 4.0))

        self.optimizer = torch.optim.AdamW(
            model.trainable_parameters,
            lr=float(train_cfg.get("lr", 2e-3)),
            betas=(0.9, 0.999),
            eps=float(train_cfg.get("optimizer_eps", 1e-10)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            amsgrad=bool(train_cfg.get("amsgrad", True)),
        )
        self.scheduler = WarmCosineScheduler(
            self.optimizer,
            base_value=float(train_cfg.get("lr", 2e-3)),
            final_value=float(train_cfg.get("final_lr", 2e-4)),
            total_iters=self.total_iters,
            warmup_iters=int(train_cfg.get("warmup_iters", 100)),
        )
        self.eval_cfg = EvalConfig(
            score_ratio=float(train_cfg.get("score_ratio", 0.01)),
            image_topk=int(train_cfg.get("image_topk", 1)),
            pro_max_tiles=int(train_cfg.get("pro_max_tiles", 500)),
            pro_size=int(train_cfg.get("pro_size", 128)),
            pro_thresholds=int(train_cfg.get("pro_thresholds", 100)),
        )

        self.iteration = 0
        self.best_score = -math.inf
        self.score_name = self.select_metric
        self.pixel_threshold: float | None = None
        self.image_threshold: float | None = None
        self.tile_threshold: float | None = None
        self.history_path = self.run_dir / "history.csv"
        self.history_header = [
            "iteration",
            "loss",
            "lr",
            "seconds",
            "pixel_auroc",
            "pixel_ap",
            "pixel_f1",
            "pro",
            "tile_auroc",
            "tile_ap",
            "image_auroc",
            "image_ap",
            "image_f1",
            "threshold",
        ]
        if not self.history_path.exists():
            self._write_history_header()

    def _write_history_header(self) -> None:
        with open(self.history_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(self.history_header)

    def resume(self, checkpoint_path: str | Path) -> None:
        checkpoint = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler, device=self.device
        )
        self.iteration = int(checkpoint.get("iteration", 0))
        self.best_score = float(checkpoint.get("best_score", -math.inf))
        self.pixel_threshold = checkpoint.get("pixel_threshold")
        self.image_threshold = checkpoint.get("image_threshold")
        self.tile_threshold = checkpoint.get("tile_threshold")
        self.score_name = checkpoint.get("score_name") or self.select_metric
        self.log(
            f"Resumed from {checkpoint_path} at iter {self.iteration} "
            f"(best {self.score_name}={self.best_score:.4f})"
        )

    def evaluate(self) -> dict[str, Any]:
        result = evaluate_loader(
            self.model,
            self.val_loader,
            self.device,
            classes=self.classes,
            cfg=self.eval_cfg,
            pixel_threshold=self.pixel_threshold,
            amp=self.amp,
            gaussian_sigma=self.sigma,
        )
        self.pixel_threshold = result["pixel"].get("best_threshold")
        if self.pixel_threshold is not None and not np.isfinite(self.pixel_threshold):
            self.pixel_threshold = None
        self.image_threshold = result["image"].get("best_threshold")
        if self.image_threshold is not None and not np.isfinite(self.image_threshold):
            self.image_threshold = None
        self.tile_threshold = result["tile"].get("threshold")
        if self.tile_threshold is not None and not np.isfinite(self.tile_threshold):
            self.tile_threshold = None
        return result

    def _save(self, name: str, metrics: dict[str, Any] | None = None) -> None:
        save_checkpoint(
            self.run_dir / name,
            self.model,
            self.optimizer,
            self.scheduler,
            self.iteration,
            self.best_score,
            self.pixel_threshold,
            self.config,
            self.classes,
            metrics=metrics,
            image_threshold=self.image_threshold,
            score_name=self.score_name,
            tile_threshold=self.tile_threshold,
        )

    def fit(self) -> dict[str, Any]:
        if self.iteration == 0:
            # Fresh run: do not mix rows from a previous run in the same dir.
            self._write_history_header()
        self.log(
            f"Training Dinomaly for {self.total_iters} iterations "
            f"(batch={self.batch_size}, amp={self.amp}, device={self.device})"
        )
        data_iter = infinite(self.train_loader)
        meter = AverageMeter()
        skipped = 0
        consecutive_skips = 0
        grad_norm = float("nan")
        started = time.time()
        last_log = started
        last_metrics: dict[str, Any] = {}

        progress = tqdm(total=self.total_iters, initial=self.iteration, desc="train", dynamic_ncols=True)
        while self.iteration < self.total_iters:
            batch = next(data_iter)
            images = batch["image"].to(self.device, non_blocking=True)

            self.model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and self.device.type == "cuda"):
                en, de = self.model(images)
            # The loss (and the gradient hook) is evaluated in fp32.
            en = [t.float() for t in en]
            de = [t.float() for t in de]

            p = min(self.hard_mining_p, self.hard_mining_p * self.iteration / 1000.0)
            loss = global_cosine_hm_percent(en, de, p=p, factor=self.hard_mining_factor)

            self.optimizer.zero_grad(set_to_none=True)
            finite_loss = bool(loss.detach().isfinite().item())
            if finite_loss:
                loss.backward()
                grad_norm = float(
                    nn.utils.clip_grad_norm_(
                        self.model.trainable_parameters,
                        max_norm=self.grad_clip if self.grad_clip > 0 else float("inf"),
                    )
                )
                finite_loss = math.isfinite(grad_norm)
            if finite_loss:
                self.optimizer.step()
                consecutive_skips = 0
            else:
                # Never let one bad batch poison the weights: without this a
                # single non-finite gradient turns every following loss into NaN.
                skipped += 1
                consecutive_skips += 1
                self.optimizer.zero_grad(set_to_none=True)
                if skipped <= 5 or skipped % 50 == 0:
                    self.log(
                        f"WARNING: non-finite loss/gradient at iter {self.iteration + 1} "
                        f"(skipped={skipped}); optimiser step skipped"
                    )
                if consecutive_skips >= 25:
                    if self.amp:
                        self.log(
                            "WARNING: 25 consecutive non-finite steps; disabling AMP and continuing in fp32"
                        )
                        self.amp = False
                        consecutive_skips = 0
                    else:
                        raise RuntimeError(
                            "Training keeps producing non-finite losses in fp32. "
                            "Resume from the last good checkpoint (last.pt/best.pt)."
                        )
            self.scheduler.step()

            self.iteration += 1
            if math.isfinite(loss.detach().item()):
                meter.update(loss.detach().item())
            progress.update(1)
            progress.set_postfix(loss=f"{meter.avg:.4f}", skipped=skipped)

            if self.iteration % self.log_every == 0:
                now = time.time()
                lr = self.optimizer.param_groups[0]["lr"]
                self.log(
                    f"iter {self.iteration}/{self.total_iters} loss={meter.avg:.4f} "
                    f"grad={grad_norm:.2e} skipped={skipped} "
                    f"lr={lr:.2e} {self.log_every/(now-last_log):.1f} it/s"
                )
                if self.tracker is not None:
                    self.tracker.log_metrics(
                        {
                            "train/loss": meter.avg,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                            "train/skipped_steps": skipped,
                            "train/it_per_sec": self.log_every / max(now - last_log, 1e-9),
                        },
                        step=self.iteration,
                    )
                meter = AverageMeter()
                last_log = now

            if (
                self.save_every > 0
                and self.iteration % self.save_every == 0
                and not (self.eval_every > 0 and self.iteration % self.eval_every == 0)
            ):
                self._save("last.pt", last_metrics or None)

            if self.eval_every > 0 and self.iteration % self.eval_every == 0:
                metrics = self.evaluate()
                score_name = self.select_metric
                score = _get_metric(metrics, score_name)
                if not np.isfinite(score):
                    # e.g. a val split with no normal images cannot produce image AUROC
                    fallback = "tile.auroc" if self.select_metric != "tile.auroc" else "pixel.auroc"
                    score_name = fallback
                    score = _get_metric(metrics, fallback)
                self.score_name = score_name
                last_metrics = metrics
                is_best = np.isfinite(score) and score > self.best_score
                if is_best:
                    self.best_score = score
                self.log(
                    f"[eval @ {self.iteration}] {format_metrics(metrics)} "
                    f"({score_name}={score:.4f}{' *best*' if is_best else ''})"
                )
                self._append_history(metrics)
                self._track_val(metrics, score_name, score)
                self._save("last.pt", metrics)
                if is_best:
                    self._save("best.pt", metrics)

        progress.close()
        if not last_metrics:
            last_metrics = self.evaluate()
            self._append_history(last_metrics)
            self._track_val(last_metrics, self.select_metric, _get_metric(last_metrics, self.select_metric))
        self._save("last.pt", last_metrics)
        if not np.isfinite(self.best_score):
            score_name = self.select_metric
            score = _get_metric(last_metrics, score_name)
            if not np.isfinite(score):
                score_name = "tile.auroc" if self.select_metric != "tile.auroc" else "pixel.auroc"
                score = _get_metric(last_metrics, score_name)
            self.score_name = score_name
            self.best_score = score
            self._save("best.pt", last_metrics)
        self.log(
            f"Training done in {(time.time()-started)/60:.1f} min; "
            f"best {self.score_name}={self.best_score:.4f}; skipped_steps={skipped}; run dir: {self.run_dir}"
        )
        save_json(last_metrics, self.run_dir / f"metrics_iter_{self.iteration}.json")
        return last_metrics

    def _track_val(self, metrics: dict[str, Any], score_name: str, score: float) -> None:
        if self.tracker is None:
            return
        payload = to_tracking_metrics(metrics, "val")
        payload["val"]["select_score"] = score
        payload["val"]["best_select_score"] = self.best_score
        self.tracker.log_metrics(payload, step=self.iteration)
        self.tracker.set_tags({"select_metric": score_name})

    def _append_history(self, metrics: dict[str, Any]) -> None:
        row = [
            self.iteration,
            "",
            self.optimizer.param_groups[0]["lr"],
            "",
            metrics["pixel"].get("auroc"),
            metrics["pixel"].get("ap"),
            metrics["pixel"].get("f1"),
            metrics.get("pro"),
            metrics["tile"].get("auroc"),
            metrics["tile"].get("ap"),
            metrics["image"].get("auroc"),
            metrics["image"].get("ap"),
            metrics["image"].get("f1"),
            self.pixel_threshold,
        ]
        with open(self.history_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(row)
