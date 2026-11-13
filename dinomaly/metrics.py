"""Pixel-, tile- and source-image-level anomaly detection metrics.

Pixel scores are accumulated in a fixed histogram instead of keeping every
pixel in memory.  With 4096 bins over [0, 2] the AUROC/AP/F1 estimates are
accurate to well below the third decimal while memory stays constant, which
matters when a test split contains tens of thousands of 1024px tiles.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from .utils import topk_mean


def _histogram(values: np.ndarray, n_bins: int, vmax: float) -> np.ndarray:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(n_bins, dtype=np.int64)
    idx = np.clip((values * (n_bins - 1) / vmax).astype(np.int32), 0, n_bins - 1)
    return np.bincount(idx, minlength=n_bins).astype(np.int64)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy 2.0 renamed trapz to trapezoid."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def _auroc_from_hist(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    tp = np.cumsum(pos[::-1], dtype=np.float64)
    fp = np.cumsum(neg[::-1], dtype=np.float64)
    tpr = tp / tp[-1]
    fpr = fp / fp[-1]
    return _trapz(tpr, fpr)


def _pr_from_hist(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Descending score order, so recall increases along the arrays.
    tp = np.cumsum(pos[::-1], dtype=np.float64)
    fp = np.cumsum(neg[::-1], dtype=np.float64)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(tp[-1], 1e-12)
    return precision, recall


def _ap_from_hist(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.sum() == 0:
        return float("nan")
    precision, recall = _pr_from_hist(pos, neg)
    return float(np.sum(np.diff(recall) * precision[1:]))


def _best_f1(pos: np.ndarray, neg: np.ndarray, vmax: float) -> tuple[float, float]:
    if pos.sum() == 0:
        return float("nan"), float("nan")
    precision, recall = _pr_from_hist(pos, neg)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    k = int(np.nanargmax(f1))
    # k indexes the reversed (descending-score) histogram.
    ascending_index = len(pos) - 1 - k
    threshold = (ascending_index + 0.5) * vmax / len(pos)
    return float(f1[k]), float(threshold)


def _f1_at(pos: np.ndarray, neg: np.ndarray, threshold: float, vmax: float) -> float:
    """F1 when every pixel above ``threshold`` is predicted anomalous."""
    if pos.sum() == 0:
        return float("nan")
    k = int(np.clip(threshold * (len(pos) - 1) / vmax, 0, len(pos) - 1))
    tp = float(pos[k:].sum())
    fp = float(neg[k:].sum())
    fn = float(pos[:k].sum())
    return 2 * tp / max(2 * tp + fp + fn, 1e-12)


def _finite(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y_score)
    if mask.all():
        return y_true, y_score
    return y_true[mask], y_score[mask]


def _score_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    y_true, y_score = _finite(np.asarray(y_true), np.asarray(y_score))
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan"), float("nan")
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    if len(thresholds) == 0:
        return float("nan"), float("nan")
    k = int(np.nanargmax(f1[:-1]))
    return float(f1[k]), float(thresholds[k])


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true, y_score = _finite(np.asarray(y_true), np.asarray(y_score))
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true, y_score = _finite(np.asarray(y_true), np.asarray(y_score))
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_pro(
    masks: np.ndarray,
    amaps: np.ndarray,
    num_thresholds: int = 100,
    max_fpr: float = 0.3,
) -> float:
    """Area under the per-region overlap (PRO) curve, computed with scipy only."""
    assert masks.shape == amaps.shape and masks.ndim == 3
    regions_per_tile = []
    for mask in masks:
        labels, n = ndimage.label(mask > 0)
        if n == 0:
            regions_per_tile.append(None)
            continue
        flat = labels.reshape(-1)
        areas = np.bincount(flat, minlength=n + 1)[1:].astype(np.float64)
        regions_per_tile.append((flat, n, areas))

    if all(r is None for r in regions_per_tile):
        return float("nan")

    lo, hi = float(amaps.min()), float(amaps.max())
    if hi <= lo:
        return float("nan")
    thresholds = np.linspace(lo, hi, num_thresholds)[:-1]

    pros = np.full(len(thresholds), np.nan)
    fprs = np.full(len(thresholds), np.nan)
    for ti, threshold in enumerate(thresholds):
        tile_pros: list[float] = []
        fp_total = 0.0
        neg_total = 0.0
        for amap, mask, regions in zip(amaps, masks, regions_per_tile):
            binary = amap > threshold
            neg = ~(mask > 0)
            fp_total += float(np.logical_and(binary, neg).sum())
            neg_total += float(neg.sum())
            if regions is None:
                continue
            flat, n, areas = regions
            tp = np.bincount(flat, weights=binary.reshape(-1), minlength=n + 1)[1:]
            tile_pros.extend((tp / areas).tolist())
        if tile_pros:
            pros[ti] = float(np.mean(tile_pros))
        fprs[ti] = fp_total / max(neg_total, 1.0)

    valid = ~np.isnan(pros)
    if valid.sum() < 2:
        return float("nan")
    pros, fprs = pros[valid], fprs[valid]
    keep = fprs <= max_fpr
    if keep.sum() < 2:
        keep = np.ones_like(fprs, dtype=bool)
    pros, fprs = pros[keep], fprs[keep]
    order = np.argsort(fprs)
    pros, fprs = pros[order], fprs[order]
    if fprs[-1] <= 0:
        return float("nan")
    fprs = fprs / fprs[-1]
    return _trapz(pros, fprs)


@dataclass
class EvalConfig:
    score_ratio: float = 0.01  # top fraction of pixels used as tile score
    image_topk: int = 1  # aggregate top-k tile scores per source image
    pro_max_tiles: int = 500
    pro_size: int = 128
    pro_thresholds: int = 100
    score_max: float = 2.0
    n_bins: int = 4096


@dataclass
class AnomalyEvaluator:
    classes: dict[int, str] | None = None
    cfg: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self) -> None:
        self.pos_hist = np.zeros(self.cfg.n_bins, dtype=np.int64)
        self.neg_hist = np.zeros(self.cfg.n_bins, dtype=np.int64)
        self.cat_pos_hist: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(self.cfg.n_bins, dtype=np.int64)
        )
        self.cat_pixels: dict[int, int] = defaultdict(int)

        self.tile_scores: list[float] = []
        self.tile_labels: list[int] = []
        self.tile_categories: list[list[int]] = []
        self.image_records: dict[str, list[tuple[float, int]]] = defaultdict(list)

        self.pro_maps: list[np.ndarray] = []
        self.pro_masks: list[np.ndarray] = []
        self.num_tiles = 0
        self.num_defect_tiles = 0
        self.num_pixels = 0
        self.num_nonfinite_tiles = 0

    @torch.no_grad()
    def update(
        self,
        anomaly_maps: torch.Tensor,
        masks: torch.Tensor,
        image_uids: list[str],
        categories: list[list[int]] | None = None,
    ) -> None:
        """``anomaly_maps``: (B,1,H,W); ``masks``: (B,H,W) uint8 class labels."""
        b = anomaly_maps.shape[0]
        maps_np = anomaly_maps.detach().float().cpu()
        masks_np = masks.detach().cpu().to(torch.uint8)
        nonfinite = ~torch.isfinite(maps_np).reshape(b, -1).all(dim=1)
        if bool(nonfinite.any()):
            self.num_nonfinite_tiles += int(nonfinite.sum())
            # A diverged model must not crash evaluation; clamp and report it.
            maps_np = torch.nan_to_num(maps_np, nan=0.0, posinf=self.cfg.score_max, neginf=0.0)
        flat_scores = maps_np.reshape(b, -1).numpy()
        flat_masks = masks_np.reshape(b, -1).numpy()

        scores = flat_scores.reshape(-1)
        labels = flat_masks.reshape(-1)
        self.num_pixels += scores.size
        positive = labels > 0
        self.pos_hist += _histogram(scores[positive], self.cfg.n_bins, self.cfg.score_max)
        self.neg_hist += _histogram(scores[~positive], self.cfg.n_bins, self.cfg.score_max)
        for cat in np.unique(labels):
            if cat == 0:
                continue
            cat = int(cat)
            selected = scores[labels == cat]
            self.cat_pos_hist[cat] += _histogram(selected, self.cfg.n_bins, self.cfg.score_max)
            self.cat_pixels[cat] += int(selected.size)

        tile_scores = topk_mean(maps_np.reshape(b, -1), self.cfg.score_ratio)
        tile_labels = (masks_np.reshape(b, -1) > 0).any(dim=1).to(torch.int64)
        self.tile_scores.extend(tile_scores.tolist())
        self.tile_labels.extend(tile_labels.tolist())
        self.num_tiles += b
        self.num_defect_tiles += int(tile_labels.sum())

        if categories is None:
            categories = [
                [int(c) for c in np.unique(flat_masks[i]) if c > 0] for i in range(b)
            ]
        self.tile_categories.extend(categories)

        for i in range(b):
            score = float(tile_scores[i])
            label = int(tile_labels[i])
            self.image_records[image_uids[i]].append((score, label))
            if label and len(self.pro_maps) < self.cfg.pro_max_tiles:
                score_map = maps_np[i : i + 1]
                mask = masks_np[i : i + 1].float()
                small_map = F.interpolate(
                    score_map, size=(self.cfg.pro_size, self.cfg.pro_size), mode="area"
                )[0, 0].numpy()
                small_mask = F.interpolate(
                    mask.unsqueeze(1), size=(self.cfg.pro_size, self.cfg.pro_size), mode="nearest"
                )[0, 0].numpy()
                if small_mask.max() > 0:
                    self.pro_maps.append(small_map)
                    self.pro_masks.append(small_mask > 0)

    def _cat_name(self, cat: int) -> str:
        if self.classes and cat in self.classes:
            return self.classes[cat]
        return f"class_{cat}"

    def compute(self, pixel_threshold: float | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "num_tiles": self.num_tiles,
            "num_defect_tiles": self.num_defect_tiles,
            "num_nonfinite_tiles": self.num_nonfinite_tiles,
            "num_pixels": self.num_pixels,
            "pixel": {},
            "tile": {},
            "image": {},
        }

        pixel = result["pixel"]
        pixel["auroc"] = _auroc_from_hist(self.pos_hist, self.neg_hist)
        pixel["ap"] = _ap_from_hist(self.pos_hist, self.neg_hist)
        pixel["f1"], pixel["best_threshold"] = _best_f1(
            self.pos_hist, self.neg_hist, self.cfg.score_max
        )
        if pixel_threshold is not None and np.isfinite(pixel_threshold):
            pixel["f1_at_threshold"] = _f1_at(
                self.pos_hist, self.neg_hist, float(pixel_threshold), self.cfg.score_max
            )
        pixel["per_category"] = {}
        for cat, pos_hist in sorted(self.cat_pos_hist.items()):
            pixel["per_category"][self._cat_name(cat)] = {
                "auroc": _auroc_from_hist(pos_hist, self.neg_hist),
                "ap": _ap_from_hist(pos_hist, self.neg_hist),
                "f1": _best_f1(pos_hist, self.neg_hist, self.cfg.score_max)[0],
                "pixels": self.cat_pixels[cat],
            }

        tile_scores = np.asarray(self.tile_scores, dtype=np.float64)
        tile_labels = np.asarray(self.tile_labels, dtype=np.int64)
        tile = result["tile"]
        tile["auroc"] = _safe_auc(tile_labels, tile_scores)
        tile["ap"] = _safe_ap(tile_labels, tile_scores)
        tile["f1"], tile["best_threshold"] = _score_f1(tile_labels, tile_scores)
        if self.tile_categories:
            tile["per_category"] = {}
            cats = sorted({c for row in self.tile_categories for c in row})
            for cat in cats:
                y = np.array([cat in row for row in self.tile_categories], dtype=np.int64)
                counts = int(y.sum())
                if counts == 0:
                    continue
                tile["per_category"][self._cat_name(cat)] = {
                    "auroc": _safe_auc(y, tile_scores),
                    "ap": _safe_ap(y, tile_scores),
                    "positive_tiles": counts,
                }

        image_scores: list[float] = []
        image_labels: list[int] = []
        for records in self.image_records.values():
            records.sort(key=lambda r: r[0], reverse=True)
            k = max(1, int(self.cfg.image_topk))
            score = float(np.mean([r[0] for r in records[:k]]))
            label = int(any(r[1] for r in records))
            image_scores.append(score)
            image_labels.append(label)
        image_scores_arr = np.asarray(image_scores, dtype=np.float64)
        image_labels_arr = np.asarray(image_labels, dtype=np.int64)
        image = result["image"]
        image["auroc"] = _safe_auc(image_labels_arr, image_scores_arr)
        image["ap"] = _safe_ap(image_labels_arr, image_scores_arr)
        image["f1"], image["best_threshold"] = _score_f1(image_labels_arr, image_scores_arr)
        image["num_images"] = len(image_labels_arr)
        image["num_defect_images"] = int(image_labels_arr.sum())

        if len(self.pro_maps) >= 2:
            result["pro"] = compute_pro(
                np.stack(self.pro_masks),
                np.stack(self.pro_maps),
                num_thresholds=self.cfg.pro_thresholds,
            )
            result["pro_tiles"] = len(self.pro_maps)
        else:
            result["pro"] = float("nan")
            result["pro_tiles"] = len(self.pro_maps)
        return result


def format_metrics(result: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "  n/a " if not isinstance(value, float) or math.isnan(value) else f"{value:.4f}"

    lines = [
        f"tiles={result['num_tiles']} (defect={result['num_defect_tiles']})"
        + (
            f" NONFINITE={result['num_nonfinite_tiles']}"
            if result.get("num_nonfinite_tiles")
            else ""
        ),
        (
            f"I-AUROC={fmt(result['image'].get('auroc'))} I-AP={fmt(result['image'].get('ap'))} "
            f"I-F1={fmt(result['image'].get('f1'))}"
        ),
        (
            f"T-AUROC={fmt(result['tile'].get('auroc'))} T-AP={fmt(result['tile'].get('ap'))} "
            f"T-F1={fmt(result['tile'].get('f1'))}"
        ),
        (
            f"P-AUROC={fmt(result['pixel'].get('auroc'))} P-AP={fmt(result['pixel'].get('ap'))} "
            f"P-F1={fmt(result['pixel'].get('f1'))} PRO={fmt(result.get('pro'))}"
        ),
    ]
    if "f1_at_threshold" in result["pixel"]:
        lines.append(f"P-F1@val-thr={fmt(result['pixel']['f1_at_threshold'])}")
    return " | ".join(lines)


def evaluate_loader(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    classes: dict[int, str] | None,
    cfg: EvalConfig,
    pixel_threshold: float | None = None,
    amp: bool = True,
    gaussian_sigma: float = 4.0,
) -> dict[str, Any]:
    from .model import anomaly_maps

    evaluator = AnomalyEvaluator(classes=classes, cfg=cfg)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                en, de = model(images)
            amap = anomaly_maps(en, de, images.shape[-1], sigma=gaussian_sigma)
            evaluator.update(
                amap.float(),
                batch["mask"],
                batch["image_uid"],
                batch.get("categories"),
            )
    return evaluator.compute(pixel_threshold=pixel_threshold)
