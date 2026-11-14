"""Pixel-, tile- and source-image-level anomaly detection metrics.

Pixel scores are accumulated in a fixed histogram instead of keeping every
pixel in memory.  With 4096 bins over [0, 2] the AUROC/AP/F1 estimates are
accurate to well below the third decimal while memory stays constant, which
matters when a test split contains tens of thousands of 1024px tiles.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from .utils import save_json, topk_mean


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


# ---------------------------------------------------------------------------
# Visual inspection of misclassified tiles
# ---------------------------------------------------------------------------

_HEAT_STOPS = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
_HEAT_COLORS = np.array(
    [[0, 0, 128], [0, 128, 255], [0, 255, 96], [255, 216, 0], [255, 32, 32]],
    dtype=np.float32,
)


def _heatmap_rgb(values: np.ndarray, vmax: float) -> np.ndarray:
    """Blue -> cyan -> green -> yellow -> red, scaled to ``[0, vmax]``."""
    norm = np.clip(np.asarray(values, dtype=np.float32) / max(float(vmax), 1e-8), 0.0, 1.0)
    channels = [np.interp(norm, _HEAT_STOPS, _HEAT_COLORS[:, c]) for c in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


def _denormalize(image: torch.Tensor) -> np.ndarray:
    from .data import IMAGENET_MEAN, IMAGENET_STD

    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    rgb = (image.detach().float() * std + mean).clamp(0, 1).cpu()
    return (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _error_overlay(
    rgb: np.ndarray, values: np.ndarray, vmax: float, mask: np.ndarray | None
) -> np.ndarray:
    """Blend the heatmap over the tile; outline the annotated defect in green."""
    heat = _heatmap_rgb(values, vmax)
    out = (0.45 * rgb.astype(np.float32) + 0.55 * heat.astype(np.float32)).clip(0, 255)
    out = out.astype(np.uint8)
    if mask is not None and bool((mask > 0).any()):
        positive = mask > 0
        edge = ndimage.binary_dilation(positive, iterations=2) & ~positive
        out[edge] = (0, 255, 0)
    return out


def _montage(
    images: list[np.ndarray],
    labels: list[str],
    out_path: Path,
    cols: int,
    border: tuple[int, int, int],
) -> None:
    if not images:
        return
    cell = int(images[0].shape[0])
    pad = 3
    cols = max(1, min(cols, len(images)))
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (cols * cell, rows * cell), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(zip(images, labels)):
        row, col = divmod(i, cols)
        x, y = col * cell, row * cell
        canvas.paste(Image.fromarray(image), (x, y))
        draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline=border, width=pad)
        draw.rectangle([x + pad, y + cell - 16, x + pad + 7 * len(label), y + cell - pad], fill=(0, 0, 0))
        draw.text((x + pad + 2, y + cell - 15), label, fill=(255, 255, 255))
    canvas.save(out_path, quality=92)


@dataclass
class EvalConfig:
    score_ratio: float = 0.01  # top fraction of pixels used as tile score
    image_topk: int = 1  # aggregate top-k tile scores per source image
    pro_max_tiles: int = 500
    pro_size: int = 128
    pro_thresholds: int = 100
    score_max: float = 2.0
    n_bins: int = 4096
    # Save the worst misclassified tiles (with heatmaps) for visual inspection.
    error_tiles: int = 0  # how many false positives *and* false negatives to keep
    error_dir: str | None = None
    error_grid_cols: int = 5


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
        # Bounded candidate pools: worst false alarms and most-missed defects.
        self.error_fp: list[dict[str, Any]] = []
        self.error_fn: list[dict[str, Any]] = []

    @torch.no_grad()
    def update(
        self,
        anomaly_maps: torch.Tensor,
        masks: torch.Tensor,
        image_uids: list[str],
        categories: list[list[int]] | None = None,
        images: torch.Tensor | None = None,
        tile_ids: list[str] | None = None,
    ) -> None:
        """``anomaly_maps``: (B,1,H,W); ``masks``: (B,H,W) uint8 class labels."""
        b = anomaly_maps.shape[0]
        base = self.num_tiles
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

        if self.cfg.error_tiles > 0 and images is not None:
            self._buffer_errors(
                maps_np, masks_np, tile_scores, tile_labels, images, tile_ids, image_uids,
                categories, base,
            )

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

    def _buffer_errors(
        self,
        maps_np: torch.Tensor,
        masks_np: torch.Tensor,
        tile_scores: torch.Tensor,
        tile_labels: torch.Tensor,
        images: torch.Tensor,
        tile_ids: list[str] | None,
        image_uids: list[str],
        categories: list[list[int]],
        base: int,
    ) -> None:
        """Keep the ``error_tiles`` most severe false positives / false negatives.

        Memory stays bounded: only candidates that currently make the cut are
        copied out of the GPU batch.
        """
        cap = int(self.cfg.error_tiles)
        for i in range(maps_np.shape[0]):
            label = int(tile_labels[i])
            score = float(tile_scores[i])
            if not math.isfinite(score):
                continue
            bucket = self.error_fp if label == 0 else self.error_fn
            if len(bucket) >= cap:
                # FP are ranked by highest score, FN by lowest.
                if label == 0 and score <= bucket[-1]["score"]:
                    continue
                if label == 1 and score >= bucket[-1]["score"]:
                    continue
            bucket.append(
                {
                    "tile_id": tile_ids[i] if tile_ids is not None else f"idx{base + i}",
                    "image_uid": image_uids[i],
                    "score": score,
                    "label": label,
                    "categories": [self._cat_name(c) for c in categories[i]],
                    "map": maps_np[i, 0].numpy().astype(np.float16),
                    "mask": masks_np[i].numpy(),
                    "image": _denormalize(images[i]),
                }
            )
            bucket.sort(key=lambda r: r["score"], reverse=label == 0)
            del bucket[cap:]

    @staticmethod
    def _heat_scale(threshold: float) -> float:
        # Fixed scale for every tile so the mosaics are comparable at a glance.
        return max(4.0 * float(threshold), 0.05)

    def _dump_errors(
        self, tile_threshold: float, pixel_threshold: float | None
    ) -> dict[str, Any]:
        out_dir = Path(str(self.cfg.error_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        info: dict[str, Any] = {
            "dir": str(out_dir.resolve()),
            "decision": "tile score = mean of top score_ratio map pixels; anomalous if score > tile_threshold",
            "tile_threshold": None if tile_threshold is None else float(tile_threshold),
            "pixel_threshold": None if pixel_threshold is None else float(pixel_threshold),
        }
        if tile_threshold is None or not math.isfinite(float(tile_threshold)):
            info["skipped"] = "no finite tile threshold (need both classes in the split)"
            save_json(info, out_dir / "errors.json")
            return info

        threshold = float(tile_threshold)
        cap = int(self.cfg.error_tiles)
        vmax = self._heat_scale(threshold)
        info["heatmap_scale"] = vmax

        scores = np.asarray(self.tile_scores, dtype=np.float64)
        labels = np.asarray(self.tile_labels, dtype=np.int64)
        info["false_positive_tiles_total"] = int(((labels == 0) & (scores > threshold)).sum())
        info["false_negative_tiles_total"] = int(((labels == 1) & (scores <= threshold)).sum())

        for kind, bucket, positive in (
            ("fp", self.error_fp, False),
            ("fn", self.error_fn, True),
        ):
            chosen = []
            for record in bucket:  # already ranked worst-first
                is_error = record["score"] > threshold if not positive else record["score"] <= threshold
                if is_error:
                    chosen.append(record)
                if len(chosen) >= cap:
                    break
            entries = []
            overlays: list[np.ndarray] = []
            captions: list[str] = []
            for record in chosen:
                name = f"{kind}_{record['tile_id']}"
                overlay = _error_overlay(
                    record["image"], record["map"].astype(np.float32), vmax, record["mask"]
                )
                Image.fromarray(overlay).save(out_dir / f"{name}.jpg", quality=92)
                Image.fromarray(
                    _heatmap_rgb(record["map"].astype(np.float32), vmax)
                ).save(out_dir / f"{name}_map.png")
                entry = {
                    "tile_id": record["tile_id"],
                    "image_uid": record["image_uid"],
                    "score": round(record["score"], 6),
                    "label": "defect" if positive else "normal",
                    "categories": record["categories"],
                    "overlay": f"{name}.jpg",
                    "heatmap": f"{name}_map.png",
                }
                if bool(record["mask"].any()):
                    Image.fromarray(
                        (record["mask"] > 0).astype(np.uint8) * 255
                    ).save(out_dir / f"{name}_mask.png")
                    entry["mask"] = f"{name}_mask.png"
                entries.append(entry)
                overlays.append(overlay)
                captions.append(f"{record['score']:.4f}")

            info["false_positives" if not positive else "false_negatives"] = entries
            if overlays:
                montage_name = "false_positives.jpg" if not positive else "false_negatives.jpg"
                _montage(
                    overlays,
                    captions,
                    out_dir / montage_name,
                    int(self.cfg.error_grid_cols),
                    (255, 64, 64) if not positive else (64, 200, 255),
                )
                info.setdefault("montages", {})["fp" if not positive else "fn"] = montage_name
            info[f"saved_{'false_positives' if not positive else 'false_negatives'}"] = len(entries)

        save_json(info, out_dir / "errors.json")
        return info

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

        if self.cfg.error_tiles > 0 and self.cfg.error_dir:
            result["errors"] = self._dump_errors(
                tile.get("best_threshold"), pixel_threshold
            )
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
    error_dir: str | Path | None = None,
    error_tiles: int = 0,
) -> dict[str, Any]:
    from .model import anomaly_maps

    if error_dir is not None and error_tiles > 0:
        cfg = replace(cfg, error_dir=str(error_dir), error_tiles=int(error_tiles))
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
                images=images if cfg.error_tiles > 0 else None,
                tile_ids=batch.get("tile_id"),
            )
    return evaluator.compute(pixel_threshold=pixel_threshold)
