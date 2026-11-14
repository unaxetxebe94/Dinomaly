#!/usr/bin/env python
"""Step 4 (optional): score one large image and write a down-sampled heatmap.

Example:
    uv run python scripts/predict.py \
        --config configs/top.yaml \
        --checkpoint runs/top/best.pt \
        --image "/run/media/.../Top/1008_111213_top_native.tiff" \
        --out-dir predictions/1008_111213
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyvips
import torch
from PIL import Image

from dinomaly.data import build_transforms
from dinomaly.engine import load_checkpoint
from dinomaly.model import anomaly_maps, build_model
from dinomaly.roi import RoiSettings, compute_roi
from dinomaly.tiling import positions_in_roi, tile_positions
from dinomaly.utils import (
    apply_overrides,
    load_config,
    resolve_device,
    save_json,
    setup_logging,
    topk_mean,
)


def _to_rgb_vips(path: str) -> pyvips.Image:
    img = pyvips.Image.new_from_file(path, access="random")
    if img.bands == 1:
        img = img.bandjoin([img, img, img])
    elif img.bands > 3:
        img = img[0:3]
    if img.interpretation != "srgb":
        img = img.colourspace("srgb")
    return img


def _colormap(gray: np.ndarray) -> np.ndarray:
    anchors = [0.0, 0.25, 0.5, 0.75, 1.0]
    red = np.interp(gray, anchors, [0, 0, 0, 255, 255])
    green = np.interp(gray, anchors, [0, 255, 255, 255, 0])
    blue = np.interp(gray, anchors, [128, 255, 0, 0, 0])
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    out_dir: Path,
    device: torch.device,
    tile_size: int,
    overlap: int,
    crop_size: int,
    downsample: int,
    threshold: float | None,
    amp: bool = True,
    sigma: float = 4.0,
    batch_size: int = 16,
    roi_settings: RoiSettings | None = None,
    logger=None,
) -> dict:
    log = logger.info if logger else print
    img = _to_rgb_vips(str(image_path))
    width, height = int(img.width), int(img.height)
    roi = None
    if roi_settings is not None and roi_settings.enabled:
        try:
            roi = compute_roi(image_path, roi_settings)
        except Exception as exc:  # noqa: BLE001 - fall back to the full image
            log(f"{image_path.name}: ROI detection failed ({exc}); using the full image")
    if roi is not None:
        xs = positions_in_roi(width, tile_size, overlap, roi.x0, roi.x1)
        ys = positions_in_roi(height, tile_size, overlap, roi.y0, roi.y1)
        log(f"{image_path.name}: ROI box=({roi.x0},{roi.y0})-({roi.x1},{roi.y1}) coverage={roi.coverage:.1%}")
    else:
        xs = tile_positions(width, tile_size, overlap)
        ys = tile_positions(height, tile_size, overlap)
    positions = [(x, y) for y in ys for x in xs]

    out_tile = max(1, tile_size // downsample)
    canvas_w = int(np.ceil(width / downsample))
    canvas_h = int(np.ceil(height / downsample))
    heat = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    transform = build_transforms(crop_size, train=False)
    records: list[dict] = []
    # A Hann window suppresses ViT border effects and makes overlapping tiles
    # blend instead of leaving visible seams.
    window = np.outer(np.hanning(out_tile), np.hanning(out_tile)).astype(np.float32) + 1e-3
    center = int(out_tile * 0.1)
    log(f"{image_path.name}: {width}x{height}, {len(positions)} tiles")

    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start : start + batch_size]
        tensors = []
        for x, y in batch_positions:
            tile = img.crop(x, y, tile_size, tile_size)
            array = np.ndarray(
                buffer=tile.write_to_memory(), dtype=np.uint8, shape=(tile_size, tile_size, 3)
            )
            tensors.append(transform(Image.fromarray(array)))
        batch = torch.stack(tensors).to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            en, de = model(batch)
        amap = anomaly_maps(en, de, crop_size, sigma=sigma).float()
        small = torch.nn.functional.interpolate(
            amap, size=(out_tile, out_tile), mode="area"
        ).cpu().numpy()[:, 0]
        # Score the centre of each tile (border patches are unreliable and are
        # covered by the neighbouring overlapping tiles anyway).
        core = small[:, center : out_tile - center, center : out_tile - center]
        flat = torch.from_numpy(core.reshape(core.shape[0], -1))
        scores = topk_mean(flat, 0.01).numpy()

        for i, (x, y) in enumerate(batch_positions):
            px, py = round(x / downsample), round(y / downsample)
            pw = min(out_tile, canvas_w - px)
            ph = min(out_tile, canvas_h - py)
            if pw <= 0 or ph <= 0:
                continue
            heat[py : py + ph, px : px + pw] += small[i, :ph, :pw] * window[:ph, :pw]
            weight[py : py + ph, px : px + pw] += window[:ph, :pw]
            records.append({"x": int(x), "y": int(y), "score": float(scores[i])})

    weight[weight == 0] = 1.0
    heat /= weight
    image_score = float(np.max([r["score"] for r in records])) if records else float("nan")

    lo, hi = np.percentile(heat, [0.5, 99.5])
    norm = np.clip((heat - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((norm * 255).astype(np.uint8)).save(out_dir / "anomaly_map.png")
    np.save(out_dir / "anomaly_map.npy", heat)

    rgb = img.resize(1.0 / downsample, kernel="lanczos3")
    rgb_np = np.ndarray(buffer=rgb.write_to_memory(), dtype=np.uint8, shape=(rgb.height, rgb.width, 3))
    rgb_np = rgb_np[: norm.shape[0], : norm.shape[1]]
    if rgb_np.shape[:2] != norm.shape:
        norm = np.asarray(Image.fromarray((norm * 255).astype(np.uint8)).resize((rgb_np.shape[1], rgb_np.shape[0])))
        norm = norm.astype(np.float32) / 255.0
    overlay = (0.65 * rgb_np + 0.35 * _colormap(norm)).astype(np.uint8)
    Image.fromarray(overlay).save(out_dir / "overlay.jpg", quality=90)

    records.sort(key=lambda r: r["score"], reverse=True)
    prediction = {
        "image": str(image_path),
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "overlap": overlap,
        "downsample": downsample,
        "image_score": image_score,
        "threshold": threshold,
        "is_anomalous": bool(threshold is not None and image_score > threshold),
        "heatmap_min": float(lo),
        "heatmap_max": float(hi),
        "num_tiles": len(records),
        "roi": roi.to_dict() if roi is not None else None,
        "top_tiles": records[:20],
    }
    save_json(prediction, out_dir / "prediction.json")
    log(
        f"{image_path.name}: score={image_score:.4f}"
        + (f" threshold={threshold:.4f}" if threshold is not None else "")
        + f" -> {out_dir}"
    )
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None, help="Image-level score threshold")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.set)
    logger = setup_logging()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(config["model"]).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    data_cfg = config.get("data", {})
    tile_size = int(args.tile_size or data_cfg.get("tile_size", 1024))
    overlap = int(args.overlap if args.overlap is not None else data_cfg.get("tile_overlap", 128))
    crop_size = int(data_cfg.get("crop_size", 392))
    threshold = args.threshold
    if threshold is None:
        threshold = checkpoint.get("image_threshold")
    image = Path(args.image)
    out_dir = Path(args.out_dir) if args.out_dir else Path("predictions") / image.stem

    predict_image(
        model=model,
        image_path=image,
        out_dir=out_dir,
        device=device,
        tile_size=tile_size,
        overlap=overlap,
        crop_size=crop_size,
        downsample=args.downsample,
        threshold=threshold,
        amp=bool(config.get("train", {}).get("amp", True)),
        sigma=float(config.get("train", {}).get("gaussian_sigma", 4.0)),
        batch_size=args.batch_size,
        roi_settings=RoiSettings.from_config(data_cfg.get("roi")),
        logger=logger,
    )


if __name__ == "__main__":
    main()
