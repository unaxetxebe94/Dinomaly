"""Turn large COCO-annotated images into tile images + pixel masks.

The big scans are read through libvips, so a 16k x 27k TIFF is never decoded
in full at once.  Every tile is saved as JPEG; masks are stored as 8-bit PNGs
where the pixel value is a compact class index (0 = background).  A parquet
manifest describes every tile and is the only thing the training code reads.

Tile grid notes
---------------
* Tiles are always exactly ``tile_size`` pixels (the last row/column is shifted
  back so it stays inside the image).  This avoids padded borders.
* ``tile_overlap`` < ``tile_size`` gives a stride of ``tile_size - overlap``;
  overlapping tiles are averaged when maps are stitched back together.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvips
from PIL import Image, ImageDraw
from tqdm import tqdm

from .coco import CocoDataset, CocoImage, annotation_bboxes_intersect_tile, load_coco
from .utils import save_json

# Keep libvips' decoded-image cache small when several workers run in parallel.
pyvips.cache_set_max(32)
pyvips.cache_set_max_mem(1024)
pyvips.cache_set_max_files(16)

JPEG_OPTIONS = {"Q": 92, "strip": True, "interlace": False}


@dataclass
class TileJob:
    source: str
    image: CocoImage
    split: str
    tile_size: int
    overlap: int
    jpeg_quality: int
    min_std: float
    category_map: dict[int, int]


def tile_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    positions = list(range(0, length - tile_size + 1, stride))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def assign_splits(
    images: list[CocoImage],
    annotations: dict[int, list[Any]],
    ratios: dict[str, float],
    seed: int,
) -> dict[int, str]:
    """Group split by source image, stratified by whether it contains defects."""
    rng = np.random.default_rng(seed)
    strata: dict[bool, list[int]] = {True: [], False: []}
    for image in images:
        strata[bool(annotations.get(image.id))].append(image.id)

    split_of: dict[int, str] = {}
    names = ["train", "val", "test"]
    fracs = np.array([float(ratios.get(n, 0.0)) for n in names], dtype=np.float64)
    if fracs.sum() <= 0:
        raise ValueError("Split ratios must sum to > 0")
    fracs /= fracs.sum()

    for ids in strata.values():
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        if n == 0:
            continue
        counts = np.floor(fracs * n).astype(int)
        # Give every non-empty stratum at least one image per split when possible.
        if n >= len(names):
            counts = np.maximum(counts, 1)
        while counts.sum() > n:
            counts[int(np.argmax(counts))] -= 1
        while counts.sum() < n:
            counts[int(np.argmin(counts))] += 1
        cursor = 0
        for name, count in zip(names, counts):
            for image_id in ids[cursor : cursor + int(count)]:
                split_of[image_id] = name
            cursor += int(count)
    return split_of


def _ring_bbox(ring: list[float]) -> tuple[float, float, float, float]:
    xs = ring[0::2]
    ys = ring[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def rasterize_tile_mask(
    annotations: Iterable[Any],
    x: int,
    y: int,
    width: int,
    height: int,
    category_map: dict[int, int],
) -> np.ndarray:
    """Rasterize polygon annotations into a uint8 mask for one tile."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    anns = [a for a in annotations if annotation_bboxes_intersect_tile(a, x, y, width, height)]
    # Draw large objects first so small, more specific defects stay visible.
    anns.sort(key=lambda a: a.area, reverse=True)
    for ann in anns:
        value = category_map.get(ann.category_id, 0)
        if value <= 0:
            continue
        for ring in ann.segmentation:
            bx0, by0, bx1, by1 = _ring_bbox(ring)
            if bx1 < x or bx0 > x + width or by1 < y or by0 > y + height:
                continue
            points = [(ring[i] - x, ring[i + 1] - y) for i in range(0, len(ring) - 1, 2)]
            if len(points) >= 3:
                draw.polygon(points, fill=int(value))
    return np.asarray(mask, dtype=np.uint8)


def _tile_image(img: pyvips.Image, x: int, y: int, size: int) -> pyvips.Image:
    return img.crop(x, y, size, size)


def process_image(
    job: TileJob, out_dir: Path, anns: list[Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create all tiles/masks for one source image.  Returns manifest rows + stats."""
    img_path = job.image.path
    stats = {"image": str(img_path), "tiles": 0, "skipped_blank": 0, "error": None}
    try:
        img = pyvips.Image.new_from_file(str(img_path), access="random")
    except Exception as exc:  # noqa: BLE001 - corrupt files should not kill the run
        stats["error"] = f"open failed: {exc}"
        return [], stats

    if img.bands == 1:
        img = img.bandjoin([img, img, img])
    elif img.bands > 3:
        img = img[0:3]
    if img.interpretation != "srgb":
        img = img.colourspace("srgb")

    width, height = int(img.width), int(img.height)
    size = job.tile_size
    overlap = job.overlap
    xs = tile_positions(width, size, overlap)
    ys = tile_positions(height, size, overlap)

    split_dir = out_dir / "tiles" / job.split
    mask_dir = out_dir / "masks" / job.split
    split_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for y in ys:
        for x in xs:
            tile = _tile_image(img, x, y, size)
            mask = rasterize_tile_mask(anns, x, y, size, size, job.category_map)
            defect_pixels = int(np.count_nonzero(mask))

            if defect_pixels == 0 and job.min_std > 0:
                try:
                    if float(tile.deviate()) < job.min_std:
                        stats["skipped_blank"] += 1
                        continue
                except pyvips.error.Error:
                    pass

            tile_id = f"{job.source}_{job.image.id:07d}_{y:06d}_{x:06d}"
            tile_rel = Path("tiles") / job.split / f"{tile_id}.jpg"
            tile_abs = out_dir / tile_rel
            options = dict(JPEG_OPTIONS)
            options["Q"] = int(job.jpeg_quality)
            # The external USB drive occasionally returns a transient read
            # error; one retry rescues the image instead of dropping it whole.
            for attempt in range(2):
                try:
                    tile.jpegsave(str(tile_abs), **options)
                    break
                except pyvips.error.Error:
                    if attempt == 1:
                        raise

            mask_rel: str = ""
            categories: list[int] = []
            if defect_pixels:
                values = np.unique(mask)
                categories = sorted(int(v) for v in values if v > 0)
                mask_rel = str(Path("masks") / job.split / f"{tile_id}.png")
                Image.fromarray(mask).save(out_dir / mask_rel, optimize=True)

            rows.append(
                {
                    "tile_id": tile_id,
                    "source": job.source,
                    "image_id": int(job.image.id),
                    "image_uid": f"{job.source}:{job.image.id}",
                    "file_name": job.image.file_name,
                    "split": job.split,
                    "x": int(x),
                    "y": int(y),
                    "w": int(size),
                    "h": int(size),
                    "image_width": width,
                    "image_height": height,
                    "tile_path": str(tile_rel),
                    "mask_path": mask_rel,
                    "has_defect": defect_pixels > 0,
                    "defect_pixels": defect_pixels,
                    "categories": categories,
                }
            )
            stats["tiles"] += 1
    return rows, stats


# ``process_image`` is executed in threads; annotations are passed explicitly
# because they are large and not part of the picklable job description.
def process_image_with_annotations(job: TileJob, out_dir: Path, annotations: list[Any]):
    return process_image(job, out_dir, annotations)


def build_category_map(datasets: list[CocoDataset]) -> tuple[dict[int, int], list[dict[str, Any]]]:
    """Map COCO category ids to compact 1..K indices used inside mask PNGs."""
    merged: dict[int, dict[str, Any]] = {}
    for dataset in datasets:
        for cid, meta in dataset.categories.items():
            merged.setdefault(cid, meta)
    entries = sorted(merged.items(), key=lambda kv: (str(kv[1].get("name", "")), kv[0]))
    category_map = {cid: idx + 1 for idx, (cid, _) in enumerate(entries)}
    classes = [
        {
            "mask_value": category_map[cid],
            "category_id": cid,
            "name": str(meta.get("name", f"class_{cid}")),
            "color": meta.get("color"),
        }
        for cid, meta in entries
    ]
    return category_map, classes


def build_tiles(
    config: dict[str, Any],
    tiles_dir: str | os.PathLike,
    workers: int = 4,
    limit_images: int | None = None,
    overwrite: bool = False,
    resume: bool = False,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    out_dir = Path(tiles_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.parquet"

    data_cfg = config["data"]
    ratios = data_cfg.get("split", {"train": 0.7, "val": 0.15, "test": 0.15})
    seed = int(config.get("project", {}).get("seed", 42))
    tile_size = int(data_cfg.get("tile_size", 1024))
    overlap = int(data_cfg.get("tile_overlap", 128))
    jpeg_quality = int(data_cfg.get("jpeg_quality", 92))
    min_std = float(data_cfg.get("min_std", 0.0))

    if overlap >= tile_size:
        raise ValueError("data.tile_overlap must be smaller than data.tile_size")

    datasets: list[tuple[str, CocoDataset]] = []
    for source in data_cfg["sources"]:
        name = str(source.get("name") or Path(source["root"]).name)
        root = source["root"]
        ann = source.get("annotations", "_annotations.coco.json")
        log(f"Loading annotations: {root} ({ann})")
        dataset = load_coco(root, ann)
        datasets.append((name, dataset))

    category_map, classes = build_category_map([d for _, d in datasets])
    save_json(classes, out_dir / "classes.json")

    jobs: list[TileJob] = []
    annotations_by_job: dict[str, list[Any]] = {}
    for source_name, dataset in datasets:
        existing = [im for im in dataset.images if im.exists]
        missing = len(dataset.images) - len(existing)
        if missing:
            log(f"[{source_name}] WARNING: {missing} annotated images are missing on disk")
        split_of = assign_splits(existing, dataset.annotations, ratios, seed)
        if limit_images is not None:
            existing = existing[: int(limit_images)]
        log(
            f"[{source_name}] {len(existing)} images -> "
            + ", ".join(
                f"{split}: {sum(split_of.get(im.id) == split for im in existing)}" for split in ("train", "val", "test")
            )
        )
        for image in existing:
            job = TileJob(
                source=source_name,
                image=image,
                split=split_of[image.id],
                tile_size=tile_size,
                overlap=overlap,
                jpeg_quality=jpeg_quality,
                min_std=min_std,
                category_map=category_map,
            )
            jobs.append(job)
            annotations_by_job[job_key(job)] = dataset.annotations.get(image.id, [])

    existing_df = None
    if resume and manifest_path.exists():
        existing_df = pd.read_parquet(manifest_path)
        done_uids = set(existing_df["image_uid"].astype(str))
        before = len(jobs)
        jobs = [job for job in jobs if job_key(job) not in done_uids]
        log(f"Resume: {before - len(jobs)} images already tiled, {len(jobs)} remaining")
        if not jobs:
            log("Nothing to do: every image already has tiles")
            return existing_df

    if not jobs:
        raise RuntimeError("No images to tile")

    log(f"Tiling {len(jobs)} images with tile_size={tile_size}, overlap={overlap}, workers={workers}")
    started = time.time()
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    # libvips/libtiff is not reliable when several threads decode huge TIFFs in
    # the same process (concurrent tiffload errors).  Worker *processes* each
    # own their libvips state and are stable; this also parallelises PNG mask
    # rasterisation, which is pure Python.
    def consume(results: Iterable[tuple[TileJob, list[dict[str, Any]], dict[str, Any]]]) -> None:
        for job, tile_rows, stats in results:
            if stats.get("error"):
                failed.append({"image": job.image.file_name, "error": stats["error"]})
            rows.extend(tile_rows)

    if workers <= 1:
        def serial() -> Iterable[tuple[TileJob, list[dict[str, Any]], dict[str, Any]]]:
            for job in tqdm(jobs, desc="tiling"):
                try:
                    tile_rows, stats = process_image_with_annotations(
                        job, out_dir, annotations_by_job[job_key(job)]
                    )
                except Exception as exc:  # noqa: BLE001
                    failed.append({"image": job.image.file_name, "error": repr(exc)})
                    continue
                yield job, tile_rows, stats

        consume(serial())
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(
                    process_image_with_annotations, job, out_dir, annotations_by_job[job_key(job)]
                ): job
                for job in jobs
            }
            produced = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="tiling"):
                job = futures[future]
                try:
                    tile_rows, stats = future.result()
                except Exception as exc:  # noqa: BLE001
                    failed.append({"image": job.image.file_name, "error": repr(exc)})
                    continue
                produced.append((job, tile_rows, stats))
            consume(produced)

    df = pd.DataFrame(rows)
    if df.empty and existing_df is None:
        raise RuntimeError("No tiles were produced")
    if existing_df is not None:
        df = pd.concat([existing_df, df], ignore_index=True) if not df.empty else existing_df
    df.to_parquet(manifest_path, index=False)

    elapsed = time.time() - started
    info = {
        "tiles_dir": str(out_dir.resolve()),
        "config": data_cfg,
        "num_images": int(df["image_uid"].nunique()),
        "num_images_attempted": len(jobs),
        "resumed": existing_df is not None,
        "num_tiles": len(df),
        "num_defect_tiles": int(df["has_defect"].sum()),
        "num_failed_images_this_run": len(failed),
        "failed_images": failed[:50],
        "elapsed_seconds": round(elapsed, 1),
        "tile_size": tile_size,
        "overlap": overlap,
        "classes": classes,
        "counts_by_split": {str(k): int(v) for k, v in df.groupby("split").size().items()},
        "defect_tiles_by_split": df.groupby("split")["has_defect"].sum().astype(int).to_dict(),
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    save_json(info, out_dir / "build_info.json")
    log(
        f"Done: {len(df)} tiles ({int(df['has_defect'].sum())} with defects) in {elapsed/60:.1f} min -> {manifest_path}"
    )
    if failed:
        log(f"WARNING: {len(failed)} images failed; see build_info.json")
    return df


def job_key(job: TileJob) -> str:
    return f"{job.source}:{job.image.id}"
