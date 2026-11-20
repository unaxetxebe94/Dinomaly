"""Minimal COCO reader tailored to polygon segmentation annotations."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int
    path: Path

    @property
    def exists(self) -> bool:
        return self.path.is_file()


@dataclass
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    segmentation: list[list[float]]
    area: float
    iscrowd: int = 0


@dataclass
class CocoDataset:
    root: Path
    images: list[CocoImage]
    annotations: dict[int, list[CocoAnnotation]]
    categories: dict[int, dict[str, Any]]
    info: dict[str, Any] = field(default_factory=dict)

    def image_by_id(self, image_id: int) -> CocoImage:
        for image in self.images:
            if image.id == image_id:
                return image
        raise KeyError(image_id)


def _polygons(segmentation: Any) -> list[list[float]] | None:
    """Return polygon ring coordinates, or None for non-polygon (e.g. RLE) data."""
    if isinstance(segmentation, list):
        if not segmentation:
            return None
        if isinstance(segmentation[0], (int, float)):
            # A single flat ring, not wrapped in a list.
            return [list(map(float, segmentation))]
        rings = []
        for ring in segmentation:
            if isinstance(ring, list) and len(ring) >= 6:
                rings.append(list(map(float, ring)))
        return rings or None
    return None


_IMAGE_EXTENSIONS = (".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp")
_NUMERIC_PREFIX = re.compile(r"^\d+_(.+)$")


def _disk_index(root: Path) -> dict[str, Path]:
    """Map lower-case stems (with and without a ``<number>_`` prefix) to files.

    The exported COCO files reference ``19359_top_native.jpg`` while the scans
    on disk are ``<seq>_19359_top_native.tiff``; both spellings resolve here.
    Exact stems win over prefix-stripped ones and TIFF over other formats.
    """
    exact: dict[str, Path] = {}
    stripped: dict[str, list[Path]] = defaultdict(list)
    rank = {ext: i for i, ext in enumerate(_IMAGE_EXTENSIONS)}
    try:
        entries = sorted(os.scandir(root), key=lambda e: e.name)
    except OSError:
        return {}
    for entry in entries:
        if not entry.is_file():
            continue
        stem, ext = os.path.splitext(entry.name)
        if ext.lower() not in rank:
            continue
        path = Path(entry.path)
        key = stem.lower()
        if key not in exact or rank[ext.lower()] < rank[exact[key].suffix.lower()]:
            exact[key] = path
        match = _NUMERIC_PREFIX.match(stem)
        if match:
            stripped[match.group(1).lower()].append(path)
    index = dict(exact)
    for key, paths in stripped.items():
        if key in index:
            continue
        paths.sort(key=lambda p: rank[p.suffix.lower()])
        best = [p for p in paths if p.suffix.lower() == paths[0].suffix.lower()]
        if len(best) == 1:  # ambiguous prefixes are left unresolved
            index[key] = best[0]
    return index


def resolve_image_path(root: Path, file_name: str, index: dict[str, Path] | None = None) -> Path:
    direct = root / file_name
    if direct.is_file():
        return direct
    if index is None:
        index = _disk_index(root)
    stem = Path(file_name.replace("\\", "/")).stem.lower()
    return index.get(stem, direct)


def load_coco(root: str | os.PathLike, annotation_file: str | os.PathLike) -> CocoDataset:
    root = Path(root)
    ann_path = Path(annotation_file)
    if not ann_path.is_absolute():
        ann_path = root / ann_path
    with open(ann_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    images: list[CocoImage] = []
    index: dict[str, Path] | None = None
    for entry in data.get("images", []):
        path = root / entry["file_name"]
        if not path.is_file():
            if index is None:
                index = _disk_index(root)
            path = resolve_image_path(root, entry["file_name"], index)
        images.append(
            CocoImage(
                id=int(entry["id"]),
                file_name=entry["file_name"],
                width=int(entry.get("width", 0)),
                height=int(entry.get("height", 0)),
                path=path,
            )
        )

    annotations: dict[int, list[CocoAnnotation]] = defaultdict(list)
    skipped = 0
    for entry in data.get("annotations", []):
        rings = _polygons(entry.get("segmentation"))
        if rings is None:
            skipped += 1
            continue
        bbox = entry.get("bbox") or [0, 0, 0, 0]
        annotations[int(entry["image_id"])].append(
            CocoAnnotation(
                id=int(entry["id"]),
                image_id=int(entry["image_id"]),
                category_id=int(entry["category_id"]),
                bbox=tuple(map(float, bbox)),  # type: ignore[arg-type]
                segmentation=rings,
                area=float(entry.get("area", 0.0)),
                iscrowd=int(entry.get("iscrowd", 0)),
            )
        )
    if skipped:
        print(f"[coco] skipped {skipped} non-polygon annotations in {ann_path}")

    categories = {int(c["id"]): c for c in data.get("categories", [])}
    return CocoDataset(
        root=root,
        images=images,
        annotations=dict(annotations),
        categories=categories,
        info=data.get("info", {}),
    )


def category_names(categories: dict[int, dict[str, Any]]) -> dict[int, str]:
    return {cid: str(c.get("name", f"class_{cid}")) for cid, c in categories.items()}


def annotation_bboxes_intersect_tile(
    annotation: CocoAnnotation, x: int, y: int, w: int, h: int, pad: float = 2.0
) -> bool:
    bx, by, bw, bh = annotation.bbox
    return not (bx + bw + pad < x or bx - pad > x + w or by + bh + pad < y or by - pad > y + h)
