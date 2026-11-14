"""Detect the scanned object (region of interest) inside a large image.

The scans have an almost black background: the object sits somewhere in the
middle of a huge canvas.  Tiling the whole canvas wastes most of the work on
empty background, so before tiling we:

1. build a small thumbnail of the image,
2. binarise it on brightness (``threshold``, 0-255),
3. label the connected components of the bright mask,
4. keep the largest one ("the widest zone") and return its bounding box.

Only tiles that intersect that bounding box are produced; everything else in
the scan is ignored.  If detection fails the caller falls back to the whole
image, so a weird file can never lose its tiles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvips
from scipy import ndimage

SELECT_MODES = ("area", "width", "height")


@dataclass
class RoiSettings:
    """Configuration of the ROI detector (``data.roi`` in the YAML)."""

    enabled: bool = False
    threshold: float = 10.0
    select: str = "area"
    margin: int = 0
    preview: int = 1024

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> RoiSettings:
        cfg = dict(cfg or {})
        settings = cls(
            enabled=bool(cfg.get("enabled", False)),
            threshold=float(cfg.get("threshold", 10.0)),
            select=str(cfg.get("select", "area")),
            margin=int(cfg.get("margin", 0)),
            preview=int(cfg.get("preview", 1024)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.select not in SELECT_MODES:
            raise ValueError(f"data.roi.select must be one of {SELECT_MODES}, got {self.select!r}")
        if self.preview < 64:
            raise ValueError("data.roi.preview must be >= 64")
        if self.margin < 0:
            raise ValueError("data.roi.margin must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Roi:
    """Bounding box of the largest bright connected component, in image pixels."""

    x0: int
    y0: int
    x1: int  # exclusive
    y1: int  # exclusive
    image_width: int
    image_height: int
    component_pixels: int
    component_frac: float  # component pixels / thumbnail pixels
    num_components: int
    select: str

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def coverage(self) -> float:
        """Fraction of the image covered by the bounding box."""
        return (self.width * self.height) / float(self.image_width * self.image_height)

    @property
    def is_full_image(self) -> bool:
        return self.x0 == 0 and self.y0 == 0 and self.x1 == self.image_width and self.y1 == self.image_height

    def to_dict(self) -> dict[str, Any]:
        return {
            "x0": int(self.x0),
            "y0": int(self.y0),
            "x1": int(self.x1),
            "y1": int(self.y1),
            "width": int(self.width),
            "height": int(self.height),
            "coverage": round(float(self.coverage), 6),
            "component_pixels": int(self.component_pixels),
            "component_frac": round(float(self.component_frac), 6),
            "num_components": int(self.num_components),
            "select": self.select,
        }


def _to_rgb(img: pyvips.Image) -> pyvips.Image:
    if img.bands == 1:
        img = img.bandjoin([img, img, img])
    elif img.bands > 3:
        img = img[0:3]
    if img.interpretation != "srgb":
        img = img.colourspace("srgb")
    return img


def load_thumbnail_rgb(path: str | Path, preview: int) -> tuple[np.ndarray, int, int]:
    """Return a down-scaled 8-bit RGB array of the image plus its full size."""
    full = pyvips.Image.new_from_file(str(path), access="sequential")
    width, height = int(full.width), int(full.height)
    thumb = pyvips.Image.thumbnail(str(path), preview)
    rgb = _to_rgb(thumb).copy_memory()
    array = np.ndarray(
        buffer=rgb.write_to_memory(), dtype=np.uint8, shape=(rgb.height, rgb.width, 3)
    ).copy()
    return array, width, height


def brightness_thumbnail(path: str | Path, preview: int) -> tuple[np.ndarray, int, int]:
    """Return a down-scaled 8-bit grayscale array of the image plus its full size.

    Uses libvips' own ``b-w`` conversion so the threshold matches what other
    tools show for the same image.
    """
    full = pyvips.Image.new_from_file(str(path), access="sequential")
    width, height = int(full.width), int(full.height)
    thumb = pyvips.Image.thumbnail(str(path), preview)
    rgb = _to_rgb(thumb).copy_memory()
    gray = rgb.colourspace("b-w").numpy()
    return np.asarray(gray), width, height


def compute_roi(path: str | Path, settings: RoiSettings) -> Roi | None:
    """Find the bounding box of the largest bright connected component.

    Returns ``None`` when the image has no pixel above ``settings.threshold``
    (the caller then tiles the whole image).
    """
    settings.validate()
    gray, width, height = brightness_thumbnail(path, settings.preview)
    mask = gray > settings.threshold

    labels, num = ndimage.label(mask)
    if num == 0:
        return None

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    slices = ndimage.find_objects(labels)

    if settings.select == "width":
        scores = np.array([sl[1].stop - sl[1].start for sl in slices], dtype=np.float64)
    elif settings.select == "height":
        scores = np.array([sl[0].stop - sl[0].start for sl in slices], dtype=np.float64)
    else:
        scores = counts[1:].astype(np.float64)
    best = int(np.argmax(scores))  # 0-based index into slices == label - 1
    ys, xs = slices[best]

    thumb_h, thumb_w = gray.shape[:2]
    sx = width / float(thumb_w)
    sy = height / float(thumb_h)
    margin = int(settings.margin)
    x0 = max(0, int(np.floor(xs.start * sx)) - margin)
    y0 = max(0, int(np.floor(ys.start * sy)) - margin)
    x1 = min(width, int(np.ceil(xs.stop * sx)) + margin)
    y1 = min(height, int(np.ceil(ys.stop * sy)) + margin)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None

    return Roi(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        image_width=width,
        image_height=height,
        component_pixels=int(counts[best + 1]),
        component_frac=float(counts[best + 1]) / float(gray.size),
        num_components=int(num),
        select=settings.select,
    )
