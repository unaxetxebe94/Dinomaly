"""Tile datasets and deterministic evaluation subset sampling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    crop_size: int,
    train: bool,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    min_scale: float = 0.6,
    rotation: bool = False,
) -> transforms.Compose:
    if train:
        ops: list[Any] = [
            transforms.RandomResizedCrop(
                crop_size,
                scale=(min_scale, 1.0),
                ratio=(0.85, 1.18),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        ]
        if rotation:
            ops.append(transforms.RandomApply([transforms.RandomRotation(90)], p=0.5))
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    else:
        ops = [
            transforms.Resize((crop_size, crop_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    return transforms.Compose(ops)


def build_mask_transform(crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((crop_size, crop_size), interpolation=InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ]
    )


class TileDataset(Dataset):
    """Dataset over rows of the parquet manifest produced by ``build_tiles``."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tiles_dir: str | os.PathLike,
        crop_size: int = 392,
        train: bool = False,
        return_mask: bool = False,
        min_scale: float = 0.6,
        rotation: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tiles_dir = Path(tiles_dir)
        self.train = train
        self.return_mask = return_mask or not train
        self.image_transform = build_transforms(crop_size, train=train, min_scale=min_scale, rotation=rotation)
        self.mask_transform = build_mask_transform(crop_size)
        # Materialise path lists to keep per-item overhead low.
        self.tile_paths = [str(self.tiles_dir / p) for p in self.frame["tile_path"].tolist()]
        self.mask_paths = [
            str(self.tiles_dir / p) if p else "" for p in self.frame["mask_path"].fillna("").tolist()
        ]
        self.labels = self.frame["has_defect"].astype(bool).tolist()
        self.tile_ids = self.frame["tile_id"].tolist()
        self.image_uids = self.frame["image_uid"].tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = Image.open(self.tile_paths[index]).convert("RGB")
        tensor = self.image_transform(image)
        item: dict[str, Any] = {
            "image": tensor,
            "has_defect": self.labels[index],
            "tile_id": self.tile_ids[index],
            "image_uid": self.image_uids[index],
        }
        if self.return_mask:
            if self.mask_paths[index]:
                mask = Image.open(self.mask_paths[index])
                mask_tensor = self.mask_transform(mask).squeeze(0).to(torch.uint8)
            else:
                mask_tensor = torch.zeros(
                    (tensor.shape[-2], tensor.shape[-1]), dtype=torch.uint8
                )
            item["mask"] = mask_tensor
        return item


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 8,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def release_loader_workers(*loaders: DataLoader | None) -> None:
    """Stop the persistent worker processes of finished loaders.

    On Windows every worker is a full Python+torch process; leaving the train/val
    workers alive while the test loader spawns its own can exhaust the commit limit
    (WinError 1455 "paging file too small").
    """
    for loader in loaders:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        if loader is not None:
            loader._iterator = None


def sample_eval_tiles(
    frame: pd.DataFrame,
    max_tiles: int | None,
    seed: int = 42,
    include_all_defects: bool = True,
) -> pd.DataFrame:
    """Keep every defect tile and a random sample of normal tiles.

    Pixel metrics are dominated by normal pixels, so a few thousand normal
    tiles are plenty; retaining *all* defect tiles keeps rare defects visible.
    """
    frame = frame.reset_index(drop=True)
    if max_tiles is None or max_tiles <= 0 or len(frame) <= max_tiles:
        return frame
    rng = np.random.default_rng(seed)
    defect = frame[frame["has_defect"]]
    normal = frame[~frame["has_defect"]]
    if include_all_defects:
        n_normal = max(0, max_tiles - len(defect))
        if len(normal) > n_normal:
            idx = rng.choice(len(normal), size=n_normal, replace=False)
            normal = normal.iloc[np.sort(idx)]
        out = pd.concat([defect, normal], ignore_index=True)
    else:
        idx = rng.choice(len(frame), size=max_tiles, replace=False)
        out = frame.iloc[np.sort(idx)]
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_frames(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: manifest[manifest["split"] == name].reset_index(drop=True) for name in ("train", "val", "test")}
