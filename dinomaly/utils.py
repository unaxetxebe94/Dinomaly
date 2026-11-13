"""Dependency-free helpers shared by the scripts."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config {path} did not contain a YAML mapping")
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``--set a.b.c=value`` overrides in place."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must look like key.subkey=value")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def setup_logging(log_file: str | os.PathLike | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("dinomaly")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Dinomaly is trained for a fixed number of iterations; deterministic cudnn
    # makes runs reproducible at a small speed cost.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _json_default(value: Any) -> Any:
    """Fallback for numpy/pandas scalars, paths and tensors."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else float("nan")


def topk_mean(values: torch.Tensor, ratio: float) -> torch.Tensor:
    """Mean of the highest ``ratio`` fraction of each row of a 2-D tensor."""
    if ratio >= 1.0:
        return values.mean(dim=1)
    k = max(1, int(values.shape[1] * ratio))
    return values.topk(k, dim=1).values.mean(dim=1)
