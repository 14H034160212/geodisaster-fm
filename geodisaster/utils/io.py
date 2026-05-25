"""File I/O helpers: configs and rasters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = OmegaConf.create(raw)
    OmegaConf.resolve(cfg)
    return cfg


def save_config(cfg: DictConfig | dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(cfg, DictConfig):
        OmegaConf.save(cfg, path)
    else:
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_raster(
    path: str | Path,
    bands: list[int] | None = None,
    window: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a raster file. Returns (array, profile)."""
    import rasterio

    with rasterio.open(path) as src:
        if bands is None:
            arr = src.read(window=window)
        else:
            arr = src.read(bands, window=window)
        profile = src.profile.copy()
        if window is not None:
            profile["transform"] = src.window_transform(window)
            profile["height"] = arr.shape[-2]
            profile["width"] = arr.shape[-1]
    return arr, profile


def write_raster(
    arr: np.ndarray,
    profile: dict[str, Any],
    path: str | Path,
    compress: str = "deflate",
) -> Path:
    """Write a raster file as Cloud-Optimized GeoTIFF when possible."""
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        arr = arr[None, ...]
    profile = profile.copy()
    profile.update(
        count=arr.shape[0],
        height=arr.shape[1],
        width=arr.shape[2],
        dtype=arr.dtype,
        compress=compress,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
    return path
