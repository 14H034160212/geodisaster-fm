"""Patch tiling across multi-source rasters.

For each event we want a directory of aligned ``(image, mask)`` patches at a
fixed pixel size + stride, ready for PyTorch ``Dataset`` consumption. Sources
(AlphaEarth, Sentinel-1, Sentinel-2, DEM, etc.) come at different resolutions
so we reproject/resample everything to the AlphaEarth 10 m grid on the fly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .catalog import DisasterEvent
from ..utils.logging import get_logger

log = get_logger("data.tile")


@dataclass
class PatchIndex:
    event_id: str
    patch_id: str
    row: int
    col: int
    size: int
    sources: dict[str, str]   # source name -> tif path
    label_path: str | None
    pos_fraction: float = 0.0  # positive-class pixel fraction (mask QA)


def _open_aligned(path, ref_profile, target_shape, resampling="nearest"):
    """Read+reproject a raster onto a reference grid."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.io import MemoryFile

    method = getattr(Resampling, resampling)
    with rasterio.open(path) as src:
        n_bands = src.count
        dst = np.zeros((n_bands, target_shape[0], target_shape[1]), dtype=np.float32)
        reproject(
            source=rasterio.band(src, list(range(1, n_bands + 1))),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=method,
        )
    return dst


def tile_event(
    event: DisasterEvent,
    sources: dict[str, str | Path],   # name -> tif path
    label_path: str | Path | None,
    out_dir: str | Path,
    size: int = 256,
    stride: int = 224,
    min_pos_fraction: float = 0.0,
    require_label: bool = True,
) -> list[PatchIndex]:
    """Slice all source rasters into co-registered patches.

    The first source listed acts as the reference grid (typically AlphaEarth).
    Other sources are reprojected onto it.
    """
    import rasterio

    sources = {k: str(v) for k, v in sources.items()}
    if not sources:
        raise ValueError("at least one source required")
    ref_name, ref_path = next(iter(sources.items()))
    out_dir = Path(out_dir) / event.event_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ref_path) as ref:
        ref_profile = ref.profile.copy()
        H, W = ref.height, ref.width

    label_arr = None
    if label_path is not None:
        label_arr = _open_aligned(label_path, ref_profile, (H, W), resampling="nearest")[0]

    indices: list[PatchIndex] = []
    for r in range(0, max(1, H - size + 1), stride):
        for c in range(0, max(1, W - size + 1), stride):
            slc_r = slice(r, r + size)
            slc_c = slice(c, c + size)
            patch_id = f"{event.event_id}_r{r:06d}_c{c:06d}"

            patch_label = None
            pos_frac = 0.0
            if label_arr is not None:
                patch_label = label_arr[slc_r, slc_c]
                valid = patch_label != 255
                if valid.sum() == 0:
                    continue
                pos_frac = float((patch_label == 1).sum()) / float(valid.sum() + 1e-6)
                if pos_frac < min_pos_fraction and not (require_label is False):
                    pass  # keep negatives too but flag — train datamodule can filter

            patch_sources: dict[str, str] = {}
            for name, path in sources.items():
                arr = _open_aligned(path, ref_profile, (H, W),
                                    resampling="bilinear" if name != "label" else "nearest")
                patch = arr[:, slc_r, slc_c]
                patch_path = out_dir / f"{patch_id}__{name}.npy"
                np.save(patch_path, patch.astype(np.float32))
                patch_sources[name] = str(patch_path)

            label_out: str | None = None
            if patch_label is not None:
                label_out_path = out_dir / f"{patch_id}__label.npy"
                np.save(label_out_path, patch_label.astype(np.uint8))
                label_out = str(label_out_path)

            indices.append(PatchIndex(
                event_id=event.event_id,
                patch_id=patch_id,
                row=r, col=c, size=size,
                sources=patch_sources,
                label_path=label_out,
                pos_fraction=pos_frac,
            ))

    manifest = {
        "event_id": event.event_id,
        "patch_size": size,
        "stride": stride,
        "n_patches": len(indices),
        "ref_source": ref_name,
        "patches": [
            {
                "patch_id": p.patch_id,
                "row": p.row, "col": p.col, "size": p.size,
                "sources": p.sources, "label_path": p.label_path,
                "pos_fraction": p.pos_fraction,
            }
            for p in indices
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("event_tiled", event=event.event_id, patches=len(indices),
             ref=ref_name, out=str(out_dir))
    return indices


def merge_manifests(event_dirs: Iterable[Path]) -> list[dict]:
    """Collect per-event manifests into a single flat list of patch entries."""
    all_patches: list[dict] = []
    for d in event_dirs:
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        for p in m["patches"]:
            p = dict(p)
            p["event_id"] = m["event_id"]
            all_patches.append(p)
    return all_patches
