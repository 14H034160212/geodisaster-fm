"""Earth Engine bootstrap + chunked region-export helpers.

`init_ee()` is idempotent and caches the auth state. `export_image_region()`
fetches a server-side `ee.Image` over an AOI at a target resolution and writes
it locally as a Cloud-Optimized GeoTIFF.

For AOIs larger than the GEE `getDownloadURL` size budget (~50 MB / 32k pixels
per side), we automatically slice the AOI into N×M sub-tiles, download each,
and merge them via ``rasterio.merge.merge``. Chunk size adapts to band count
and dtype so AlphaEarth (64 bands × float32) uses smaller chunks than
Sentinel-1 (2 bands × float32).
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..catalog import DisasterEvent
from ...utils.logging import get_logger

log = get_logger("gee")
_EE_INITIALIZED = False

# GEE getDownloadURL per-request limits (Jan 2026): ~50 MB, 32k px per side.
# We leave generous headroom so retries don't burn the user's quota.
GEE_MAX_BYTES_PER_REQUEST = 45_000_000
GEE_MAX_PIXELS_PER_SIDE = 16_000   # well under the hard 32k limit


class EEInitError(RuntimeError):
    pass


def init_ee(project: str | None = None, high_volume: bool = True) -> None:
    """Initialize the Earth Engine Python API (cached)."""
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        return
    try:
        import ee  # noqa: F401
    except ImportError as e:
        raise EEInitError(
            "earthengine-api not installed. Run `pip install earthengine-api`."
        ) from e

    import ee
    kwargs: dict[str, Any] = {}
    if project:
        kwargs["project"] = project
    if high_volume:
        kwargs["opt_url"] = "https://earthengine-highvolume.googleapis.com"
    sa_key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    try:
        if sa_key and Path(sa_key).exists():
            creds = ee.ServiceAccountCredentials(email=None, key_file=sa_key)
            ee.Initialize(credentials=creds, **kwargs)
        else:
            ee.Initialize(**kwargs)
    except Exception as e:
        raise EEInitError(
            "Earth Engine initialization failed. Run `earthengine authenticate` "
            "or set GOOGLE_APPLICATION_CREDENTIALS to a service-account key."
        ) from e
    _EE_INITIALIZED = True
    log.info("ee_initialized", project=project, high_volume=high_volume)


def event_aoi(event: DisasterEvent):
    import ee
    if event.bbox is None:
        raise ValueError(f"event {event.event_id} has no bbox")
    minx, miny, maxx, maxy = event.bbox
    return ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:4326", geodesic=False)


@dataclass
class ExportSpec:
    image_id: str
    image: Any
    region: Any
    scale_m: float = 10.0
    crs: str = "EPSG:4326"
    dtype: str = "float32"
    bands: list[str] | None = None
    n_bands_hint: int | None = None  # used to size chunks before we resolve the band list


def _utm_crs_for_bbox(bbox: tuple[float, float, float, float]) -> str:
    cx = 0.5 * (bbox[0] + bbox[2])
    cy = 0.5 * (bbox[1] + bbox[3])
    zone = int((cx + 180) // 6) + 1
    epsg = 32600 + zone if cy >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def _max_chunk_side_px(n_bands: int, dtype: str,
                       max_bytes: int = GEE_MAX_BYTES_PER_REQUEST,
                       hard_cap_px: int = GEE_MAX_PIXELS_PER_SIDE) -> int:
    """Pixel-side that fits ``max_bytes`` for a square chunk of given shape."""
    bpp = max(1, n_bands) * np.dtype(dtype).itemsize
    side = int((max_bytes / bpp) ** 0.5)
    return max(64, min(side, hard_cap_px))


def _chunk_bbox(
    bbox: tuple[float, float, float, float],
    chunk_side_px: int,
    scale_m: float,
    crs: str,
) -> list[tuple[float, float, float, float]]:
    """Split bbox into sub-bboxes whose side <= chunk_side_px at scale_m.

    bbox is in EPSG:4326; we compute chunk extents in approximate meters
    (longitude scaled by cos(lat)).
    """
    import math
    minx, miny, maxx, maxy = bbox
    cy = 0.5 * (miny + maxy)
    deg_per_m_lat = 1.0 / 111_320.0
    deg_per_m_lon = 1.0 / (111_320.0 * max(math.cos(math.radians(cy)), 1e-6))
    chunk_m = chunk_side_px * scale_m
    dy = chunk_m * deg_per_m_lat
    dx = chunk_m * deg_per_m_lon

    chunks: list[tuple[float, float, float, float]] = []
    y = miny
    while y < maxy:
        y2 = min(maxy, y + dy)
        x = minx
        while x < maxx:
            x2 = min(maxx, x + dx)
            chunks.append((x, y, x2, y2))
            x = x2
        y = y2
    return chunks


def _download_chunk(image, region, scale_m, crs, dtype, bands, out_path):
    import geemap
    if bands is not None:
        image = image.select(list(bands))
    geemap.download_ee_image(
        image=image,
        filename=str(out_path),
        region=region,
        scale=scale_m,
        crs=crs,
        dtype=dtype,
        num_threads=8,
    )


def _merge_chunks(chunk_paths: list[Path], out_path: Path, compress: str = "deflate") -> Path:
    import rasterio
    from rasterio.merge import merge

    srcs = [rasterio.open(p) for p in chunk_paths]
    try:
        mosaic, transform = merge(srcs)
        profile = srcs[0].profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            compress=compress,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for s in srcs:
            s.close()
    return out_path


def export_image_region(
    spec: ExportSpec,
    out_path: str | Path,
    mode: str = "local",
    bbox_4326: tuple[float, float, float, float] | None = None,
    chunk: bool | None = None,
    keep_chunks: bool = False,
) -> Path:
    """Export an ee.Image clipped to a region to a local GeoTIFF.

    If ``chunk`` is None (the default), we estimate the request size and
    automatically chunk when it exceeds ``GEE_MAX_BYTES_PER_REQUEST`` or the
    per-side pixel limit. ``chunk=False`` forces a single request, ``chunk=True``
    forces chunking even for small regions (useful when GEE has been flaky).
    """
    init_ee()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = spec.image
    if spec.bands is not None:
        image = image.select(list(spec.bands))

    crs = spec.crs
    if bbox_4326 is not None and crs.upper() == "EPSG:4326":
        crs = _utm_crs_for_bbox(bbox_4326)

    if mode == "drive":
        import ee
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=spec.image_id,
            folder="geodisaster",
            fileNamePrefix=spec.image_id,
            region=spec.region,
            scale=spec.scale_m,
            crs=crs,
            maxPixels=int(1e13),
        )
        task.start()
        log.info("ee_export_drive_started", image_id=spec.image_id, task_id=task.id)
        return out_path

    if mode != "local":
        raise ValueError(f"unknown export mode: {mode}")

    # ---- decide whether to chunk ----
    n_bands = spec.n_bands_hint or (len(spec.bands) if spec.bands else 1)
    chunk_side = _max_chunk_side_px(n_bands, spec.dtype)
    needs_chunk = bool(chunk) if chunk is not None else False
    if bbox_4326 is not None and chunk is None:
        import math
        cy = 0.5 * (bbox_4326[1] + bbox_4326[3])
        deg_per_m_lat = 1.0 / 111_320.0
        deg_per_m_lon = 1.0 / (111_320.0 * max(math.cos(math.radians(cy)), 1e-6))
        approx_h = int(((bbox_4326[3] - bbox_4326[1]) / (spec.scale_m * deg_per_m_lat)))
        approx_w = int(((bbox_4326[2] - bbox_4326[0]) / (spec.scale_m * deg_per_m_lon)))
        needs_chunk = (approx_h > chunk_side) or (approx_w > chunk_side)
        log.info("ee_size_check", image_id=spec.image_id,
                 approx_px=(approx_h, approx_w), chunk_side=chunk_side,
                 needs_chunk=needs_chunk)

    if not needs_chunk:
        log.info("ee_download_local", image_id=spec.image_id, scale_m=spec.scale_m, crs=crs)
        _download_chunk(image, spec.region, spec.scale_m, crs, spec.dtype, spec.bands, out_path)
        return out_path

    # ---- chunked path ----
    if bbox_4326 is None:
        raise ValueError("chunked export requires bbox_4326 to subdivide the AOI")
    import ee
    chunks = _chunk_bbox(bbox_4326, chunk_side, spec.scale_m, crs)
    log.info("ee_chunked_export", image_id=spec.image_id, n_chunks=len(chunks),
             chunk_side_px=chunk_side)
    tmp_dir = Path(tempfile.mkdtemp(prefix="gee_chunks_", dir=out_path.parent))
    chunk_paths: list[Path] = []
    try:
        for i, c in enumerate(chunks):
            sub_region = ee.Geometry.Rectangle(list(c), proj="EPSG:4326", geodesic=False)
            chunk_path = tmp_dir / f"chunk_{i:04d}.tif"
            try:
                _download_chunk(image.clip(sub_region), sub_region,
                                spec.scale_m, crs, spec.dtype, spec.bands, chunk_path)
                chunk_paths.append(chunk_path)
            except Exception as e:
                log.warning("ee_chunk_failed", chunk=i, bbox=c, err=str(e))
        if not chunk_paths:
            raise RuntimeError(f"all chunks failed for {spec.image_id}")
        _merge_chunks(chunk_paths, out_path)
        log.info("ee_chunks_merged", image_id=spec.image_id,
                 chunks_ok=len(chunk_paths), total=len(chunks))
    finally:
        if not keep_chunks:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_path
