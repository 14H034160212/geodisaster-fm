"""AlphaEarth annual satellite embeddings (Google).

Image collection: ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL``
- 64 dims per pixel, 10 m resolution, one image per year.
- Released alongside the AlphaEarth Foundations paper (Brown et al. 2025).

The downloader picks the year matching an event's pre-window by default; pass
``cfg.year`` to force a specific year, or ``year_from_event="post"`` to pull a
post-event slice (useful for cross-temporal experiments).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from omegaconf import DictConfig

from ..catalog import DisasterEvent
from ...utils.logging import get_logger
from .base import ExportSpec, event_aoi, export_image_region, init_ee

log = get_logger("gee.alphaearth")


def _pick_year(event: DisasterEvent, cfg: DictConfig) -> int:
    if cfg.get("year") is not None:
        return int(cfg.year)
    mode = cfg.get("year_from_event", "pre")
    win = event.pre_window if mode == "pre" else event.post_window
    if win is not None:
        return win[0].year
    if event.event_date is not None:
        return event.event_date.year
    return date.today().year - 1


def download(event: DisasterEvent, cfg: DictConfig, out_dir: str | Path) -> Path:
    init_ee(project=cfg.get("project"))
    import ee

    year = _pick_year(event, cfg)
    coll = ee.ImageCollection(cfg.get("collection", "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"))
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    aoi = event_aoi(event)

    image = (
        coll.filterDate(start, end)
            .filterBounds(aoi)
            .mosaic()
            .clip(aoi)
    )
    bands = cfg.get("bands")
    if bands:
        image = image.select(list(bands))

    out_path = Path(out_dir) / f"{event.event_id}_alphaearth_{year}.tif"
    spec = ExportSpec(
        image_id=f"alphaearth_{event.event_id}_{year}",
        image=image,
        region=aoi,
        scale_m=10.0,
        crs="EPSG:4326",
        dtype="float32",
        bands=list(bands) if bands else None,
        # AlphaEarth Satellite Embedding annual is 64-d. Hint so the auto-chunker
        # sizes tiles correctly even when bands=None (download all).
        n_bands_hint=len(bands) if bands else 64,
    )
    return export_image_region(spec, out_path, bbox_4326=event.bbox)
