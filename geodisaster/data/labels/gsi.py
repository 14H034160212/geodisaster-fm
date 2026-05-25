"""GSI (Geospatial Information Authority of Japan) label ingestion.

GSI publishes 浸水範囲 (flood inundation extent) and 崩壊地分布 (landslide
distribution) polygons after each major event. Source files vary in format:
GeoJSON, Shapefile, GML, or KML — we normalize to GeoPackage with a stable
schema before rasterization.

Inputs are expected to live under ``data/external/gsi/{event_id}/`` after a
manual one-time download (GSI doesn't expose an open programmatic API).
"""
from __future__ import annotations

from pathlib import Path

from ..catalog import DisasterEvent
from ...utils.logging import get_logger
from .rasterize import rasterize_label_file

log = get_logger("labels.gsi")


def _find_polygon_file(event_dir: Path) -> Path:
    candidates = []
    for pat in ("*.geojson", "*.gpkg", "*.shp", "*.kml", "*.gml"):
        candidates.extend(event_dir.glob(pat))
    if not candidates:
        raise FileNotFoundError(
            f"No polygon label files under {event_dir}. "
            "Download GSI inundation/landslide polygons there first."
        )
    return sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[0]


def ingest_gsi_flood(event: DisasterEvent, raw_root: str | Path, out_root: str | Path) -> Path:
    raw_dir = Path(raw_root) / event.event_id
    src = _find_polygon_file(raw_dir)
    out = Path(out_root) / event.event_id / f"{event.event_id}_label_flood.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("gsi_flood_ingest", event=event.event_id, src=str(src), out=str(out))
    return rasterize_label_file(src, event, out, resolution_m=10.0)


def ingest_gsi_landslide(event: DisasterEvent, raw_root: str | Path, out_root: str | Path) -> Path:
    raw_dir = Path(raw_root) / event.event_id
    src = _find_polygon_file(raw_dir)
    out = Path(out_root) / event.event_id / f"{event.event_id}_label_landslide.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("gsi_landslide_ingest", event=event.event_id, src=str(src), out=str(out))
    return rasterize_label_file(src, event, out, resolution_m=10.0)
