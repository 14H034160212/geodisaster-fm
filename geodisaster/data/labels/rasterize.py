"""Generic polygon -> aligned raster mask utilities."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..catalog import DisasterEvent
from ...utils.logging import get_logger

log = get_logger("labels.rasterize")


def _event_transform(event: DisasterEvent, resolution_m: float = 10.0):
    """Build an affine + shape for an event's AOI at the requested resolution.

    Output grid is EPSG:4326 with degree-per-pixel approximated at the centroid
    latitude. For higher-resolution work, swap to UTM.
    """
    from rasterio.transform import from_bounds

    if event.bbox is None:
        raise ValueError(f"event {event.event_id} has no bbox")
    minx, miny, maxx, maxy = event.bbox
    cy = 0.5 * (miny + maxy)
    deg_per_m_lat = 1.0 / 111_320.0
    deg_per_m_lon = 1.0 / (111_320.0 * max(math.cos(math.radians(cy)), 1e-6))
    px_w = max(1, int(round((maxx - minx) / (resolution_m * deg_per_m_lon))))
    px_h = max(1, int(round((maxy - miny) / (resolution_m * deg_per_m_lat))))
    transform = from_bounds(minx, miny, maxx, maxy, px_w, px_h)
    return transform, (px_h, px_w)


def polygons_to_mask(
    gdf, event: DisasterEvent, resolution_m: float = 10.0,
    value_field: str | None = None, default_value: int = 1,
) -> tuple[np.ndarray, dict]:
    from rasterio.features import rasterize

    transform, shape = _event_transform(event, resolution_m)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")

    if value_field and value_field in gdf.columns:
        shapes = [(geom, int(val)) for geom, val in zip(gdf.geometry, gdf[value_field]) if geom is not None]
    else:
        shapes = [(geom, default_value) for geom in gdf.geometry if geom is not None]

    if not shapes:
        log.warning("rasterize_empty", event=event.event_id)
        mask = np.zeros(shape, dtype=np.uint8)
    else:
        mask = rasterize(
            shapes=shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )

    profile = {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": 255,
    }
    return mask, profile


def rasterize_label_file(
    label_path: str | Path,
    event: DisasterEvent,
    out_path: str | Path,
    resolution_m: float = 10.0,
    value_field: str | None = None,
) -> Path:
    import geopandas as gpd
    from ...utils.io import write_raster

    gdf = gpd.read_file(label_path)
    mask, profile = polygons_to_mask(
        gdf, event, resolution_m=resolution_m, value_field=value_field
    )
    out_path = Path(out_path)
    write_raster(mask, profile, out_path)
    log.info(
        "label_rasterized",
        event=event.event_id, label=str(label_path),
        positives=int((mask == 1).sum()), out=str(out_path),
    )
    return out_path
