"""Geospatial helpers: CRS, bounding boxes, pixel <-> geo transforms."""
from __future__ import annotations

from typing import Any

from pyproj import Transformer


def bbox_to_geometry(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    """Convert (minx, miny, maxx, maxy) to a GeoJSON Polygon."""
    minx, miny, maxx, maxy = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
                [minx, miny],
            ]
        ],
    }


def reproject_bbox(
    bbox: tuple[float, float, float, float],
    src_crs: str,
    dst_crs: str,
) -> tuple[float, float, float, float]:
    if src_crs == dst_crs:
        return bbox
    tr = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    minx, miny = tr.transform(bbox[0], bbox[1])
    maxx, maxy = tr.transform(bbox[2], bbox[3])
    return (minx, miny, maxx, maxy)


def raster_resolution_to_crs_units(resolution_m: float, crs: str) -> float:
    """Approximate target resolution in CRS units. Geographic CRS gets degrees."""
    if crs.lower() in {"epsg:4326", "wgs84"}:
        return resolution_m / 111_320.0
    return resolution_m


def pixel_to_geo(
    transform: Any, row: int | float, col: int | float
) -> tuple[float, float]:
    x, y = transform * (col, row)
    return x, y


def geo_to_pixel(
    transform: Any, x: float, y: float
) -> tuple[int, int]:
    inv = ~transform
    col, row = inv * (x, y)
    return int(round(row)), int(round(col))
