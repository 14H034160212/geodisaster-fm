from .io import load_config, save_config, ensure_dir, read_raster, write_raster
from .logging import get_logger, setup_logging

# geo helpers depend on pyproj. They're only needed for GEE downloaders and
# label processing, not for training/smoke. Tolerate a missing pyproj install.
try:
    from .geo import (
        bbox_to_geometry,
        reproject_bbox,
        raster_resolution_to_crs_units,
        pixel_to_geo,
        geo_to_pixel,
    )
except ImportError:  # pragma: no cover
    bbox_to_geometry = reproject_bbox = None  # type: ignore
    raster_resolution_to_crs_units = pixel_to_geo = geo_to_pixel = None  # type: ignore

__all__ = [
    "load_config", "save_config", "ensure_dir", "read_raster", "write_raster",
    "get_logger", "setup_logging",
    "bbox_to_geometry", "reproject_bbox", "raster_resolution_to_crs_units",
    "pixel_to_geo", "geo_to_pixel",
]
