"""Earth Engine downloaders.

All modules in this subpackage expose a ``download(event, cfg, out_dir)``
callable. They share auth + region-export helpers from ``base.py``.
"""
from .base import (
    init_ee,
    event_aoi,
    export_image_region,
    EEInitError,
)

__all__ = ["init_ee", "event_aoi", "export_image_region", "EEInitError"]
