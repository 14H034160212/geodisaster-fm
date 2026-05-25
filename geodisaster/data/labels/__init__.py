"""Label ingestion: convert official polygons / global benchmark labels into
per-event rasterized masks aligned to the AlphaEarth 10 m grid.
"""
from .rasterize import polygons_to_mask, rasterize_label_file
from .gsi import ingest_gsi_flood, ingest_gsi_landslide
from .global_datasets import (
    ingest_sen1floods11,
    ingest_xbd,
    ingest_openearthmap,
    DatasetManifest,
)

__all__ = [
    "polygons_to_mask",
    "rasterize_label_file",
    "ingest_gsi_flood",
    "ingest_gsi_landslide",
    "ingest_sen1floods11",
    "ingest_xbd",
    "ingest_openearthmap",
    "DatasetManifest",
]
