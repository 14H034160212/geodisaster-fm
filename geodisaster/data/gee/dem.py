"""DEM + terrain derivatives (slope, curvature, aspect, TWI, HAND).

Default source is SRTM 1-arcsec (USGS/SRTMGL1_003). For landslide work, prefer
ALOS World 3D (JAXA/ALOS/AW3D30/V3_2) by setting cfg.dem.source.
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from ..catalog import DisasterEvent
from ...utils.logging import get_logger
from .base import ExportSpec, event_aoi, export_image_region, init_ee

log = get_logger("gee.dem")


def _derivatives(dem, requested: list[str]):
    """Compute requested terrain derivatives. Returns ee.Image with named bands."""
    import ee

    bands = {"elevation": dem}

    if "slope" in requested:
        bands["slope"] = ee.Terrain.slope(dem)
    if "aspect" in requested:
        bands["aspect"] = ee.Terrain.aspect(dem)
    if "curvature" in requested:
        # Discrete Laplacian as a curvature proxy
        kernel = ee.Kernel.laplacian8(normalize=True)
        bands["curvature"] = dem.convolve(kernel)
    if "twi" in requested:
        # Topographic Wetness Index = ln(a / tan(beta))
        slope = ee.Terrain.slope(dem).max(0.001).multiply(3.14159 / 180.0)
        flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").rename("flow_acc")
        twi = flow_acc.add(1).log().divide(slope.tan())
        bands["twi"] = twi
    if "hand" in requested:
        # Height-Above-Nearest-Drainage from JRC/MERIT HAND if available.
        try:
            hand = ee.Image("MERIT/Hydro/v1_0_1").select("hnd")
            bands["hand"] = hand
        except Exception:
            log.warning("hand_unavailable_fallback_to_elevation")
            bands["hand"] = dem

    img = ee.Image.cat(list(bands.values())).rename(list(bands.keys()))
    return img


def download(event: DisasterEvent, cfg: DictConfig, out_dir: str | Path) -> Path:
    init_ee(project=cfg.get("project"))
    import ee

    src = cfg.get("source", "USGS/SRTMGL1_003")
    derivatives = list(cfg.get("derivatives", ["slope", "curvature"]))

    if src.endswith("AW3D30/V3_2"):
        dem = ee.ImageCollection(src).select("DSM").mosaic().rename("elevation")
    else:
        dem = ee.Image(src).select(0).rename("elevation")

    aoi = event_aoi(event)
    image = _derivatives(dem, derivatives).clip(aoi)
    out_path = Path(out_dir) / f"{event.event_id}_dem.tif"
    band_names = image.bandNames().getInfo()
    spec = ExportSpec(
        image_id=f"dem_{event.event_id}",
        image=image,
        region=aoi,
        scale_m=30.0,  # SRTM is 30m; will be resampled to 10m at tile time
        crs="EPSG:4326",
        dtype="float32",
        bands=band_names,
        n_bands_hint=len(band_names),
    )
    return export_image_region(spec, out_path, bbox_4326=event.bbox)
