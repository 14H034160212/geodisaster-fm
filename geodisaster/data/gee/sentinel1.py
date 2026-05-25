"""Sentinel-1 SAR pre/post composites.

Outputs two GeoTIFFs per event: ``{event_id}_s1_pre.tif`` and
``{event_id}_s1_post.tif``. Each carries VV + VH (dB) by default. SAR is the
primary signal for flood/inundation per data-requirements §3.1.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from omegaconf import DictConfig

from ..catalog import DisasterEvent
from ...utils.logging import get_logger
from .base import ExportSpec, event_aoi, export_image_region, init_ee

log = get_logger("gee.sentinel1")


def _refined_lee(img):
    """Lee speckle filter (Lopes et al. 1990 / refined Lee).

    Implementation follows the GEE community recipe; works on log-scale dB.
    """
    import ee
    bands = img.bandNames()

    def _per_band(b):
        b = ee.String(b)
        band = img.select([b])
        kernel = ee.Kernel.square(2)
        mean = band.reduceNeighborhood(ee.Reducer.mean(), kernel)
        var = band.reduceNeighborhood(ee.Reducer.variance(), kernel)
        sigma_v = var.divide(mean.multiply(mean)).reduceNeighborhood(
            ee.Reducer.mean(), kernel
        )
        b_coef = var.subtract(mean.multiply(mean).multiply(sigma_v)) \
                    .divide(var.multiply(sigma_v.add(1)))
        return mean.add(b_coef.multiply(band.subtract(mean))).rename([b])

    return ee.ImageCollection(bands.map(_per_band)).toBands().rename(bands)


def _composite(coll, aoi, start: str, end: str, polarizations, orbit: str | None,
               filter_speckle: bool, kind: str):
    import ee
    c = (
        coll.filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
    )
    for p in polarizations:
        c = c.filter(ee.Filter.listContains("transmitterReceiverPolarisation", p))
    if orbit:
        c = c.filter(ee.Filter.eq("orbitProperties_pass", orbit))
    c = c.select(list(polarizations))

    n = c.size().getInfo()
    if n == 0:
        log.warning("s1_no_scenes", start=start, end=end, polarizations=list(polarizations))
        return None
    log.info("s1_composite", kind=kind, start=start, end=end, scenes=n)

    if kind == "median":
        img = c.median()
    elif kind == "min":
        img = c.min()
    elif kind == "max":
        img = c.max()
    else:
        img = c.mean()

    if filter_speckle:
        img = _refined_lee(img)
    return img.clip(aoi)


def download(event: DisasterEvent, cfg: DictConfig, out_dir: str | Path) -> list[Path]:
    init_ee(project=cfg.get("project"))
    import ee

    if event.pre_window is None or event.post_window is None:
        raise ValueError(f"{event.event_id} needs pre_window and post_window for S1")
    aoi = event_aoi(event)
    coll = ee.ImageCollection(cfg.get("collection", "COPERNICUS/S1_GRD"))
    polarizations = list(cfg.get("polarizations", ["VV", "VH"]))
    orbit = cfg.get("orbit", None)
    speckle = (cfg.get("speckle_filter") or "").lower() in {"refined_lee", "lee"}
    composite = cfg.get("composite", "median")

    out_paths: list[Path] = []
    for kind, win in (("pre", event.pre_window), ("post", event.post_window)):
        start, end = win[0].isoformat(), win[1].isoformat()
        img = _composite(coll, aoi, start, end, polarizations, orbit, speckle, composite)
        if img is None:
            continue
        out_path = Path(out_dir) / f"{event.event_id}_s1_{kind}.tif"
        spec = ExportSpec(
            image_id=f"s1_{event.event_id}_{kind}",
            image=img,
            region=aoi,
            scale_m=10.0,
            crs="EPSG:4326",
            dtype="float32",
            bands=polarizations,
            n_bands_hint=len(polarizations),
        )
        out_paths.append(export_image_region(spec, out_path, bbox_4326=event.bbox))
    return out_paths
