"""Sentinel-2 surface reflectance, cloud-masked composites."""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from ..catalog import DisasterEvent
from ...utils.logging import get_logger
from .base import ExportSpec, event_aoi, export_image_region, init_ee

log = get_logger("gee.sentinel2")


def _mask_s2(img, max_cloud_prob: float):
    """Mask using QA60 + SCL (scene classification).

    SCL classes 3 (shadow), 8/9/10 (clouds + cirrus), 11 (snow) -> masked.
    """
    import ee
    qa = img.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    qa_clear = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    scl = img.select("SCL")
    scl_clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.updateMask(qa_clear.And(scl_clear))


def _composite_window(coll, aoi, start, end, bands, max_cloud_prob, composite):
    import ee
    c = (
        coll.filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_prob))
            .map(lambda i: _mask_s2(i, max_cloud_prob))
            .select(list(bands))
    )
    if c.size().getInfo() == 0:
        log.warning("s2_no_scenes", start=start, end=end)
        return None
    if composite == "median":
        return c.median().clip(aoi)
    if composite == "mean":
        return c.mean().clip(aoi)
    return c.first().clip(aoi)


def download(event: DisasterEvent, cfg: DictConfig, out_dir: str | Path) -> list[Path]:
    init_ee(project=cfg.get("project"))
    import ee

    aoi = event_aoi(event)
    coll = ee.ImageCollection(cfg.get("collection", "COPERNICUS/S2_SR_HARMONIZED"))
    bands = list(cfg.get("bands", ["B2", "B3", "B4", "B8"]))
    max_cloud = float(cfg.get("max_cloud_prob", 30))
    composite = cfg.get("composite", "median")

    out_paths: list[Path] = []
    windows = {
        "pre": event.pre_window,
        "post": event.post_window,
    }
    for kind, win in windows.items():
        if win is None:
            continue
        img = _composite_window(
            coll, aoi, win[0].isoformat(), win[1].isoformat(),
            bands, max_cloud, composite,
        )
        if img is None:
            continue
        out_path = Path(out_dir) / f"{event.event_id}_s2_{kind}.tif"
        spec = ExportSpec(
            image_id=f"s2_{event.event_id}_{kind}",
            image=img,
            region=aoi,
            scale_m=10.0,
            crs="EPSG:4326",
            dtype="uint16",
            bands=bands,
            n_bands_hint=len(bands),
        )
        out_paths.append(export_image_region(spec, out_path, bbox_4326=event.bbox))
    return out_paths
