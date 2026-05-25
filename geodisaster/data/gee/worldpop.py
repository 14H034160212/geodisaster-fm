"""WorldPop unconstrained 100 m population count."""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from ..catalog import DisasterEvent
from .base import ExportSpec, event_aoi, export_image_region, init_ee


def download(event: DisasterEvent, cfg: DictConfig, out_dir: str | Path) -> Path:
    init_ee(project=cfg.get("project"))
    import ee

    year_cfg = cfg.get("year", "latest_pre_event")
    if year_cfg == "latest_pre_event" and event.pre_window is not None:
        year = event.pre_window[1].year
    elif isinstance(year_cfg, int):
        year = year_cfg
    else:
        year = (event.event_date or event.pre_window[1]).year - 1

    coll_id = cfg.get("collection", "WorldPop/GP/100m/pop")
    aoi = event_aoi(event)
    coll = ee.ImageCollection(coll_id).filter(ee.Filter.eq("year", year))
    if coll.size().getInfo() == 0:
        coll = ee.ImageCollection(coll_id).sort("year", False).limit(1)
    image = coll.mosaic().clip(aoi).rename("population")

    out_path = Path(out_dir) / f"{event.event_id}_worldpop_{year}.tif"
    spec = ExportSpec(
        image_id=f"worldpop_{event.event_id}_{year}",
        image=image,
        region=aoi,
        scale_m=100.0,
        crs="EPSG:4326",
        dtype="float32",
        bands=["population"],
        n_bands_hint=1,
    )
    return export_image_region(spec, out_path, bbox_4326=event.bbox)
