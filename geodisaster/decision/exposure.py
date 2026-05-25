"""Exposure decision metrics (proposal §3 Task 3, §8 decision-metric validation).

Given a predicted impact-area raster and auxiliary vectors (buildings, roads,
facilities) + a WorldPop raster, compute:
    - affected_buildings : count + footprint area
    - affected_road_length : km, by highway class
    - affected_population : sum of pop within mask
    - facility_exposure : per-facility-type counts
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ExposureResult:
    event_id: str
    metric: str
    total: float
    by_class: dict[str, float]
    units: str

    def as_dict(self) -> dict:
        return {
            "event_id": self.event_id, "metric": self.metric,
            "total": self.total, "by_class": self.by_class, "units": self.units,
        }


def _open_mask(mask_path: str | Path):
    import rasterio
    with rasterio.open(mask_path) as src:
        return src.read(1), src.transform, src.crs


def _polys_to_gdf(path: str | Path):
    import geopandas as gpd
    return gpd.read_file(path)


def _sample_raster_at_points(raster_path, points_gdf):
    import rasterio
    with rasterio.open(raster_path) as src:
        coords = [(geom.x, geom.y) for geom in points_gdf.geometry]
        return np.array([v[0] for v in src.sample(coords)])


def _mask_sample_at_geometries(mask_path, gdf, statistic: str = "mean") -> np.ndarray:
    """For each geometry, fraction (or sum) of mask pixels inside it == 1."""
    import rasterio
    from rasterio.mask import mask as rio_mask

    out = np.zeros(len(gdf), dtype=np.float32)
    with rasterio.open(mask_path) as src:
        for i, geom in enumerate(gdf.geometry):
            if geom is None or geom.is_empty:
                continue
            try:
                arr, _ = rio_mask(src, [geom.__geo_interface__], crop=True, all_touched=True, filled=False)
            except Exception:
                continue
            data = arr.compressed() if hasattr(arr, "compressed") else arr[arr != src.nodata]
            if data.size == 0:
                continue
            if statistic == "mean":
                out[i] = float((data == 1).mean())
            elif statistic == "any":
                out[i] = float((data == 1).any())
            elif statistic == "sum":
                out[i] = float((data == 1).sum())
    return out


def affected_buildings(
    event_id: str,
    impact_mask_path: str | Path,
    buildings_path: str | Path,
    threshold: float = 0.3,
) -> ExposureResult:
    gdf = _polys_to_gdf(buildings_path)
    intersect_frac = _mask_sample_at_geometries(impact_mask_path, gdf, statistic="mean")
    affected = intersect_frac >= threshold
    total = int(affected.sum())
    # group by 'building' tag if present
    by_class: dict[str, float] = {}
    if "building" in gdf.columns:
        gdf["__affected"] = affected
        by_class = (
            gdf.groupby(gdf["building"].astype(str))["__affected"].sum().astype(int).to_dict()
        )
    return ExposureResult(
        event_id=event_id, metric="affected_buildings",
        total=float(total), by_class={k: float(v) for k, v in by_class.items()},
        units="count",
    )


def affected_road_length(
    event_id: str,
    impact_mask_path: str | Path,
    roads_path: str | Path,
    threshold: float = 0.2,
) -> ExposureResult:
    import geopandas as gpd
    gdf = _polys_to_gdf(roads_path)
    intersect_frac = _mask_sample_at_geometries(impact_mask_path, gdf, statistic="mean")
    affected = intersect_frac >= threshold
    # Project to a meters CRS (UTM auto) for length
    gdf_m = gdf.to_crs(gdf.estimate_utm_crs())
    lengths_m = gdf_m.length.values
    total_km = float(lengths_m[affected].sum() / 1000.0)
    by_class: dict[str, float] = {}
    if "highway" in gdf.columns:
        gdf["__affected"] = affected
        gdf["__len_km"] = lengths_m / 1000.0
        by_class = (
            gdf[gdf["__affected"]].groupby(gdf["highway"].astype(str))["__len_km"]
                                  .sum().to_dict()
        )
    return ExposureResult(
        event_id=event_id, metric="affected_road_length",
        total=total_km, by_class={k: float(v) for k, v in by_class.items()},
        units="km",
    )


def affected_population(
    event_id: str,
    impact_mask_path: str | Path,
    population_path: str | Path,
) -> ExposureResult:
    import rasterio
    from rasterio.warp import reproject, Resampling

    mask, mask_tf, mask_crs = _open_mask(impact_mask_path)
    with rasterio.open(population_path) as pop_src:
        pop = np.zeros(mask.shape, dtype=np.float32)
        reproject(
            source=rasterio.band(pop_src, 1), destination=pop,
            src_transform=pop_src.transform, src_crs=pop_src.crs,
            dst_transform=mask_tf, dst_crs=mask_crs,
            resampling=Resampling.bilinear,
        )
    # WorldPop pixels are people/100m^2 (already counts), so resampling preserves intent if
    # we use a sum-preserving downsample; for simplicity we use bilinear and report relative.
    affected_pop = float(pop[mask == 1].sum())
    return ExposureResult(
        event_id=event_id, metric="affected_population",
        total=affected_pop, by_class={}, units="people",
    )


def facility_exposure(
    event_id: str,
    impact_mask_path: str | Path,
    facilities_path: str | Path,
    threshold: float = 0.2,
) -> ExposureResult:
    gdf = _polys_to_gdf(facilities_path)
    # Some facilities are points; buffer slightly for raster sampling
    if (gdf.geometry.type == "Point").any():
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.buffer(0.0003)  # ~30m at JP latitudes
    intersect_frac = _mask_sample_at_geometries(impact_mask_path, gdf, statistic="any")
    affected = intersect_frac >= threshold
    by_class: dict[str, float] = {}
    for col in ("amenity", "emergency"):
        if col in gdf.columns:
            gdf["__affected"] = affected
            grp = gdf.groupby(gdf[col].astype(str))["__affected"].sum().astype(int).to_dict()
            for k, v in grp.items():
                if k and k != "nan":
                    by_class[k] = float(v)
    return ExposureResult(
        event_id=event_id, metric="facility_exposure",
        total=float(affected.sum()), by_class=by_class, units="count",
    )
