"""Refine the Brazil zero-shot prediction into flood-only impact metrics.

Steps:
  1. Fetch JRC Global Surface Water permanent water mask (occurrence > 50%)
     for the AOI, aligned to the prediction grid.
  2. flood_mask = prediction AND NOT permanent_water
  3. Pull OSM buildings + major roads for AOI.
  4. Compute decision metrics on flood_mask only.

This mirrors Hu et al. 2026 Nature: detection -> downstream policy-relevant
quantity, with a defensible permanent-water control.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import ee
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

sys.path.insert(0, ".")
from geodisaster.data.gee.base import init_ee


def fetch_permanent_water_mask(bbox, ref_tif_path: Path,
                                threshold: float = 50.0,
                                project: str = "ee-bqmbill714") -> np.ndarray:
    """JRC Global Surface Water 'occurrence' >= threshold (%) reprojected onto ref grid."""
    init_ee(project=project)
    region = ee.Geometry.Rectangle(list(bbox), proj="EPSG:4326", geodesic=False)
    img = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").clip(region)

    with rasterio.open(ref_tif_path) as ref:
        crs = ref.crs
        tf = ref.transform
        h, w = ref.height, ref.width

    # Adaptive tile (single-band so plenty of room)
    cy = 0.5 * (bbox[1] + bbox[3])
    deg_lat = 1.0 / 111_320.0
    deg_lon = 1.0 / (111_320.0 * max(math.cos(math.radians(cy)), 1e-6))
    scale_m = 10
    full_h = int((bbox[3] - bbox[1]) / (scale_m * deg_lat))
    full_w = int((bbox[2] - bbox[0]) / (scale_m * deg_lon))
    max_side = 2371   # 1-band float32 fits ~2371² in 45MB
    tile_split = max(1, math.ceil(max(full_h, full_w) / max_side))
    th = (full_h + tile_split - 1) // tile_split
    tw = (full_w + tile_split - 1) // tile_split

    out = np.zeros((full_h, full_w), dtype=np.float32)
    for ri in range(tile_split):
        for ci in range(tile_split):
            row_off = ri * th
            col_off = ci * tw
            ch = min(th, full_h - row_off)
            cw = min(tw, full_w - col_off)
            if ch <= 0 or cw <= 0:
                continue
            minx = bbox[0] + col_off * scale_m * deg_lon
            maxy = bbox[3] - row_off * scale_m * deg_lat
            arr = ee.data.computePixels({
                "expression": img.float().unmask(0),
                "fileFormat": "NUMPY_NDARRAY",
                "grid": {
                    "dimensions": {"width": cw, "height": ch},
                    "affineTransform": {
                        "scaleX": scale_m * deg_lon, "shearX": 0, "translateX": minx,
                        "shearY": 0, "scaleY": -scale_m * deg_lat, "translateY": maxy,
                    },
                    "crsCode": "EPSG:4326",
                },
            })
            bands = sorted(arr.dtype.names)
            sub = arr[bands[0]].astype(np.float32)
            out[row_off:row_off + ch, col_off:col_off + cw] = sub

    # Crop/pad to ref shape
    out = out[:h, :w]
    permanent = (out >= threshold).astype(np.uint8)
    return permanent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", default="brazil_rs_2024")
    p.add_argument("--zero-shot-dir", default="outputs/zero_shot")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    event_dir = Path(args.zero_shot_dir) / args.event
    pred_tif = event_dir / f"{args.event}_pred.tif"
    summary_path = event_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    bbox = tuple(summary["bbox"])

    print(f"Event: {summary['name']}")
    print(f"bbox: {bbox}")
    print(f"Prediction: {pred_tif}")

    # 1. Pull permanent water mask
    print("\nFetching JRC permanent water mask...")
    permanent = fetch_permanent_water_mask(bbox, pred_tif)
    print(f"  permanent water: {int(permanent.sum()):,} px "
          f"({100 * permanent.sum() / permanent.size:.2f}%)")

    # 2. Read prediction + diff
    with rasterio.open(pred_tif) as src:
        pred = src.read(1)
        profile = src.profile.copy()
    flood_only = ((pred == 1) & (permanent == 0)).astype(np.uint8)
    print(f"\nPredicted water: {int((pred == 1).sum()):,} px "
          f"({100 * (pred == 1).sum() / pred.size:.2f}%)")
    print(f"Flood-only (water minus permanent): {int(flood_only.sum()):,} px "
          f"({100 * flood_only.sum() / flood_only.size:.2f}%)")

    flood_tif = event_dir / f"{args.event}_flood_only.tif"
    profile.update(count=1, dtype="uint8", nodata=255)
    with rasterio.open(flood_tif, "w", **profile) as dst:
        dst.write(flood_only, 1)
    print(f"  Saved: {flood_tif}")

    perm_tif = event_dir / f"{args.event}_permanent_water.tif"
    with rasterio.open(perm_tif, "w", **profile) as dst:
        dst.write(permanent, 1)
    print(f"  Saved: {perm_tif}")

    # 3. OSM buildings + roads
    print("\nFetching OSM buildings + major roads...")
    import osmnx as ox
    from shapely.geometry import box
    aoi = box(*bbox)

    n_bld_total = n_bld_affected = 0
    rd_km_total = rd_km_affected = 0.0
    try:
        buildings = ox.features_from_polygon(aoi, tags={"building": True})
        buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
        n_bld_total = int(len(buildings))
        if n_bld_total:
            from rasterio.mask import mask as rio_mask
            with rasterio.open(flood_tif) as src:
                for geom in buildings.geometry:
                    try:
                        arr, _ = rio_mask(src, [geom.__geo_interface__], crop=True,
                                          all_touched=True, filled=False)
                        data = arr.compressed() if hasattr(arr, "compressed") else arr.ravel()
                        if data.size and (data == 1).mean() >= 0.2:
                            n_bld_affected += 1
                    except Exception:
                        continue
    except Exception as e:
        print(f"  buildings err: {e}")

    try:
        cf = '["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]'
        g = ox.graph_from_polygon(aoi, custom_filter=cf, simplify=True)
        edges = ox.graph_to_gdfs(g, nodes=False).reset_index()
        edges_m = edges.to_crs(edges.estimate_utm_crs())
        rd_km_total = round(float(edges_m.length.sum() / 1000.0), 2)
        from rasterio.mask import mask as rio_mask
        with rasterio.open(flood_tif) as src:
            for i, row in edges.iterrows():
                try:
                    arr, _ = rio_mask(src, [row.geometry.__geo_interface__],
                                      crop=True, all_touched=True, filled=False)
                    data = arr.compressed() if hasattr(arr, "compressed") else arr.ravel()
                    if data.size and (data == 1).mean() >= 0.15:
                        rd_km_affected += edges_m.length.iloc[i] / 1000.0
                except Exception:
                    continue
        rd_km_affected = round(rd_km_affected, 2)
    except Exception as e:
        print(f"  roads err: {e}")

    print(f"\nOSM buildings in AOI: {n_bld_total}")
    print(f"  flood-affected: {n_bld_affected}  ({100 * n_bld_affected / max(n_bld_total, 1):.2f}%)")
    print(f"OSM major roads in AOI: {rd_km_total} km")
    print(f"  flood-affected: {rd_km_affected} km  ({100 * rd_km_affected / max(rd_km_total, 1):.2f}%)")

    # 4. Save full summary
    summary["permanent_water_pct"] = round(100 * permanent.sum() / permanent.size, 3)
    summary["flood_only_water_pct"] = round(100 * flood_only.sum() / flood_only.size, 3)
    summary["flood_only_km2"] = round(float(flood_only.sum() * 100 / 1e6), 2)
    summary["buildings_total_osm"] = n_bld_total
    summary["buildings_flood_affected"] = n_bld_affected
    summary["roads_total_km_osm"] = rd_km_total
    summary["roads_flood_affected_km"] = rd_km_affected
    summary["buildings_affected_pct"] = round(100 * n_bld_affected / max(n_bld_total, 1), 2)
    summary["roads_affected_pct"] = round(100 * rd_km_affected / max(rd_km_total, 1), 2)

    out_path = Path(args.out) if args.out else event_dir / "flood_decision_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved final summary: {out_path}")


if __name__ == "__main__":
    main()
