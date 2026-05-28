"""Ingest a Copernicus EMS Rapid Mapping activation into a *ground-truth*
validation patch set, for real-event end-to-end validation (Nature P0).

EMS Rapid Mapping vector packages contain an "observed event area" layer — the
analyst-delineated flood extent from satellite imagery. We treat that polygon as
ground truth: fetch the matching Sentinel-1/2 imagery over the AOI (reusing the
zero-shot GEE pipeline), rasterise the EMS polygon onto the same grid, tile to
512, and save patches with REAL flood labels. The Sen1Floods11-trained model can
then be evaluated on genuinely unseen recent events with quantified accuracy,
and the reasoning layer validated against the ground-truth mask.

Access note: EMS vector packages are gated (STAC API needs an account token; no
public per-product download URL). Download the activation's vector package zip
from https://rapidmapping.emergency.copernicus.eu/<CODE> (or via a STAC token),
extract it into --ems-dir, then run this script. Use --parse-only to verify the
vector parsing before any GEE fetch.

Standard EMS layers handled: observed-event area (flood extent), area-of-interest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

sys.path.insert(0, ".")

OBSERVED_PATTERNS = ["observedevent", "obsevent", "observed_event", "_obs_"]
AOI_PATTERNS = ["areaofinterest", "area_of_interest", "_aoi", "aoi_"]
FLOOD_HINTS = ["flood", "water", "inundat"]


def _read_all_vectors(ems_dir: Path) -> list[tuple[str, gpd.GeoDataFrame]]:
    layers = []
    for ext in ("*.shp", "*.gpkg", "*.geojson", "*.json"):
        for f in ems_dir.rglob(ext):
            try:
                if f.suffix == ".gpkg":
                    import fiona
                    for ln in fiona.listlayers(f):
                        layers.append((f"{f.stem}:{ln}", gpd.read_file(f, layer=ln)))
                else:
                    layers.append((f.stem, gpd.read_file(f)))
            except Exception as e:
                print(f"  skip {f.name}: {e}")
    return layers


def _is_polygon(gdf) -> bool:
    return bool(len(gdf)) and gdf.geom_type.isin(["Polygon", "MultiPolygon"]).any()


def parse_ems_package(ems_dir: Path) -> dict:
    """Return {flood_gdf, aoi_bbox(4326), crs, n_layers, source_layers}."""
    layers = _read_all_vectors(ems_dir)
    if not layers:
        raise RuntimeError(f"No vector files found under {ems_dir}")
    obs, aoi = [], []
    for name, gdf in layers:
        nl = name.lower()
        if not _is_polygon(gdf):
            continue
        if any(p in nl for p in OBSERVED_PATTERNS):
            obs.append((name, gdf))
        elif any(p in nl for p in AOI_PATTERNS):
            aoi.append((name, gdf))
    # fallback: attribute-based flood detection if no observed-event layer matched
    if not obs:
        for name, gdf in layers:
            if not _is_polygon(gdf):
                continue
            cols = " ".join(map(str, gdf.columns)).lower()
            vals = " ".join(map(str, gdf.select_dtypes("object").head(50).values.ravel())).lower()
            if any(h in cols or h in vals for h in FLOOD_HINTS):
                obs.append((name, gdf))
    if not obs:
        raise RuntimeError("No observed-event / flood polygon layer found. "
                           f"Layers seen: {[n for n,_ in layers]}")

    parts = []
    for _, gdf in obs:
        g = gdf.to_crs(4326)
        parts.append(unary_union(g.geometry.values))
    flood_geom = unary_union(parts)
    flood_gdf = gpd.GeoDataFrame(geometry=[flood_geom], crs=4326)

    if aoi:
        a = aoi[0][1].to_crs(4326)
        bbox = tuple(a.total_bounds)
    else:
        bbox = tuple(flood_gdf.total_bounds)
    return {"flood_gdf": flood_gdf, "aoi_bbox": [float(x) for x in bbox],
            "source_layers": [n for n, _ in obs], "n_layers": len(layers)}


def rasterize_gt(flood_gdf, transform, height, width) -> np.ndarray:
    from rasterio.features import rasterize
    shapes = [(geom, 1) for geom in flood_gdf.geometry]
    mask = rasterize(shapes, out_shape=(height, width), transform=transform,
                     fill=0, dtype="uint8")
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ems-dir", required=True, help="extracted EMS vector package dir")
    p.add_argument("--code", required=True, help="activation code, e.g. EMSR871")
    p.add_argument("--start", help="imagery window start YYYY-MM-DD (event date)")
    p.add_argument("--end", help="imagery window end YYYY-MM-DD")
    p.add_argument("--out-root", default="data/processed/patches")
    p.add_argument("--scale-m", type=float, default=10.0)
    p.add_argument("--parse-only", action="store_true",
                   help="just parse + report the flood polygon / bbox (no GEE)")
    p.add_argument("--project", default="ee-bqmbill714")
    args = p.parse_args()

    info = parse_ems_package(Path(args.ems_dir))
    bbox = info["aoi_bbox"]
    area_km2 = float(info["flood_gdf"].to_crs(3857).area.sum()) / 1e6
    print(f"[{args.code}] observed-event layers: {info['source_layers']}")
    print(f"  AOI bbox (4326): {bbox}")
    print(f"  flood polygon area: {area_km2:.1f} km^2")
    out_dir = Path(args.out_root) / f"ems_{args.code}"
    out_dir.mkdir(parents=True, exist_ok=True)
    info["flood_gdf"].to_file(out_dir / "observed_flood.geojson", driver="GeoJSON")
    (out_dir / "ems_meta.json").write_text(json.dumps(
        {"code": args.code, "bbox": bbox, "flood_area_km2": area_km2,
         "source_layers": info["source_layers"], "start": args.start, "end": args.end}, indent=2))
    print(f"  wrote {out_dir}/observed_flood.geojson + ems_meta.json")
    if args.parse_only:
        return 0

    # ---- GEE fetch + rasterise GT + tile (reuse zero-shot machinery) ----
    if not (args.start and args.end):
        print("ERROR: --start and --end required for GEE fetch (omit --parse-only "
              "only when you pass the event window).")
        return 1
    import ee
    from scripts.deploy_zero_shot import (fetch_s1_composite, fetch_s2_composite,
                                          compute_pixels, tile_to_512)
    ee.Initialize(project=args.project)
    print("  fetching Sentinel-1/2 composites from GEE ...")
    s1 = fetch_s1_composite(bbox, args.start, args.end)
    s2 = fetch_s2_composite(bbox, args.start, args.end)
    s1_arr, transform = compute_pixels(s1, ee.Geometry.Rectangle(list(bbox)), bbox,
                                       scale_m=args.scale_m, n_bands_hint=2)
    s2_arr, _ = compute_pixels(s2, ee.Geometry.Rectangle(list(bbox)), bbox,
                               scale_m=args.scale_m, n_bands_hint=13)
    H, W = s1_arr.shape[1], s1_arr.shape[2]
    gt = rasterize_gt(info["flood_gdf"], transform, H, W)
    print(f"  imagery {s1_arr.shape} / {s2_arr.shape}; GT flood pixels: {gt.sum()} "
          f"({100*gt.mean():.2f}% of AOI)")

    s1_tiles = tile_to_512(s1_arr, transform); s2_tiles = tile_to_512(s2_arr, transform)
    gt_tiles = tile_to_512(gt[None], transform)
    records = []
    for i, ((s1t, _), (s2t, _), (gtt, _)) in enumerate(zip(s1_tiles, s2_tiles, gt_tiles)):
        pid = f"{args.code}_t{i:03d}"
        s1p = out_dir / f"{pid}__sentinel1.npy"; s2p = out_dir / f"{pid}__sentinel2.npy"
        lp = out_dir / f"{pid}__label.npy"
        np.save(s1p, s1t.astype(np.float32)); np.save(s2p, s2t.astype(np.float32))
        np.save(lp, gtt[0].astype(np.uint8))
        records.append({"patch_id": pid, "row": 0, "col": 0, "size": 512,
                        "sources": {"sentinel1": str(s1p), "sentinel2": str(s2p)},
                        "label_path": str(lp), "pos_fraction": float(gtt[0].mean())})
    (out_dir / "manifest.json").write_text(json.dumps(
        {"event_id": f"ems_{args.code}", "hazard": "flood", "patch_size": 512,
         "stride": 512, "n_patches": len(records), "ref_source": "sentinel2",
         "ground_truth": "Copernicus EMS observed event", "patches": records}, indent=2))
    print(f"  saved {len(records)} GT validation patches -> {out_dir}")


if __name__ == "__main__":
    sys.exit(main() or 0)
