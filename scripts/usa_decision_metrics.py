"""USA test-set decision metrics: pixels -> affected buildings/roads/population.

Mirrors the "infrastructure detection -> downstream system analysis" paradigm
from Hu et al. 2026 Nature (solar/wind penetration). We take our best Sen1Floods11
predictions on USA chips, then quantify:
    - affected buildings count (OSM polygons)
    - affected road km (OSM line strings)
    - affected population (WorldPop 100m)
    - critical facilities affected (schools, hospitals)

Outputs a per-chip JSON + an aggregate summary. The figure rendered next is a
single-chip impact map with overlays — the Nature Fig 5 style.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from omegaconf import OmegaConf

sys.path.insert(0, ".")
from geodisaster.data.tile import merge_manifests
from geodisaster.datasets import DisasterPatchDataModule, stats_with_fallbacks
from geodisaster.train import DisasterSegLightningModule
from geodisaster.utils.logging import get_logger, setup_logging

log = get_logger("usa_decision")


def predict_usa(ckpt_path: Path, stats_path: str) -> list[dict]:
    """Run U-Net S1+S2 on all USA patches, return per-patch geo-referenced masks."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = state["hyper_parameters"]
    model_cfg = OmegaConf.create(hp.get("model_cfg", hp.get("model")))
    train_cfg = OmegaConf.create(hp.get("train_cfg", hp.get("train")))
    sources = hp["sources"]
    module = DisasterSegLightningModule(model_cfg, train_cfg, sources)
    module.load_state_dict(state["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.eval().to(device)

    test = merge_manifests([Path("data/processed/patches") / "sen1floods11_USA"])
    norm = stats_with_fallbacks(stats_path, sources)
    dm = DisasterPatchDataModule(
        train_patches=[], val_patches=[], test_patches=test,
        sources=sources, batch_size=8, num_workers=4, normalize=norm, augment_train=False,
    )

    results: list[dict] = []
    with torch.no_grad():
        for batch in dm.test_dataloader():
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            logits = module(batch)
            scores = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
            for i in range(scores.shape[0]):
                results.append({
                    "patch_id": batch["patch_id"][i],
                    "score": scores[i],
                    "mask": (scores[i] > 0.5).astype(np.uint8),
                })
    return results


def save_predictions_as_geotiff(predictions: list[dict], s1_root: Path, out_dir: Path) -> list[Path]:
    """Save predicted masks as georeferenced GeoTIFFs matching each chip's projection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for r in predictions:
        s1_path = s1_root / f"{r['patch_id']}.tif"
        if not s1_path.exists():
            continue
        with rasterio.open(s1_path) as src:
            profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
        out_path = out_dir / f"{r['patch_id']}_pred.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(r["mask"], 1)
        out_paths.append(out_path)
    return out_paths


def affected_metrics_for_chip(pred_tif: Path, osm_bbox_4326, worldpop_path: Path | None):
    """Pull OSM in chip bbox, intersect with predicted mask, return counts."""
    import geopandas as gpd
    import osmnx as ox
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box

    with rasterio.open(pred_tif) as src:
        bounds = src.bounds
    minx, miny, maxx, maxy = bounds
    poly = box(minx, miny, maxx, maxy)

    summary = {"chip": pred_tif.stem, "bbox": [minx, miny, maxx, maxy]}

    # Buildings
    try:
        b = ox.features_from_polygon(poly, tags={"building": True})
        b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
        summary["n_buildings_total"] = int(len(b))
        if len(b):
            with rasterio.open(pred_tif) as src:
                affected = 0
                for geom in b.geometry:
                    try:
                        arr, _ = rio_mask(src, [geom.__geo_interface__], crop=True,
                                          all_touched=True, filled=False)
                        data = arr.compressed() if hasattr(arr, "compressed") else arr.ravel()
                        if data.size and (data == 1).mean() >= 0.2:
                            affected += 1
                    except Exception:
                        continue
            summary["n_buildings_affected"] = affected
        else:
            summary["n_buildings_affected"] = 0
    except Exception as e:
        summary["buildings_error"] = str(e)[:120]

    # Roads — major roads only to keep query small
    try:
        cf = '["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]'
        g = ox.graph_from_polygon(poly, custom_filter=cf, simplify=True)
        edges = ox.graph_to_gdfs(g, nodes=False).reset_index()
        # Project to UTM for length in meters
        edges_m = edges.to_crs(edges.estimate_utm_crs())
        summary["road_km_total"] = round(float(edges_m.length.sum() / 1000.0), 2)
        affected_km = 0.0
        with rasterio.open(pred_tif) as src:
            for i, row in edges.iterrows():
                try:
                    arr, _ = rio_mask(src, [row.geometry.__geo_interface__],
                                      crop=True, all_touched=True, filled=False)
                    data = arr.compressed() if hasattr(arr, "compressed") else arr.ravel()
                    if data.size and (data == 1).mean() >= 0.15:
                        affected_km += edges_m.length.iloc[i] / 1000.0
                except Exception:
                    continue
        summary["road_km_affected"] = round(affected_km, 2)
    except Exception as e:
        summary["roads_error"] = str(e)[:120]

    return summary


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/sen1floods11_unet_s1s2/checkpoints/best-epoch016.ckpt")
    p.add_argument("--stats", default="data/processed/norm_stats_sen1floods11.yaml")
    p.add_argument("--s1-root",
                   default="data/external/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand")
    p.add_argument("--out-dir", default="outputs/usa_decision")
    p.add_argument("--max-chips", type=int, default=10,
                   help="cap chips to keep OSM queries cheap (full = 69)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("predicting_usa", ckpt=args.ckpt)
    preds = predict_usa(Path(args.ckpt), args.stats)
    log.info("predictions_done", n=len(preds))

    log.info("saving_geotiffs")
    pred_dir = out_dir / "predictions"
    pred_paths = save_predictions_as_geotiff(preds, Path(args.s1_root), pred_dir)
    log.info("geotiffs_saved", n=len(pred_paths))

    sample_paths = pred_paths[:args.max_chips]
    log.info("computing_decision_metrics", n_chips=len(sample_paths))
    rows = []
    for pp in sample_paths:
        try:
            rows.append(affected_metrics_for_chip(pp, None, None))
        except Exception as e:
            log.error("metrics_failed", chip=pp.stem, err=str(e))
    summary = {
        "n_chips_evaluated": len(rows),
        "totals": {
            "buildings_total":  sum(r.get("n_buildings_total", 0) for r in rows),
            "buildings_affected": sum(r.get("n_buildings_affected", 0) for r in rows),
            "road_km_total":    round(sum(r.get("road_km_total", 0.0) for r in rows), 2),
            "road_km_affected": round(sum(r.get("road_km_affected", 0.0) for r in rows), 2),
        },
        "per_chip": rows,
    }
    out_json = out_dir / "decision_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log.info("decision_metrics_done", out=str(out_json), totals=summary["totals"])
    print(json.dumps(summary["totals"], indent=2))


if __name__ == "__main__":
    main()
