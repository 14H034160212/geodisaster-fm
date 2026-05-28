"""Stage 1 (slow, run once): cache per-building decision features for real
Sen1Floods11 flood events, so the structured-decision method + baselines can be
compared offline without re-hitting OSM.

For each chip: predict the flood PROBABILITY map, save a georeferenced score
GeoTIFF, fetch OSM buildings, and record per building:
  mean_prob (evidence) · flood_frac_hard (>0.5 fraction, for B1/B2) ·
  centroid in metric CRS (for the graph) · area · gt_affected (from LabelHand).
Output: outputs/decision/features_<region>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

log = get_logger("decision_feat")
HAND = Path("data/external/sen1floods11/v1.1/data/flood_events/HandLabeled")


def predict_scores(ckpt_path, stats_path, region):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = state["hyper_parameters"]
    mcfg = OmegaConf.create(hp.get("model_cfg", hp.get("model")))
    tcfg = OmegaConf.create(hp.get("train_cfg", hp.get("train")))
    sources = hp["sources"]
    m = DisasterSegLightningModule(mcfg, tcfg, sources)
    m.load_state_dict(state["state_dict"], strict=True)
    m.eval()
    patches = merge_manifests([Path("data/processed/patches") / f"sen1floods11_{region}"])
    norm = stats_with_fallbacks(stats_path, sources)
    dm = DisasterPatchDataModule(train_patches=[], val_patches=[], test_patches=patches,
                                 sources=sources, batch_size=8, num_workers=4, normalize=norm)
    out = {}
    with torch.no_grad():
        for b in dm.test_dataloader():
            sc = torch.sigmoid(m(b).squeeze(1)).cpu().numpy()
            for i in range(sc.shape[0]):
                out[b["patch_id"][i]] = sc[i].astype(np.float32)
    return out


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/sen1floods11_unet_s1s2/checkpoints/best-epoch016.ckpt")
    p.add_argument("--stats", default="data/processed/norm_stats_sen1floods11.yaml")
    p.add_argument("--region", default="USA")
    p.add_argument("--max-chips", type=int, default=20)
    p.add_argument("--min-buildings", type=int, default=15)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import time as _time
    import geopandas as gpd
    import osmnx as ox
    ox.settings.requests_timeout = 180
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box

    MIRRORS = ["https://overpass-api.de/api/interpreter",
               "https://overpass.kumi.systems/api/interpreter",
               "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]

    def fetch_buildings(poly, pid, cache_dir, retries=3):
        cache = cache_dir / f"{pid}_buildings.geojson"
        if cache.exists():
            try:
                g = gpd.read_file(cache)
                return g if len(g) else None
            except Exception:
                pass
        for attempt in range(retries):
            mirror = MIRRORS[attempt % len(MIRRORS)]
            ox.settings.overpass_url = mirror
            try:
                b = ox.features_from_polygon(poly, tags={"building": True})
                b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
                b = b[["geometry"]].copy()
                b.to_file(cache, driver="GeoJSON")
                return b if len(b) else None
            except Exception as e:
                log.warning("osm_retry", chip=pid, attempt=attempt, mirror=mirror.split("//")[1][:20],
                            err=str(e)[:60])
                _time.sleep(5 * (attempt + 1))
        return None

    out = Path(args.out or f"outputs/decision/features_{args.region}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    score_dir = out.parent / "score_tifs"; score_dir.mkdir(exist_ok=True)
    osm_cache = out.parent / "osm_cache"; osm_cache.mkdir(exist_ok=True)

    def _save(feats, n_chips, t0):
        out.write_text(json.dumps({
            "region": args.region, "n_chips": n_chips, "n_buildings": len(feats),
            "gt_affected_total": int(sum(f["gt_affected"] for f in feats)),
            "wall_time_min": round((time.time() - t0) / 60, 1), "features": feats}, indent=2))

    log.info("predict_scores", region=args.region)
    scores = predict_scores(Path(args.ckpt), args.stats, args.region)

    # rank chips by GT water fraction → focus on flooded chips
    ranked = []
    for pid in scores:
        gt_tif = HAND / "LabelHand" / f"{pid.replace('S1Hand','LabelHand')}.tif"
        if gt_tif.exists():
            with rasterio.open(gt_tif) as src:
                ranked.append((pid, float((src.read(1) == 1).mean()), gt_tif))
    ranked.sort(key=lambda t: -t[1])

    feats = []; n_chips = 0; t0 = time.time()
    for pid, gtwater, gt_tif in ranked:
        if n_chips >= args.max_chips:
            break
        s1_tif = HAND / "S1Hand" / f"{pid}.tif"
        if not s1_tif.exists():
            continue
        # write score (prob) geotiff with the chip's projection
        with rasterio.open(s1_tif) as src:
            prof = src.profile.copy()
        prof.update(count=1, dtype="float32", nodata=None, compress="deflate")
        score_tif = score_dir / f"{pid}_score.tif"
        with rasterio.open(score_tif, "w", **prof) as dst:
            dst.write(scores[pid], 1)
        with rasterio.open(gt_tif) as src:
            b = src.bounds
        poly = box(b.left, b.bottom, b.right, b.top)
        bld = fetch_buildings(poly, pid, osm_cache)
        if bld is None or len(bld) < args.min_buildings:
            continue
        utm = bld.estimate_utm_crs()
        cent = bld.to_crs(utm).geometry.centroid
        area = bld.to_crs(utm).geometry.area
        n_chips += 1
        with rasterio.open(score_tif) as ssrc, rasterio.open(gt_tif) as gsrc:
            for k, geom in enumerate(bld.geometry):
                gj = [geom.__geo_interface__]
                try:
                    sa, _ = rio_mask(ssrc, gj, crop=True, all_touched=True, filled=False)
                    sd = sa.compressed() if hasattr(sa, "compressed") else sa.ravel()
                    ga, _ = rio_mask(gsrc, gj, crop=True, all_touched=True, filled=False)
                    gd = ga.compressed() if hasattr(ga, "compressed") else ga.ravel()
                except Exception:
                    continue
                if sd.size == 0 or gd.size == 0:
                    continue
                feats.append({
                    "region": args.region, "chip": pid, "bld": k,
                    "mean_prob": float(sd.mean()),
                    "flood_frac_hard": float((sd > 0.5).mean()),
                    "area_m2": float(area.iloc[k]),
                    "cx": float(cent.iloc[k].x), "cy": float(cent.iloc[k].y),
                    "gt_affected": bool((gd == 1).mean() >= 0.2),
                })
        log.info("chip_done", chip=pid, n_buildings=len(bld), n_chips=n_chips,
                 gt_water=round(gtwater, 3))
        _save(feats, n_chips, t0)   # incremental: never lose progress to OSM flakiness

    meta = {"region": args.region, "n_chips": n_chips, "n_buildings": len(feats),
            "gt_affected_total": int(sum(f["gt_affected"] for f in feats)),
            "wall_time_min": round((time.time() - t0) / 60, 1), "features": feats}
    out.write_text(json.dumps(meta, indent=2))
    print(f"cached {len(feats)} buildings over {n_chips} chips "
          f"({meta['gt_affected_total']} GT-affected) -> {out} "
          f"[{meta['wall_time_min']} min]")


if __name__ == "__main__":
    sys.exit(main() or 0)
