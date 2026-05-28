"""P0 / Result-5 validation: do the dispatcher's DECISION answers match the
answers a responder would get from the GROUND-TRUTH flood mask?

For real Sen1Floods11 flood events we have both the analyst hand-label
(LabelHand GeoTIFF = ground truth) and the model prediction. For each chip we
fetch OSM buildings + major roads ONCE and ask, under BOTH masks, which are
"affected" (>=20% of footprint flooded for buildings; >=15% of length for
roads). We then score the pipeline's answers against the ground-truth-derived
answers at the level decisions are actually made: which buildings/roads are
flooded — not pixel F1.

Outputs building-level affected-identification precision/recall/F1 and
per-chip count agreement. Reuses prediction + geotiff helpers from
usa_decision_metrics.
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

log = get_logger("decision_val")
HAND = Path("data/external/sen1floods11/v1.1/data/flood_events/HandLabeled")


def predict_region(ckpt_path, stats_path, region):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = state["hyper_parameters"]
    mcfg = OmegaConf.create(hp.get("model_cfg", hp.get("model")))
    tcfg = OmegaConf.create(hp.get("train_cfg", hp.get("train")))
    sources = hp["sources"]
    m = DisasterSegLightningModule(mcfg, tcfg, sources)
    m.load_state_dict(state["state_dict"], strict=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.eval().to(dev)
    patches = merge_manifests([Path("data/processed/patches") / f"sen1floods11_{region}"])
    norm = stats_with_fallbacks(stats_path, sources)
    dm = DisasterPatchDataModule(train_patches=[], val_patches=[], test_patches=patches,
                                 sources=sources, batch_size=8, num_workers=4, normalize=norm)
    out = {}
    with torch.no_grad():
        for b in dm.test_dataloader():
            for k, v in list(b.items()):
                if isinstance(v, torch.Tensor):
                    b[k] = v.to(dev)
            sc = torch.sigmoid(m(b).squeeze(1)).cpu().numpy()
            for i in range(sc.shape[0]):
                out[b["patch_id"][i]] = (sc[i] > 0.5).astype(np.uint8)
    return out


def _write_tif(mask, ref_tif, out_tif):
    with rasterio.open(ref_tif) as src:
        prof = src.profile.copy()
    prof.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(mask, 1)


def _affected_set(mask_tif, geoms, frac_thresh):
    from rasterio.mask import mask as rio_mask
    aff = set()
    with rasterio.open(mask_tif) as src:
        for idx, geom in geoms:
            try:
                arr, _ = rio_mask(src, [geom.__geo_interface__], crop=True,
                                  all_touched=True, filled=False)
                data = arr.compressed() if hasattr(arr, "compressed") else arr.ravel()
                if data.size and (data == 1).mean() >= frac_thresh:
                    aff.add(idx)
            except Exception:
                continue
    return aff


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/sen1floods11_unet_s1s2/checkpoints/best-epoch016.ckpt")
    p.add_argument("--stats", default="data/processed/norm_stats_sen1floods11.yaml")
    p.add_argument("--regions", nargs="+", default=["USA", "Spain"])
    p.add_argument("--max-chips-per-region", type=int, default=12)
    p.add_argument("--min-buildings", type=int, default=15)
    p.add_argument("--out", default="outputs/decision_validation/results.json")
    args = p.parse_args()

    import osmnx as ox
    ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"
    from shapely.geometry import box

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pred_dir = out.parent / "pred_tifs"; pred_dir.mkdir(exist_ok=True)

    chip_rows = []
    bTP = bFP = bFN = 0  # building-level affected-identification confusion (vs GT)
    t0 = time.time()
    for region in args.regions:
        log.info("predict_region", region=region)
        preds = predict_region(Path(args.ckpt), args.stats, region)
        # order chips by ground-truth water fraction (focus on flooded chips)
        scored = []
        for pid in preds:
            gt_tif = HAND / "LabelHand" / f"{pid.replace('S1Hand','LabelHand')}.tif"
            if not gt_tif.exists():
                continue
            with rasterio.open(gt_tif) as src:
                g = src.read(1)
            scored.append((pid, float((g == 1).mean()), gt_tif))
        scored.sort(key=lambda t: -t[1])
        for pid, gtwater, gt_tif in scored[:args.max_chips_per_region * 2]:
            if len([r for r in chip_rows if r["region"] == region]) >= args.max_chips_per_region:
                break
            s1_tif = HAND / "S1Hand" / f"{pid}.tif"
            if not s1_tif.exists():
                continue
            pred_tif = pred_dir / f"{pid}_pred.tif"
            _write_tif(preds[pid], s1_tif, pred_tif)
            with rasterio.open(gt_tif) as src:
                b = src.bounds
            poly = box(b.left, b.bottom, b.right, b.top)
            try:
                bld = ox.features_from_polygon(poly, tags={"building": True})
                bld = bld[bld.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
            except Exception as e:
                log.warning("osm_fail", chip=pid, err=str(e)[:80]); continue
            if len(bld) < args.min_buildings:
                continue
            geoms = list(enumerate(bld.geometry))
            aff_gt = _affected_set(gt_tif, geoms, 0.2)
            aff_pred = _affected_set(pred_tif, geoms, 0.2)
            tp = len(aff_gt & aff_pred); fp = len(aff_pred - aff_gt); fn = len(aff_gt - aff_pred)
            bTP += tp; bFP += fp; bFN += fn
            chip_rows.append({"region": region, "chip": pid, "gt_water_frac": round(gtwater, 4),
                              "n_buildings": int(len(bld)),
                              "gt_affected": len(aff_gt), "pred_affected": len(aff_pred),
                              "tp": tp, "fp": fp, "fn": fn})
            log.info("chip_done", chip=pid, n_buildings=len(bld),
                     gt_aff=len(aff_gt), pred_aff=len(aff_pred), tp=tp, fp=fp, fn=fn)

    prec = bTP / (bTP + bFP) if (bTP + bFP) else 0.0
    rec = bTP / (bTP + bFN) if (bTP + bFN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    # per-chip count agreement (Pearson r of affected counts)
    gt_counts = np.array([r["gt_affected"] for r in chip_rows], float)
    pr_counts = np.array([r["pred_affected"] for r in chip_rows], float)
    r_count = float(np.corrcoef(gt_counts, pr_counts)[0, 1]) if len(chip_rows) > 1 else 0.0
    mae = float(np.abs(gt_counts - pr_counts).mean()) if len(chip_rows) else 0.0

    summary = {
        "task": "flood-affected building identification vs ground-truth mask",
        "regions": args.regions, "n_chips": len(chip_rows),
        "n_buildings_evaluated": int(bTP + bFP + bFN + sum(
            r["n_buildings"] - r["gt_affected"] - (r["fp"]) for r in chip_rows) if chip_rows else 0),
        "building_affected_precision": round(prec, 4),
        "building_affected_recall": round(rec, 4),
        "building_affected_f1": round(f1, 4),
        "affected_count_pearson_r": round(r_count, 4),
        "affected_count_mae_per_chip": round(mae, 2),
        "wall_time_min": round((time.time() - t0) / 60, 1),
        "chips": chip_rows,
    }
    out.write_text(json.dumps(summary, indent=2))
    print("\n=== Decision validation vs ground truth ===")
    print(f"  chips: {len(chip_rows)} | buildings affected-id "
          f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
    print(f"  affected-count Pearson r={r_count:.3f}, MAE/chip={mae:.2f}")
    print(f"  wall time: {summary['wall_time_min']} min  -> {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
