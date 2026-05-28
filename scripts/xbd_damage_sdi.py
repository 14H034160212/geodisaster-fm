"""Headline experiment: does Structured Decision Inference (SDI) beat baselines
at the DECISION level — 'which buildings are damaged' — on xBD?

This is the building-damage analogue of Xu et al. (Nat. Commun. 2022): per-
building damage from satellite imagery, but we ask whether a structured MRF over
the building graph (spatial smoothness of damage state) improves the DECISION
answer over per-building thresholding of a damage model's evidence.

Pipeline (in-domain, image-level train/test split):
  1. train a post-image damage segmentation model (P(class>=3: major/destroyed))
     on the train images' 512 tiles (reuses the GeoDisaster-FM training stack);
  2. predict on test tiles, stitch to the full 1024 image probability map;
  3. for each test building (xBD JSON polygon + subtype) compute mean predicted
     damage prob, footprint flood/damage fraction, centroid, and GT-damaged;
  4. run SDI vs the 3 baselines and report affected-building F1 vs GT.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, ".")
from geodisaster.data.tile import merge_manifests
from geodisaster.datasets import DisasterPatchDataModule, stats_with_fallbacks
from geodisaster.train import DisasterSegLightningModule, make_trainer
from geodisaster.dispatch.structured_decision import (
    SDIConfig, infer_affected, baseline_raw_threshold, baseline_any_intersection,
    baseline_prob_threshold, prf,
)
from geodisaster.utils.io import load_config, ensure_dir
from geodisaster.utils.logging import get_logger, setup_logging

log = get_logger("xbd_dmg_sdi")
ROOT = Path("data/processed/patches")
LABELS = Path("data/raw/xbd/train/labels")
DAMAGED_SUBTYPES = {"major-damage", "destroyed"}


def image_of(pid):
    # hurricane-harvey_00000123_post_disaster_r0_c512 -> hurricane-harvey_00000123_post_disaster
    return pid.split("_r")[0]


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--disasters", nargs="+", default=["hurricane-harvey", "palu-tsunami"])
    p.add_argument("--stats", default="data/processed/norm_stats_xbd.yaml")
    p.add_argument("--model-config", default="configs/model/unet_xbd.yaml")
    p.add_argument("--default-config", default="configs/default.yaml")
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulate", type=int, default=2)
    p.add_argument("--workdir", default="outputs/xbd_damage_sdi")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import pytorch_lightning as pl
    pl.seed_everything(args.seed)
    workdir = ensure_dir(args.workdir)

    # ---- gather patches, split by IMAGE ----
    patches = merge_manifests([ROOT / f"xbddmg_{d}" for d in args.disasters])
    by_img = defaultdict(list)
    for rec in patches:
        by_img[image_of(rec["patch_id"])].append(rec)
    imgs = sorted(by_img)
    rng = np.random.RandomState(args.seed); rng.shuffle(imgs)
    n_test = max(1, int(len(imgs) * args.test_frac))
    test_imgs, train_imgs = imgs[:n_test], imgs[n_test:]
    train_p = [r for im in train_imgs for r in by_img[im]]
    test_p = [r for im in test_imgs for r in by_img[im]]
    val_p = train_p[::8]  # small val slice
    log.info("split", train_imgs=len(train_imgs), test_imgs=len(test_imgs),
             train_tiles=len(train_p), test_tiles=len(test_p))

    # ---- train damage model ----
    defaults = load_config(args.default_config)
    defaults.train.batch_size = args.batch_size
    defaults.train.accumulate_grad_batches = args.accumulate
    model_cfg = load_config(args.model_config)
    sources = ["optical"]
    norm = stats_with_fallbacks(args.stats, sources)
    dm = DisasterPatchDataModule(train_patches=train_p, val_patches=val_p, test_patches=test_p,
                                 sources=sources, batch_size=args.batch_size,
                                 num_workers=int(defaults.train.num_workers), normalize=norm)
    module = DisasterSegLightningModule(model_cfg=model_cfg, train_cfg=defaults.train, sources=sources)
    trainer = make_trainer(defaults.train, workdir=workdir)
    trainer.fit(module, datamodule=dm)

    # ---- predict on test tiles, stitch to 1024 prob per image ----
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.eval().to(dev)
    prob1024 = {im: np.zeros((1024, 1024), np.float32) for im in test_imgs}
    test_dm = DisasterPatchDataModule(train_patches=[], val_patches=[], test_patches=test_p,
                                      sources=sources, batch_size=args.batch_size,
                                      num_workers=4, normalize=norm)
    pid2rc = {r["patch_id"]: (r.get("row", 0), r.get("col", 0)) for r in test_p}
    with torch.no_grad():
        for b in test_dm.test_dataloader():
            for k, v in list(b.items()):
                if isinstance(v, torch.Tensor):
                    b[k] = v.to(dev)
            sc = torch.sigmoid(module(b).squeeze(1)).cpu().numpy()
            for i, pid in enumerate(b["patch_id"]):
                im = image_of(pid); r, c = pid2rc[pid]
                prob1024[im][r:r + 512, c:c + 512] = sc[i]

    # ---- per-building features from JSON ----
    from rasterio.features import rasterize
    from shapely import wkt as shapely_wkt
    feats = []
    for im in test_imgs:
        jf = LABELS / f"{im}.json"
        if not jf.exists():
            continue
        blds = json.loads(jf.read_text()).get("features", {}).get("xy", [])
        if not blds:
            continue
        polys, subs = [], []
        for b in blds:
            try:
                polys.append(shapely_wkt.loads(b["wkt"]))
                subs.append(b.get("properties", {}).get("subtype", "no-damage"))
            except Exception:
                continue
        if not polys:
            continue
        pr = prob1024[im]
        idx_raster = rasterize([(g, i + 1) for i, g in enumerate(polys)],
                               out_shape=(1024, 1024), fill=0, dtype="int32")
        for i, (g, sub) in enumerate(zip(polys, subs)):
            m = idx_raster == (i + 1)
            if not m.any():
                continue
            vals = pr[m]
            feats.append({"region": image_of(im).rsplit("_", 2)[0], "chip": im, "bld": i,
                          "mean_prob": float(vals.mean()),
                          "flood_frac_hard": float((vals > 0.5).mean()),
                          "cx": float(g.centroid.x), "cy": float(g.centroid.y),
                          "gt_affected": bool(sub in DAMAGED_SUBTYPES)})
    n_pos = sum(f["gt_affected"] for f in feats)
    log.info("features", n_buildings=len(feats), n_damaged=n_pos, test_images=len(test_imgs))
    (Path(workdir) / "features.json").write_text(json.dumps(
        {"n_buildings": len(feats), "n_damaged": n_pos, "features": feats}, indent=2))

    # ---- SDI vs baselines (tune lambda / thr on first half of test images) ----
    by_chip = defaultdict(list)
    for f in feats:
        by_chip[f["chip"]].append(f)
    chips = sorted(by_chip); rng.shuffle(chips)
    half = max(1, len(chips) // 2)
    tune = {k: by_chip[k] for k in chips[:half]}
    test = {k: by_chip[k] for k in chips[half:]}

    def flat(ch, key):
        return np.concatenate([np.array([r[key] for r in rs]) for rs in ch.values()]) if ch else np.array([])

    def sdi_pred(ch, lam):
        preds, gts = [], []
        for rs in ch.values():
            prob = np.array([r["mean_prob"] for r in rs])
            cent = np.array([[r["cx"], r["cy"]] for r in rs])
            preds.append(infer_affected(prob, cent, SDIConfig(radius_m=60, lambda_smooth=lam,
                                                              sigma_m=40)))
            gts.append(np.array([r["gt_affected"] for r in rs], bool))
        return np.concatenate(preds), np.concatenate(gts)

    b3_grid = np.linspace(0.1, 0.9, 17)
    b3_thr = max(b3_grid, key=lambda t: prf(baseline_prob_threshold(flat(tune, "mean_prob"), t),
                                            flat(tune, "gt_affected"))["f1"])
    lam_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    sdi_lam = max(lam_grid, key=lambda l: prf(*sdi_pred(tune, l))["f1"])

    gt = flat(test, "gt_affected"); ffh = flat(test, "flood_frac_hard"); mp = flat(test, "mean_prob")
    res = {
        "B1_raw_threshold": prf(baseline_raw_threshold(ffh), gt),
        "B2_any_intersection": prf(baseline_any_intersection(ffh), gt),
        "B3_prob_threshold": prf(baseline_prob_threshold(mp, b3_thr), gt),
        "SDI_ours": prf(*sdi_pred(test, sdi_lam)),
        "SDI_no_structure(lambda0)": prf(*sdi_pred(test, 0.0)),
    }
    summary = {"disasters": args.disasters, "n_buildings": len(feats), "n_damaged": n_pos,
               "test_buildings": int(len(gt)), "tuned_b3_thr": float(b3_thr),
               "tuned_sdi_lambda": float(sdi_lam), "methods": res}
    (Path(workdir) / "sdi_results.json").write_text(json.dumps(summary, indent=2))

    print("\n=== xBD building-damage decision (SDI vs baselines, TEST split) ===")
    for k in ["B2_any_intersection", "B1_raw_threshold", "SDI_no_structure(lambda0)",
              "B3_prob_threshold", "SDI_ours"]:
        m = res[k]; print(f"  {k:28s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    best_base = max(res[k]["f1"] for k in res if not k.startswith("SDI_ours"))
    print(f"\n  SDI F1={res['SDI_ours']['f1']:.3f} (best baseline {best_base:.3f}, "
          f"gain {res['SDI_ours']['f1'] - best_base:+.3f}) | test buildings={len(gt)}, damaged={int(gt.sum())}")
    print(f"Saved {workdir}/sdi_results.json")


if __name__ == "__main__":
    sys.exit(main() or 0)
