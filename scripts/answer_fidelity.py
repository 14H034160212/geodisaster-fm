"""P0 / Result-1: do the dispatcher's decision-relevant ANSWERS match the
answers a responder would derive from ground truth, on real flood events?

The most basic, OSM-free decision answer is the flooded EXTENT. For each of the
10 real Sen1Floods11 flood events we deploy that region's own leave-one-region-
out model (genuine unseen-event setting) and compare the predicted flooded area
(km^2) against the analyst hand-label area, per chip and per event. We also time
the perception stage to anchor the end-to-end 'time-to-answer' claim.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, ".")
from geodisaster.data.tile import merge_manifests
from geodisaster.datasets import DisasterPatchDataModule, stats_with_fallbacks
from geodisaster.train import DisasterSegLightningModule
from geodisaster.utils.logging import get_logger, setup_logging

log = get_logger("answer_fidelity")
LOO = Path("outputs/leave_one_region_out")
PIX_KM2 = (10.0 * 10.0) / 1e6   # 10 m pixel -> km^2
REGIONS = ["Ghana", "India", "Mekong", "Nigeria", "Pakistan",
           "Paraguay", "Somalia", "Spain", "Sri-Lanka", "USA"]


def _latest_ckpt(region):
    cks = sorted((LOO / f"test_{region}" / "checkpoints").glob("*.ckpt"))
    return cks[-1] if cks else None


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--stats", default="data/processed/norm_stats_sen1floods11.yaml")
    p.add_argument("--patch-root", default="data/processed/patches")
    p.add_argument("--out", default="outputs/decision/answer_fidelity.json")
    args = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_chip = []         # (region, area_gt_km2, area_pred_km2)
    per_region = {}
    perc_times = []
    for region in REGIONS:
        ck = _latest_ckpt(region)
        if ck is None:
            log.warning("no_ckpt", region=region); continue
        state = torch.load(ck, map_location="cpu", weights_only=False)
        hp = state["hyper_parameters"]
        mcfg = OmegaConf.create(hp.get("model_cfg", hp.get("model")))
        tcfg = OmegaConf.create(hp.get("train_cfg", hp.get("train")))
        sources = hp["sources"]
        m = DisasterSegLightningModule(mcfg, tcfg, sources)
        m.load_state_dict(state["state_dict"], strict=True); m.eval().to(dev)
        patches = merge_manifests([Path(args.patch_root) / f"sen1floods11_{region}"])
        norm = stats_with_fallbacks(args.stats, sources)
        dm = DisasterPatchDataModule(train_patches=[], val_patches=[], test_patches=patches,
                                     sources=sources, batch_size=8, num_workers=4, normalize=norm)
        t0 = time.time(); n_chips = 0
        gt_tot = pred_tot = 0.0
        rows = []
        with torch.no_grad():
            for b in dm.test_dataloader():
                for k, v in list(b.items()):
                    if isinstance(v, torch.Tensor):
                        b[k] = v.to(dev)
                pred = (torch.sigmoid(m(b).squeeze(1)) > 0.5).cpu().numpy()
                mask = b["mask"].cpu().numpy()
                for i in range(pred.shape[0]):
                    valid = mask[i] != 255
                    a_gt = float((mask[i] == 1).sum()) * PIX_KM2
                    a_pr = float((pred[i] & valid).sum()) * PIX_KM2
                    rows.append((a_gt, a_pr)); gt_tot += a_gt; pred_tot += a_pr
                    n_chips += 1
        dt = time.time() - t0
        perc_times.append(dt / max(n_chips, 1))
        for a_gt, a_pr in rows:
            per_chip.append((region, a_gt, a_pr))
        rel = abs(pred_tot - gt_tot) / max(gt_tot, 1e-6)
        per_region[region] = {"n_chips": n_chips, "gt_area_km2": round(gt_tot, 3),
                              "pred_area_km2": round(pred_tot, 3),
                              "rel_area_error": round(rel, 3),
                              "perception_s_per_chip": round(dt / max(n_chips, 1), 3)}
        log.info("region", region=region, n=n_chips, gt_km2=round(gt_tot, 1),
                 pred_km2=round(pred_tot, 1), rel_err=round(rel, 3))

    gt = np.array([c[1] for c in per_chip]); pr = np.array([c[2] for c in per_chip])
    r = float(np.corrcoef(gt, pr)[0, 1]) if len(gt) > 1 else 0.0
    # region-total relative error (mean over regions)
    mean_rel = float(np.mean([v["rel_area_error"] for v in per_region.values()]))
    summary = {
        "n_events": len(per_region), "n_chips": len(per_chip),
        "flooded_area_pearson_r": round(r, 4),
        "mean_region_rel_area_error": round(mean_rel, 4),
        "perception_s_per_chip_mean": round(float(np.mean(perc_times)), 3),
        "device": str(dev), "per_region": per_region,
        "per_chip": [{"region": c[0], "gt_km2": round(c[1], 4), "pred_km2": round(c[2], 4)}
                     for c in per_chip],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\n=== Flooded-area answer fidelity (10 real held-out flood events) ===")
    print(f"  chips={len(per_chip)}  events={len(per_region)}")
    print(f"  per-chip flooded-area Pearson r = {r:.4f}")
    print(f"  mean per-event relative area error = {mean_rel:.3f}")
    print(f"  perception: {summary['perception_s_per_chip_mean']:.3f} s/chip on {dev}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
