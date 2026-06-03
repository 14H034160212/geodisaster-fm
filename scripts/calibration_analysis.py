"""Quantify the 'calibration is the lever' finding across all 10 real flood
events. For each region's leave-one-region-out model we measure F1 at the
default 0.5 threshold vs F1 at the region-optimal threshold, the optimal
threshold itself (it is NOT 0.5 and varies by region), and the expected
calibration error. This is the headroom the RL calibration policy targets.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
sys.path.insert(0, ".")
from geodisaster.data.tile import merge_manifests
from geodisaster.datasets import DisasterPatchDataModule, stats_with_fallbacks
from geodisaster.train import DisasterSegLightningModule
from geodisaster.metrics import expected_calibration_error
from geodisaster.utils.logging import get_logger, setup_logging
log = get_logger("calib")
LOO = Path("outputs/leave_one_region_out")   # default; override with --ckpt-root
REGIONS = ["Ghana", "India", "Mekong", "Nigeria", "Pakistan", "Paraguay",
           "Somalia", "Spain", "Sri-Lanka", "USA"]


def f1_at(probs, labels, t):
    pred = probs >= t; pos = labels == 1
    tp = np.sum(pred & pos); fp = np.sum(pred & ~pos); fn = np.sum(~pred & pos)
    d = 2 * tp + fp + fn
    return float(2 * tp / d) if d > 0 else 0.0


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="data/processed/norm_stats_sen1floods11.yaml")
    ap.add_argument("--out", default="outputs/decision/calibration_analysis.json")
    ap.add_argument("--ckpt-root", default="outputs/leave_one_region_out",
                    help="LOO dir (override for AlphaEarth: outputs/leave_one_region_out_ae)")
    ap.add_argument("--regions", nargs="+", default=None,
                    help="default = REGIONS; pass e.g. Pakistan Somalia Paraguay India for AE LOO")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = np.linspace(0.05, 0.95, 19)
    per_region = {}
    loo_root = Path(args.ckpt_root)
    regions = args.regions if args.regions else REGIONS
    for region in regions:
        cks = sorted((loo_root / f"test_{region}" / "checkpoints").glob("*.ckpt"))
        if not cks:
            continue
        st = torch.load(cks[-1], map_location="cpu", weights_only=False)
        hp = st["hyper_parameters"]
        m = DisasterSegLightningModule(OmegaConf.create(hp.get("model_cfg", hp.get("model"))),
                                       OmegaConf.create(hp.get("train_cfg", hp.get("train"))), hp["sources"])
        m.load_state_dict(st["state_dict"], strict=True); m.eval().to(dev)
        patches = merge_manifests([Path("data/processed/patches") / f"sen1floods11_{region}"])
        norm = stats_with_fallbacks(args.stats, hp["sources"])
        dm = DisasterPatchDataModule(train_patches=[], val_patches=[], test_patches=patches,
                                     sources=hp["sources"], batch_size=8, num_workers=4, normalize=norm)
        P, L = [], []
        with torch.no_grad():
            for b in dm.test_dataloader():
                for k, v in list(b.items()):
                    if isinstance(v, torch.Tensor): b[k] = v.to(dev)
                pr = torch.sigmoid(m(b).squeeze(1)).cpu().numpy().reshape(-1)
                lb = b["mask"].cpu().numpy().reshape(-1)
                ok = lb != 255; P.append(pr[ok]); L.append(lb[ok])
        probs = np.concatenate(P); labels = np.concatenate(L)
        f1_05 = f1_at(probs, labels, 0.5)
        f1s = [f1_at(probs, labels, t) for t in grid]
        bi = int(np.argmax(f1s)); best_t = float(grid[bi]); f1_best = float(f1s[bi])
        ece = float(expected_calibration_error(torch.tensor(probs), torch.tensor(labels)))
        per_region[region] = {"f1_at_0.5": round(f1_05, 4), "f1_at_best": round(f1_best, 4),
                              "best_threshold": best_t, "calib_gain": round(f1_best - f1_05, 4),
                              "ece": round(ece, 4)}
        log.info("region", region=region, f1_05=round(f1_05, 3), f1_best=round(f1_best, 3),
                 best_t=best_t, gain=round(f1_best - f1_05, 3))
    gains = [v["calib_gain"] for v in per_region.values()]
    thrs = [v["best_threshold"] for v in per_region.values()]
    summary = {"n_events": len(per_region),
               "mean_calib_gain": round(float(np.mean(gains)), 4),
               "max_calib_gain": round(float(np.max(gains)), 4),
               "best_threshold_range": [min(thrs), max(thrs)],
               "frac_events_optimal_not_0.5": round(float(np.mean([t != 0.5 for t in thrs])), 3),
               "per_region": per_region}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\n=== Calibration headroom across 10 real flood events ===")
    for r, v in per_region.items():
        print(f"  {r:10s} F1@0.5={v['f1_at_0.5']:.3f}  F1@best={v['f1_at_best']:.3f}  "
              f"best_t={v['best_threshold']:.2f}  gain={v['calib_gain']:+.3f}  ECE={v['ece']:.3f}")
    print(f"\n  mean calibration gain = {summary['mean_calib_gain']:+.3f} F1 | "
          f"optimal thresholds span {summary['best_threshold_range']}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
