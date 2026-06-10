"""Per-fire-year calibration headroom on HLS-Burn-Scars (cross-hazard H2 test).

For each of the 4 LOEO checkpoints (held-out year), forward every chip from
that year and measure F1 at the default 0.5 threshold vs F1 at the
year-optimal threshold, plus ECE. Mirrors scripts/calibration_analysis.py
but reads HLS TIFFs (6-band HLS S30) instead of Sen1Floods11 chip arrays.

Output: outputs/decision/calibration_analysis_burnscars.json
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
import torch
import segmentation_models_pytorch as smp
import rasterio
sys.path.insert(0, ".")
from geodisaster.metrics import expected_calibration_error
from geodisaster.utils.logging import get_logger, setup_logging

log = get_logger("calib-burn")

ALL_YEARS = [2018, 2019, 2020, 2021]
RAW = Path("data/raw/hls_burn_scars")
CKPT_ROOT = Path("outputs/leave_one_event_out_burnscars")
OUT = Path("outputs/decision/calibration_analysis_burnscars.json")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YEAR_RE = re.compile(r"\.(\d{4})\d{3}\.v1\.4_merged\.tif$")


def chips_for_year(year: int) -> list[Path]:
    out = []
    for split in ("training", "validation"):
        for p in sorted((RAW / split).glob("*_merged.tif")):
            m = YEAR_RE.search(p.name)
            if m and int(m.group(1)) == year:
                out.append(p)
    return out


def load_model(ckpt_path: Path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=state.get("in_channels", 6), classes=1)
    model.load_state_dict(state["model"], strict=True)
    return model.eval().to(DEV)


@torch.no_grad()
def forward_chip(model, tif_path: Path):
    """Return (probs, labels) flattened over valid pixels for one chip."""
    with rasterio.open(tif_path) as src:
        img = src.read().astype(np.float32)
    img = (img - img.mean(axis=(1, 2), keepdims=True)) / \
          (img.std(axis=(1, 2), keepdims=True) + 1e-3)
    mask_p = tif_path.with_name(tif_path.name.replace("_merged.tif", ".mask.tif"))
    with rasterio.open(mask_p) as src:
        m = src.read(1).astype(np.int64)
    x = torch.from_numpy(img).unsqueeze(0).to(DEV)
    pr = torch.sigmoid(model(x).squeeze(1)).cpu().numpy().reshape(-1).astype(np.float32)
    lb = m.reshape(-1)
    valid = lb >= 0
    return pr[valid], lb[valid].astype(np.uint8)


def f1_at(pr, lb, t):
    pred = pr >= t; pos = lb == 1
    tp = float(np.sum(pred & pos)); fp = float(np.sum(pred & ~pos))
    fn = float(np.sum(~pred & pos))
    d = 2*tp + fp + fn
    return float(2*tp / d) if d > 0 else 0.0


def main():
    setup_logging()
    grid = np.linspace(0.05, 0.95, 19)
    per_year = {}
    for year in ALL_YEARS:
        ck = CKPT_ROOT / f"test_{year}" / "checkpoints" / "best.ckpt"
        if not ck.exists():
            log.warning("missing_ckpt", year=year); continue
        log.info("loading", year=year, ckpt=ck.name)
        model = load_model(ck)
        chips = chips_for_year(year)
        P, L = [], []
        for p in chips:
            pr, lb = forward_chip(model, p)
            P.append(pr); L.append(lb)
        probs = np.concatenate(P); labels = np.concatenate(L)
        f1_05 = f1_at(probs, labels, 0.5)
        f1s = [f1_at(probs, labels, t) for t in grid]
        bi = int(np.argmax(f1s)); best_t = float(grid[bi]); f1_best = float(f1s[bi])
        ece = float(expected_calibration_error(torch.tensor(probs), torch.tensor(labels)))
        per_year[str(year)] = {
            "n_chips": len(chips),
            "f1_at_0.5": round(f1_05, 4),
            "f1_at_best": round(f1_best, 4),
            "best_threshold": best_t,
            "calib_gain": round(f1_best - f1_05, 4),
            "ece": round(ece, 4),
        }
        log.info("done_year", year=year, f1_05=round(f1_05, 3),
                 f1_best=round(f1_best, 3), best_t=best_t,
                 gain=round(f1_best - f1_05, 3))
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    gains = [v["calib_gain"] for v in per_year.values()]
    thrs = [v["best_threshold"] for v in per_year.values()]
    summary = {
        "n_events": len(per_year), "events": list(per_year.keys()),
        "mean_calib_gain": round(float(np.mean(gains)), 4),
        "max_calib_gain": round(float(np.max(gains)), 4),
        "best_threshold_range": [min(thrs), max(thrs)],
        "frac_events_optimal_not_0.5": round(
            float(np.mean([abs(t - 0.5) > 0.001 for t in thrs])), 3),
        "per_event": per_year,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n=== HLS Burn-Scars calibration headroom (4 LOEO fire-season events) ===")
    print(f"{'year':<6} {'F1@0.5':>8} {'F1@best':>9} {'τ*':>6} {'gain':>8} {'ECE':>7} {'n_chips':>8}")
    for y, v in per_year.items():
        print(f"{y:<6} {v['f1_at_0.5']:>8.3f} {v['f1_at_best']:>9.3f} "
              f"{v['best_threshold']:>6.2f} {v['calib_gain']:>+8.3f} "
              f"{v['ece']:>7.3f} {v['n_chips']:>8}")
    print(f"\nmean gain {summary['mean_calib_gain']:+.4f}  "
          f"threshold range {summary['best_threshold_range']}  "
          f"frac τ*≠0.5 = {summary['frac_events_optimal_not_0.5']}")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    sys.exit(main() or 0)
