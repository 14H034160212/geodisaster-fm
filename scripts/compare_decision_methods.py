"""Stage 2 (fast, offline): compare the Structured Decision Inference (SDI)
method against three baselines on DECISION-level accuracy (affected-building
identification F1 vs ground truth), using the cached per-building features.

Methods:
  B1 raw-threshold     : footprint >=20% wet under the 0.5 mask (current reasoner)
  B2 any-intersection  : footprint touches any flood pixel (naive over-predict)
  B3 prob-threshold*   : mean prob >= tuned threshold (calibrated-prob ablation)
  SDI (ours)           : MRF over the building graph (evidence + spatial smoothness)

Hyperparameters of B3 (threshold) and SDI (lambda) are tuned on a chip-level
TUNE split and all methods are reported on the held-out TEST split, so the win
is not from hyperparameter overfitting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from geodisaster.dispatch.structured_decision import (
    SDIConfig, infer_affected, baseline_raw_threshold, baseline_any_intersection,
    baseline_prob_threshold, prf,
)


def load_features(paths):
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        rows.extend(d["features"])
    return rows


def by_chip(rows):
    chips = {}
    for r in rows:
        chips.setdefault((r["region"], r["chip"]), []).append(r)
    return chips


def sdi_predict(chips, cfg):
    """Run SDI per chip; return concatenated pred + gt aligned to a flat order."""
    preds, gts = [], []
    for key, rs in chips.items():
        prob = np.array([r["mean_prob"] for r in rs])
        cent = np.array([[r["cx"], r["cy"]] for r in rs])
        gt = np.array([r["gt_affected"] for r in rs], bool)
        x = infer_affected(prob, cent, cfg)
        preds.append(x); gts.append(gt)
    return np.concatenate(preds), np.concatenate(gts)


def flat(chips, key):
    return np.concatenate([np.array([r[key] for r in rs]) for rs in chips.values()])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--out", default="outputs/decision/method_comparison.json")
    p.add_argument("--fig", default="outputs/figures/fig14_decision_methods.png")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = load_features(args.features)
    chips = by_chip(rows)
    chip_keys = sorted(chips)
    rng = np.random.RandomState(args.seed); rng.shuffle(chip_keys)
    half = max(1, len(chip_keys) // 2)
    tune_keys, test_keys = chip_keys[:half], chip_keys[half:]
    tune = {k: chips[k] for k in tune_keys}
    test = {k: chips[k] for k in test_keys}
    print(f"{len(rows)} buildings over {len(chip_keys)} chips "
          f"| tune {len(tune_keys)} / test {len(test_keys)} chips")

    # ---- tune B3 threshold + SDI lambda on TUNE split ----
    def f1_b3(thr, ch):
        return prf(baseline_prob_threshold(flat(ch, "mean_prob"), thr), flat(ch, "gt_affected"))["f1"]
    b3_grid = np.linspace(0.1, 0.9, 17)
    b3_thr = max(b3_grid, key=lambda t: f1_b3(t, tune))

    lam_grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    def f1_sdi(lam, ch):
        pr, gt = sdi_predict(ch, SDIConfig(lambda_smooth=lam))
        return prf(pr, gt)["f1"]
    sdi_lam = max(lam_grid, key=lambda l: f1_sdi(l, tune))
    print(f"tuned: B3 thr={b3_thr:.3f} | SDI lambda={sdi_lam}")

    # ---- evaluate all methods on TEST split ----
    gt = flat(test, "gt_affected")
    ffh = flat(test, "flood_frac_hard")
    mp = flat(test, "mean_prob")
    results = {
        "B1_raw_threshold":    prf(baseline_raw_threshold(ffh), gt),
        "B2_any_intersection": prf(baseline_any_intersection(ffh), gt),
        "B3_prob_threshold":   prf(baseline_prob_threshold(mp, b3_thr), gt),
    }
    sdi_pred, sdi_gt = sdi_predict(test, SDIConfig(lambda_smooth=sdi_lam))
    results["SDI_ours"] = prf(sdi_pred, sdi_gt)
    # ablation: SDI with lambda=0 (no structure) == just prob>0.5 evidence
    sdi0_pred, _ = sdi_predict(test, SDIConfig(lambda_smooth=0.0))
    results["SDI_no_structure(lambda0)"] = prf(sdi0_pred, gt)

    summary = {
        "n_buildings_total": len(rows), "n_chips": len(chip_keys),
        "test_chips": len(test_keys), "test_buildings": int(len(gt)),
        "tuned_b3_threshold": float(b3_thr), "tuned_sdi_lambda": float(sdi_lam),
        "methods": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print("\n=== Decision-level affected-building identification (TEST split) ===")
    order = ["B2_any_intersection", "B1_raw_threshold", "SDI_no_structure(lambda0)",
             "B3_prob_threshold", "SDI_ours"]
    for k in order:
        m = results[k]
        print(f"  {k:28s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    best_base = max(results[k]["f1"] for k in results if k != "SDI_ours")
    print(f"\n  SDI F1={results['SDI_ours']['f1']:.3f}  "
          f"(best baseline F1={best_base:.3f}, "
          f"gain {results['SDI_ours']['f1'] - best_base:+.3f})")

    # ---- figure ----
    try:
        import matplotlib.pyplot as plt
        labels = ["B2 any-\nintersect", "B1 raw-\nthreshold", "SDI λ=0\n(no struct)",
                  "B3 prob-\nthreshold*", "SDI (ours)"]
        f1s = [results[k]["f1"] for k in order]
        colors = ["#bbbbbb", "#d62728", "#ff7f0e", "#9467bd", "#1f77b4"]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(range(len(order)), f1s, color=colors, edgecolor="black", linewidth=0.6)
        for i, v in enumerate(f1s):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Affected-building F1 vs ground truth", fontsize=11)
        ax.set_title("Decision-level accuracy: Structured Decision Inference vs baselines\n"
                     f"(held-out test: {len(test_keys)} chips, {int(len(gt))} buildings)",
                     fontsize=11.5, loc="left")
        ax.grid(True, alpha=0.3, axis="y"); ax.set_ylim(0, max(f1s) * 1.18)
        Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(args.fig, dpi=200, bbox_inches="tight"); plt.close()
        print(f"  figure -> {args.fig}")
    except Exception as e:
        print("  (figure skipped:", e, ")")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
