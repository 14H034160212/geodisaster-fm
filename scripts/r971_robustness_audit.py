"""Robustness audit of the r=0.971 flooded-area fidelity headline.

ChatGPT-review demanded: "除了 Pearson r, 要给 MAE/MAPE、Spearman、per-event
scatter、leave-one-event sensitivity、去掉大面积事件后的结果."

Inputs: outputs/decision/answer_fidelity.json — per-chip {pred_km2, gt_km2, region}
Outputs:
  - outputs/decision/r971_robustness.json
  - prints summary table to stdout
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats as sps

SRC = Path("outputs/decision/answer_fidelity.json")
OUT = Path("outputs/decision/r971_robustness.json")


def stats_for(pred, gt):
    pred = np.asarray(pred, float); gt = np.asarray(gt, float)
    n = len(pred)
    if n < 2: return {"n": n}
    pr = float(sps.pearsonr(pred, gt).statistic)
    sr = float(sps.spearmanr(pred, gt).statistic)
    mae = float(np.mean(np.abs(pred - gt)))
    rel_err = np.abs(pred - gt) / np.maximum(gt, 0.001)   # 0.001 km² floor avoids dry-chip blowup
    mape_med = float(np.median(rel_err))
    mape_mean = float(np.mean(rel_err))
    return {"n": n, "pearson_r": pr, "spearman_rho": sr,
            "mae_km2": mae, "mape_median": mape_med, "mape_mean": mape_mean}


def main():
    d = json.loads(SRC.read_text())
    per_chip = d["per_chip"]
    by_event = {}
    for c in per_chip:
        by_event.setdefault(c["region"], []).append(c)
    all_pred = [c["pred_km2"] for c in per_chip]
    all_gt   = [c["gt_km2"]   for c in per_chip]

    overall = stats_for(all_pred, all_gt)
    print(f"\n=== Overall (n={overall['n']} chips, {len(by_event)} events) ===")
    for k, v in overall.items():
        print(f"  {k:20s} {v}")

    # Per-event
    per_event = {}
    print(f"\n=== Per-event (sorted by gt-area sum) ===")
    events = sorted(by_event, key=lambda e: -sum(c["gt_km2"] for c in by_event[e]))
    print(f"{'event':<12} {'n':>4} {'Σgt_km2':>11} {'r':>7} {'ρ':>7} {'MAE_pix':>10} {'MAPE_med':>9}")
    for ev in events:
        chips = by_event[ev]
        pr_ev = [c["pred_km2"] for c in chips]
        gt_ev = [c["gt_km2"]   for c in chips]
        s = stats_for(pr_ev, gt_ev)
        per_event[ev] = s
        per_event[ev]["sum_gt_km2"] = float(sum(gt_ev))
        print(f"{ev:<12} {s['n']:>4} {per_event[ev]['sum_gt_km2']:>11.0f} "
              f"{s.get('pearson_r', float('nan')):>7.3f} {s.get('spearman_rho', float('nan')):>7.3f} "
              f"{s.get('mae_pixels', float('nan')):>10.1f} {s.get('mape_median', float('nan')):>9.3f}")

    # Leave-one-event-out: drop each event and see how r moves
    print(f"\n=== Leave-one-event-out: how does Pearson r change if we drop one event? ===")
    loeo = {}
    for ev in events:
        keep_pred = [c["pred_km2"] for c in per_chip if c["region"] != ev]
        keep_gt   = [c["gt_km2"]   for c in per_chip if c["region"] != ev]
        s = stats_for(keep_pred, keep_gt)
        loeo[ev] = s
        delta = s["pearson_r"] - overall["pearson_r"]
        print(f"  drop {ev:<12} → r={s['pearson_r']:.4f}  Δ={delta:+.4f}  (n={s['n']})")

    # Big-event domination check: re-compute on bottom-50% by area
    sorted_chips_by_area = sorted(per_chip, key=lambda c: c["gt_km2"])
    half = len(sorted_chips_by_area) // 2
    small = sorted_chips_by_area[:half]
    big = sorted_chips_by_area[half:]
    small_stats = stats_for([c["pred_km2"] for c in small], [c["gt_km2"] for c in small])
    big_stats = stats_for([c["pred_km2"] for c in big],     [c["gt_km2"] for c in big])
    print(f"\n=== Big-vs-small chip stratification ===")
    print(f"  bottom-50% area chips (n={small_stats['n']}): r={small_stats['pearson_r']:.4f}  ρ={small_stats['spearman_rho']:.4f}")
    print(f"  top-50%    area chips (n={big_stats['n']}): r={big_stats['pearson_r']:.4f}  ρ={big_stats['spearman_rho']:.4f}")
    print(f"  → big-chip domination: {'YES' if (big_stats['pearson_r'] - small_stats['pearson_r']) > 0.10 else 'NO'} "
          f"(Δr={big_stats['pearson_r']-small_stats['pearson_r']:+.3f})")

    out = {
        "overall": overall, "per_event": per_event, "leave_one_event_out": loeo,
        "bottom50_chips": small_stats, "top50_chips": big_stats,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
