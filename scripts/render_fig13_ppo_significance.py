"""Fig 13 — Layer 3 PPO multi-seed significance.

Replaces the single-split bar chart with an honest multi-seed view:
  (a) per-method aggregate test F1, mean +/- 95% CI over seeds.
  (b) paired differences (PPO - baseline) with 95% CI + paired-test p-values
      (a forest plot: does the interval clear zero?).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(results_json: str | Path, out_path: str | Path) -> Path:
    d = json.loads(Path(results_json).read_text())
    agg = d["aggregate"]
    paired = d["paired"]
    n_seeds = d["seeds"]; budget = d["budget"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # (a) per-method mean +/- 95% CI
    methods = [("base", "zero-shot\n(0.5 thr)", "#bbbbbb"),
               ("random", "random\ncalib", "#d62728"),
               ("uncertainty", "uncertainty\ncalib", "#ff7f0e"),
               ("coreset", "coreset\ncalib", "#8c564b"),
               ("ppo", "PPO\npolicy", "#1f77b4"),
               ("full_pool", "full-pool\noracle", "#2ca02c")]
    methods = [m for m in methods if m[0] in agg]   # tolerate missing keys
    xs = np.arange(len(methods))
    means = [agg[m]["mean"] for m, _, _ in methods]
    lo = [agg[m]["mean"] - agg[m]["ci95"][0] for m, _, _ in methods]
    hi = [agg[m]["ci95"][1] - agg[m]["mean"] for m, _, _ in methods]
    colors = [c for _, _, c in methods]
    ax1.bar(xs, means, color=colors, yerr=[lo, hi], capsize=5,
            edgecolor="black", linewidth=0.6, error_kw={"elinewidth": 1.4})
    ax1.set_xticks(xs); ax1.set_xticklabels([lab for _, lab, _ in methods], fontsize=9.5)
    ax1.set_ylabel("Test F1 after threshold calibration", fontsize=11)
    ax1.set_title(f"(a) Mean test F1 over {n_seeds} seeds (±95% CI)",
                  fontsize=12, loc="left", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(min(means) - 0.06, max([agg[m]['ci95'][1] for m, _, _ in methods]) + 0.03)
    for x, m in zip(xs, means):
        ax1.text(x, m + 0.004, f"{m:.3f}", ha="center", fontsize=8.5)

    # (b) paired-difference forest plot
    rows = [("ppo_vs_zeroshot", "PPO − zero-shot"),
            ("ppo_vs_random", "PPO − random"),
            ("ppo_vs_uncertainty", "PPO − uncertainty"),
            ("ppo_vs_coreset", "PPO − coreset")]
    rows = [r for r in rows if r[0] in paired]   # tolerate missing
    ys = np.arange(len(rows))[::-1]
    for y, (key, lab) in zip(ys, rows):
        pr = paired[key]
        m = pr["mean"]; ci = pr["ci95"]; p = pr["t_p"]
        sig = ci[0] > 0 or ci[1] < 0
        col = "#1c7f4f" if sig else "#a86a1f"
        ax2.errorbar([m], [y], xerr=[[m - ci[0]], [ci[1] - m]], fmt="o",
                     color=col, ms=10, capsize=6, elinewidth=2)
        ax2.text(ci[1] + 0.002, y, f"{m:+.3f}  (p={p:.3f})", va="center",
                 fontsize=9.5, color=col)
    ax2.axvline(0, ls="--", color="#888", lw=1.3)
    ax2.set_yticks(ys); ax2.set_yticklabels([lab for _, lab in rows], fontsize=10.5)
    ax2.set_xlabel("Paired F1 difference (mean ± 95% CI over seeds)", fontsize=11)
    ax2.set_title("(b) Is the PPO advantage significant?", fontsize=12,
                  loc="left", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.set_ylim(-0.6, len(rows) - 0.4)
    xmax = max(paired[k]["ci95"][1] for k, _ in rows)
    xmin = min(paired[k]["ci95"][0] for k, _ in rows)
    pad = 0.4 * (xmax - xmin) + 0.01
    ax2.set_xlim(min(xmin, 0) - 0.01, xmax + pad)

    pr_r = paired["ppo_vs_random"]
    verdict = ("significant" if pr_r["ci95"][0] > 0 else "not significant")
    fig.suptitle(
        f"Layer 3 PPO policy — multi-seed significance ({n_seeds} seeds, "
        f"{budget}-chip budget). PPO vs random: {pr_r['mean']:+.3f} F1, "
        f"95% CI [{pr_r['ci95'][0]:+.3f}, {pr_r['ci95'][1]:+.3f}], "
        f"paired t p={pr_r['t_p']:.3f} — {verdict}.",
        fontsize=10, color="#444")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="outputs/layer3_ppo/ppo_significance.json")
    p.add_argument("--out", default="outputs/figures/fig13_ppo_significance.png")
    args = p.parse_args()
    print(f"Saved: {render(args.results, args.out)}")
