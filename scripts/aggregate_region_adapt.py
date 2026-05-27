"""Aggregate per-region active-adaptation runs into a smart-vs-random summary.

Reads every outputs/active_adapt/adapt_<region>.json, computes the F1 GAIN
over each region's own zero-shot baseline at each budget, then averages
across regions (mean ± s.e.m.) for uncertainty vs random selection.
Renders Fig 11 and writes a summary JSON the blog can read.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    adapt_dir = Path("outputs/active_adapt")
    files = sorted(adapt_dir.glob("adapt_*.json"))
    per_region = {}
    for f in files:
        d = json.loads(f.read_text())
        per_region[d["region"]] = d

    # Only aggregate budgets present in EVERY region (avoids mixing
    # Pakistan's extra budgets, which would give misleading sem=0 points).
    common = None
    for d in per_region.values():
        ks = set(d["budgets"])
        common = ks if common is None else (common & ks)
    common = sorted(common) if common else []

    unc_gain = {}   # budget -> list of gains across regions
    rnd_gain = {}
    for region, d in per_region.items():
        zs = d["zero_shot_f1"]
        unc = {c["k"]: c["f1"] for c in d["curves"]["uncertainty"]}
        rnd = {c["k"]: c["f1"] for c in d["curves"]["random"]}
        for k in common:
            if k in unc:
                unc_gain.setdefault(k, []).append(unc[k] - zs)
            if k in rnd:
                rnd_gain.setdefault(k, []).append(rnd[k] - zs)

    def _stats(gd):
        ks = sorted(gd)
        means = [statistics.mean(gd[k]) for k in ks]
        sems = [statistics.stdev(gd[k]) / (len(gd[k]) ** 0.5) if len(gd[k]) > 1 else 0
                for k in ks]
        return ks, means, sems

    uk, um, us = _stats(unc_gain)
    rk, rm, rs = _stats(rnd_gain)

    summary = {
        "n_regions": len(per_region),
        "regions": list(per_region),
        "budgets": uk,
        "uncertainty_mean_gain": um,
        "uncertainty_sem": us,
        "random_mean_gain": rm,
        "random_sem": rs,
    }
    Path("outputs/active_adapt/summary_all_regions.json").write_text(
        json.dumps(summary, indent=2))

    # ---- Fig 11 (2 panels) ----
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                  gridspec_kw={"width_ratios": [1, 1.15]})
    ax.axhline(0, ls="--", color="#999", lw=1, label="zero-shot (no adaptation)")
    ax.errorbar(uk, um, yerr=us, fmt="o-", color="#1f77b4", lw=2.4, ms=9,
                capsize=4, label="uncertainty (entropy) selection")
    ax.errorbar(rk, rm, yerr=rs, fmt="s-", color="#d62728", lw=2, ms=8,
                capsize=4, label="random selection")
    ax.set_xlabel("In-region labelled chips", fontsize=11.5)
    ax.set_ylabel("Mean F1 gain over zero-shot  (± s.e.m., %d regions)" % len(per_region),
                  fontsize=11.5)
    ax.set_title("Active region adaptation aggregated over all Sen1Floods11 regions",
                 fontsize=12.5, loc="left", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9.5)

    # annotate the average advantage
    if um and rm:
        adv = statistics.mean([u - r for u, r in zip(um, rm)])
        ax.text(0.98, 0.05,
                f"uncertainty beats random by\n{adv:+.3f} F1 gain on average",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                color="#1c5e3a",
                bbox=dict(boxstyle="round,pad=0.4", fc="#dcecdc", ec="#1c7f4f"))

    # ---- Panel 2: per-region best gain (adaptation helps where needed) ----
    reg_gain = []
    for region, d in per_region.items():
        zs = d["zero_shot_f1"]
        best = max([zs] + [c["f1"] for c in d["curves"]["uncertainty"]]
                   + [c["f1"] for c in d["curves"]["random"]])
        reg_gain.append((region, zs, best - zs))
    reg_gain.sort(key=lambda t: -t[2])
    regions_sorted = [t[0] for t in reg_gain]
    gains = [t[2] for t in reg_gain]
    zss = [t[1] for t in reg_gain]
    yy = np.arange(len(regions_sorted))
    bar_colors = ["#1c7f4f" if g > 0.05 else "#a86a1f" if g > 0.01 else "#bbbbbb"
                  for g in gains]
    ax2.barh(yy, gains, color=bar_colors, edgecolor="black", linewidth=0.5)
    for i, (g, z) in enumerate(zip(gains, zss)):
        ax2.text(g + 0.002, i, f"+{g:.3f}  (zs {z:.2f})", va="center", fontsize=8.5)
    ax2.set_yticks(yy); ax2.set_yticklabels(regions_sorted, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Best F1 gain from adaptation (any budget)", fontsize=10.5)
    ax2.set_title("Adaptation helps most where the gap is largest",
                  fontsize=12, loc="left", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.set_xlim(0, max(gains) * 1.45 + 0.01)

    fig.suptitle("Smart label selection beats random; the hardest regions "
                 "(low zero-shot F1) gain the most from a few in-region labels",
                 fontsize=10, color="#444")
    fig.tight_layout()
    out = "outputs/figures/fig11_region_adapt_summary.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Aggregated {len(per_region)} regions → {out}")
    print(f"  budgets: {uk}")
    print(f"  uncertainty mean gain: {[round(x,3) for x in um]}")
    print(f"  random mean gain:      {[round(x,3) for x in rm]}")


if __name__ == "__main__":
    main()
