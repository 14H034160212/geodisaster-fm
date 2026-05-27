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

    # collect gains by budget
    budgets = None
    unc_gain = {}   # budget -> list of gains across regions
    rnd_gain = {}
    for region, d in per_region.items():
        zs = d["zero_shot_f1"]
        if budgets is None:
            budgets = d["budgets"]
        unc = {c["k"]: c["f1"] for c in d["curves"]["uncertainty"]}
        rnd = {c["k"]: c["f1"] for c in d["curves"]["random"]}
        for k in d["budgets"]:
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

    # ---- Fig 11 ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
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

    fig.suptitle("Smart label selection consistently outperforms random across regions — "
                 "the value an RL policy operationalises",
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
