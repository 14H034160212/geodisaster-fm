"""Fig 5 — pixels to decisions, USA test set.

Mirrors Hu et al. 2026 Nature paradigm: detection -> downstream system metric.
For our case: per-chip pixel-level flood mask -> chip-level (buildings affected,
road km affected). Left panel = aggregate; right panel = per-chip distributions.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(summary_json: str | Path, out_path: str | Path) -> Path:
    s = json.loads(Path(summary_json).read_text())
    totals = s["totals"]
    per_chip = s["per_chip"]

    fig, (ax_agg, ax_dist) = plt.subplots(1, 2, figsize=(14, 5.5),
                                         gridspec_kw={"width_ratios": [1, 1.6]})

    # --- Left: aggregate impact bars ---
    bld_t = totals["buildings_total"]
    bld_a = totals["buildings_affected"]
    rd_t = totals["road_km_total"]
    rd_a = totals["road_km_affected"]

    cats = ["Buildings\n(count)", "Roads\n(km)"]
    totals_y = [bld_t, rd_t]
    affected_y = [bld_a, rd_a]
    x = np.arange(len(cats))
    w = 0.36
    ax_agg.bar(x - w/2, totals_y, w, color="#cbd5e1", edgecolor="black",
               label="Total in chips")
    ax_agg.bar(x + w/2, affected_y, w, color="#d62728", edgecolor="black",
               label="Predicted affected")
    for i, (t, a) in enumerate(zip(totals_y, affected_y)):
        pct = 100 * a / max(t, 1)
        ax_agg.text(i + w/2, a + 0.03 * max(totals_y), f"{a:.0f}\n({pct:.1f}%)",
                    ha="center", fontsize=10, fontweight="bold", color="#a01818")
        ax_agg.text(i - w/2, t + 0.03 * max(totals_y), f"{t:.0f}",
                    ha="center", fontsize=10)
    ax_agg.set_xticks(x)
    ax_agg.set_xticklabels(cats)
    ax_agg.set_ylabel("Count / kilometres", fontsize=11)
    ax_agg.set_title("USA test set aggregate impact",
                     fontsize=12, loc="left", fontweight="bold")
    ax_agg.grid(True, alpha=0.3, axis="y")
    ax_agg.legend(loc="upper right", fontsize=9)

    # --- Right: per-chip distribution ---
    bld_aff = [c.get("n_buildings_affected", 0) for c in per_chip]
    rd_aff = [c.get("road_km_affected", 0.0) for c in per_chip]

    ax_dist2 = ax_dist.twinx()
    chip_x = np.arange(len(per_chip))
    ax_dist.bar(chip_x, bld_aff, color="#d62728", alpha=0.7,
                label="Buildings affected")
    ax_dist2.plot(chip_x, rd_aff, "o-", color="#1f77b4", lw=1.5, ms=4,
                  label="Road km affected", alpha=0.85)

    ax_dist.set_xlabel("Chip index (USA test set)", fontsize=11)
    ax_dist.set_ylabel("Buildings affected (count)", fontsize=11, color="#d62728")
    ax_dist2.set_ylabel("Road km affected", fontsize=11, color="#1f77b4")
    ax_dist.tick_params(axis="y", labelcolor="#d62728")
    ax_dist2.tick_params(axis="y", labelcolor="#1f77b4")
    ax_dist.set_title(
        f"Per-chip distribution ({len(per_chip)} chips)",
        fontsize=12, loc="left", fontweight="bold"
    )
    ax_dist.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Pixels → decisions: U-Net (S1+S2) flood predictions translated to "
        f"infrastructure impact across {len(per_chip)} USA test chips  |  "
        f"{bld_a}/{bld_t} buildings  ({100*bld_a/max(bld_t,1):.1f}%)   "
        f"{rd_a:.1f}/{rd_t:.1f} km roads  ({100*rd_a/max(rd_t,1):.1f}%)",
        fontsize=10.5, color="#444"
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="outputs/usa_decision/decision_summary.json")
    p.add_argument("--out",     default="outputs/figures/fig5_usa_decision.png")
    args = p.parse_args()
    out = render(args.summary, args.out)
    print(f"Saved: {out}")
