"""Fig 10 — Layer 3 active region adaptation curves.

Plots test F1 vs number of in-region labels for the hard hold-out region,
comparing uncertainty sampling against random selection, both starting
from the zero-shot (cross-region) baseline. This is the environment a
Layer-3 RL policy optimises; the gap between random and uncertainty is
the headroom an RL agent can claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(adapt_json: str | Path, out_path: str | Path) -> Path:
    d = json.loads(Path(adapt_json).read_text())
    region = d["region"]
    zs = d["zero_shot_f1"]
    unc = d["curves"]["uncertainty"]
    rnd = d["curves"]["random"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # zero-shot baseline
    ax.axhline(zs, ls="--", color="#7f7f7f", lw=1.5,
               label=f"zero-shot (0 in-region labels) = {zs:.3f}")

    # random with error band
    kr = [c["k"] for c in rnd]
    fr = [c["f1"] for c in rnd]
    sr = [c.get("f1_std", 0) for c in rnd]
    ax.plot([0] + kr, [zs] + fr, "s-", color="#d62728", lw=2, ms=8,
            label="random chip selection")
    ax.fill_between(kr, np.array(fr) - np.array(sr), np.array(fr) + np.array(sr),
                    color="#d62728", alpha=0.15)

    # uncertainty
    ku = [c["k"] for c in unc]
    fu = [c["f1"] for c in unc]
    ax.plot([0] + ku, [zs] + fu, "o-", color="#1f77b4", lw=2.4, ms=9,
            label="uncertainty (entropy) selection")

    ax.set_xlabel(f"In-region labelled chips ({region})", fontsize=11.5)
    ax.set_ylabel(f"Test F1 on held-out {region} chips", fontsize=11.5)
    ax.set_title(f"Layer 3 environment — active region adaptation on {region}",
                 fontsize=12.5, loc="left", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9.5)

    # annotate the best achievable
    best = max(fu + fr)
    ax.annotate(f"+{best - zs:.2f} F1 recovered\nwith ≤{max(ku)} labels",
                xy=(max(ku), best), xytext=(max(ku) * 0.45, zs + 0.02),
                arrowprops=dict(arrowstyle="->", color="#1c7f4f", lw=1.2),
                fontsize=9.5, color="#1c7f4f",
                bbox=dict(boxstyle="round,pad=0.4", fc="#dcecdc", ec="#1c7f4f"))

    fig.suptitle("A few in-region labels close most of the cross-region gap — "
                 "the headroom an RL policy can optimise",
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
    p.add_argument("--adapt-json", default="outputs/active_adapt/adapt_Pakistan.json")
    p.add_argument("--out", default="outputs/figures/fig10_active_adapt.png")
    args = p.parse_args()
    print(f"Saved: {render(args.adapt_json, args.out)}")
