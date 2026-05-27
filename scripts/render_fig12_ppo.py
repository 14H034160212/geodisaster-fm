"""Fig 12 — Layer 3 PPO policy results.

Two panels:
  (a) PPO training curve (mean episode F1-gain return vs update).
  (b) Per-region test F1 at fixed label budget: zero-shot vs random vs
      uncertainty vs PPO vs full-pool oracle.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(results_json: str | Path, out_path: str | Path) -> Path:
    d = json.loads(Path(results_json).read_text())
    hist = d.get("ppo_history", [])
    regions = d["regions"]
    budget = d["budget"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                   gridspec_kw={"width_ratios": [1, 1.3]})

    # (a) training curve
    ax1.plot(hist, color="#1a3d6e", lw=1.6)
    if len(hist) > 10:
        k = 10
        ma = np.convolve(hist, np.ones(k) / k, mode="valid")
        ax1.plot(range(k - 1, len(hist)), ma, color="#d62728", lw=2,
                 label=f"{k}-update moving avg")
        ax1.legend(loc="lower right", fontsize=9)
    ax1.set_xlabel("PPO update", fontsize=11)
    ax1.set_ylabel("Mean episode F1-gain return", fontsize=11)
    ax1.set_title("(a) PPO training curve", fontsize=12, loc="left", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # (b) per-region comparison
    rs = list(regions)
    x = np.arange(len(rs))
    w = 0.2
    base = [regions[r]["base_f1"] for r in rs]
    rnd = [regions[r]["random_f1"] for r in rs]
    unc = [regions[r]["uncertainty_f1"] for r in rs]
    ppo = [regions[r]["ppo_f1"] for r in rs]
    ax2.bar(x - 1.5 * w, base, w, color="#bbbbbb", label="zero-shot (0.5 thr)")
    ax2.bar(x - 0.5 * w, rnd, w, color="#d62728", label="random calib")
    ax2.bar(x + 0.5 * w, unc, w, color="#ff7f0e", label="uncertainty calib")
    ax2.bar(x + 1.5 * w, ppo, w, color="#1f77b4", label="PPO policy")
    ax2.set_xticks(x); ax2.set_xticklabels(rs, fontsize=9)
    ax2.set_ylabel("Test F1 after threshold calibration", fontsize=11)
    ax2.set_title(f"(b) Per-region F1 at {budget}-chip budget",
                  fontsize=12, loc="left", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(loc="lower right", fontsize=8.5, ncol=2)
    ax2.set_ylim(0, max(max(base), max(ppo), max(unc)) * 1.15 + 0.05)

    agg = d.get("aggregate", {})
    fig.suptitle(
        "Layer 3 PPO policy for label-efficient threshold calibration  —  "
        f"avg test F1: zero-shot {agg.get('base_f1',0):.3f} → "
        f"PPO {agg.get('ppo_f1',0):.3f} (random {agg.get('random_f1',0):.3f}, "
        f"uncertainty {agg.get('uncertainty_f1',0):.3f})",
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
    p.add_argument("--results", default="outputs/layer3_ppo/ppo_results.json")
    p.add_argument("--out", default="outputs/figures/fig12_ppo.png")
    args = p.parse_args()
    print(f"Saved: {render(args.results, args.out)}")
