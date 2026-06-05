"""Fig 28 — Feature-richness ablation: 5-d (v2) vs 10-d (v3) chip features.

Documented negative result: enriching chip features with decision-frontier
proximity and probability quantiles does NOT help the learned PPO policy.
v3 loses paired significance against random/CoreSet/zero-shot that v2 had,
and becomes significantly worse than the full-pool oracle. This figure is
the honest evidence that v2's compact 5-d feature set is the right design
choice given the actor MLP capacity (2×64 Tanh) and training budget (300
updates).
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

V2 = json.loads(Path("outputs/layer3_ppo/ppo_loeo_v2_aggregate.json").read_text())
V3 = json.loads(Path("outputs/layer3_ppo/ppo_loeo_v3_aggregate.json").read_text())

comps = [("full_pool", "vs full-pool\noracle"),
         ("base",      "vs zero-shot\n(τ=0.5)"),
         ("coreset",   "vs CoreSet"),
         ("uncertainty", "vs uncertainty"),
         ("random",    "vs random")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.2, 1]})

# (a) paired-difference comparison
y = np.arange(len(comps))
yo = 0.20
for yi, (k, _) in enumerate(comps):
    p2 = V2["paired_vs_ppo"][k]; p3 = V3["paired_vs_ppo"][k]
    ax1.errorbar([p2["mean"]], [yi + yo],
                 xerr=[[p2["mean"] - p2["ci95"][0]], [p2["ci95"][1] - p2["mean"]]],
                 fmt="o", color="#1f77b4", capsize=4, markersize=8,
                 label="v2 (5-d features)" if yi == 0 else None)
    ax1.errorbar([p3["mean"]], [yi - yo],
                 xerr=[[p3["mean"] - p3["ci95"][0]], [p3["ci95"][1] - p3["mean"]]],
                 fmt="s", color="#a83232", capsize=4, markersize=8,
                 label="v3 (10-d features)" if yi == 0 else None)
    # annotate significance verdicts
    sig2 = "* " if p2["t_p"] < 0.05 else ("(*)" if p2["t_p"] < 0.10 else "n.s.")
    sig3 = "* " if p3["t_p"] < 0.05 else ("(*)" if p3["t_p"] < 0.10 else "n.s.")
    ax1.text(0.030, yi + yo, f"  t-p={p2['t_p']:.3f} {sig2}",
             fontsize=8.5, color="#1f77b4", va="center")
    ax1.text(0.030, yi - yo, f"  t-p={p3['t_p']:.3f} {sig3}",
             fontsize=8.5, color="#a83232", va="center")

ax1.axvline(0, color="#444", lw=1, ls="--")
ax1.set_yticks(y); ax1.set_yticklabels([lab for _, lab in comps], fontsize=10)
ax1.set_xlabel("PPO − comparator (F1, 95% CI)", fontsize=10)
ax1.set_title("(a) v2 (5-d chip features) significantly beats CoreSet + zero-shot;\n"
              "v3 (10-d) loses all parametric significance vs baselines",
              fontsize=10, loc="left", fontweight="bold")
ax1.legend(fontsize=9, loc="upper right"); ax1.grid(True, alpha=0.3, axis="x")
ax1.set_xlim(-0.030, 0.060); ax1.invert_yaxis()

# (b) pooled F1 ladder
methods_show = ["base", "random", "coreset", "uncertainty", "ppo", "full_pool"]
labels_show = ["zero-shot\n(τ=0.5)", "random", "CoreSet", "uncertainty",
               "PPO", "full-pool\noracle"]
v2_vals = [V2["pooled_mean"][k] for k in methods_show]
v3_vals = [V3["pooled_mean"][k] for k in methods_show]
xs = np.arange(len(methods_show)); w = 0.40
ax2.bar(xs - w/2, v2_vals, w, color="#1f77b4", label="v2 (5-d)", edgecolor="white", lw=0.7)
ax2.bar(xs + w/2, v3_vals, w, color="#a83232", label="v3 (10-d)", edgecolor="white", lw=0.7)
for xi, (v2v, v3v) in enumerate(zip(v2_vals, v3_vals)):
    ax2.text(xi - w/2, v2v + 0.0015, f"{v2v:.4f}", ha="center", fontsize=7.5)
    ax2.text(xi + w/2, v3v + 0.0015, f"{v3v:.4f}", ha="center", fontsize=7.5)
ax2.set_xticks(xs); ax2.set_xticklabels(labels_show, fontsize=8.5)
ax2.set_ylabel("Pooled test F1 (100 paired pairs)", fontsize=10)
ax2.set_title("(b) Pooled F1: PPO-v3 drops 0.005 F1 below PPO-v2\nbaselines unchanged (deterministic given pool/test seed)",
              fontsize=10, loc="left", fontweight="bold")
ax2.set_ylim(0.815, 0.85); ax2.grid(True, alpha=0.3, axis="y")
ax2.legend(fontsize=9, loc="lower right")

fig.suptitle("Feature-richness ablation under LOEO: 5-d compact features (v2) > 10-d richer features (v3)\n"
             "Negative result: enriching chip features with decision-frontier proximity + 4 probability quantiles "
             "does NOT help the learned policy; v2 is retained as the headline configuration.",
             fontsize=10, color="#444", y=1.02)
fig.tight_layout()
out = Path("outputs/figures/fig28_v2_v3_ablation.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
print(f"Saved {out}")
