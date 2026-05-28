"""Fig 20 — RL calibration is a backbone-agnostic lever.

Two panels: (a) mean test F1 per method (±95% CI over 10 seeds) for the same PPO
calibration protocol applied to BOTH a trainable U-Net and a frozen AlphaEarth
foundation backbone; (b) paired PPO-minus-baseline differences side-by-side —
PPO beats all three standard active-learning baselines on both backbones, with
*larger* gains on AlphaEarth.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

UN = json.loads(Path("outputs/layer3_ppo/ppo_significance.json").read_text())
AE = json.loads(Path("outputs/layer3_ppo/ppo_significance_ae.json").read_text())

methods = [("base", "zero-shot\n(0.5)"),
           ("random", "random"),
           ("uncertainty", "uncertainty"),
           ("coreset", "coreset"),
           ("ppo", "PPO\npolicy"),
           ("full_pool", "full-pool\noracle")]
paired = [("ppo_vs_zeroshot", "vs zero-shot"),
          ("ppo_vs_random", "vs random"),
          ("ppo_vs_uncertainty", "vs uncertainty"),
          ("ppo_vs_coreset", "vs coreset")]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})

# (a) mean F1 ± 95% CI per method, U-Net vs AE
x = np.arange(len(methods)); w = 0.38
for i, (data, name, col) in enumerate(zip([UN, AE], ["U-Net (S1+S2)", "AlphaEarth (S1+S2)"],
                                          ["#1f77b4", "#d62728"])):
    a = data["aggregate"]
    means = [a[k]["mean"] for k, _ in methods]
    lo = [a[k]["mean"] - a[k]["ci95"][0] for k, _ in methods]
    hi = [a[k]["ci95"][1] - a[k]["mean"] for k, _ in methods]
    offs = (-w/2) if i == 0 else (w/2)
    ax.bar(x + offs, means, w, yerr=[lo, hi], capsize=4, color=col, edgecolor="black",
           lw=0.5, label=name, error_kw={"elinewidth": 1.2})
ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in methods], fontsize=9.5)
ax.set_ylabel("Test F1 after calibration", fontsize=11)
ax.set_title("(a) Same PPO protocol, two backbones — 10-seed mean ± 95% CI",
             fontsize=11.5, loc="left", fontweight="bold")
ax.legend(fontsize=10, loc="lower right"); ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0.60, 0.86)

# (b) paired PPO − baseline differences side-by-side
y = np.arange(len(paired))[::-1]
for i, (data, col, label) in enumerate(zip([UN, AE], ["#1f77b4", "#d62728"],
                                            ["U-Net", "AlphaEarth"])):
    offs = (-0.18) if i == 0 else (0.18)
    for j, (key, _) in enumerate(paired):
        p = data["paired"][key]; m = p["mean"]; ci = p["ci95"]; pv = p["t_p"]
        ax2.errorbar([m], [y[j] + offs], xerr=[[m - ci[0]], [ci[1] - m]], fmt="o",
                     color=col, ms=9, capsize=5, elinewidth=1.6,
                     label=label if j == 0 else None)
        ax2.text(ci[1] + 0.002, y[j] + offs, f"{m:+.3f} (p={pv:.3f})",
                 va="center", fontsize=8.5, color=col)
ax2.axvline(0, ls="--", color="#888", lw=1.2)
ax2.set_yticks(y); ax2.set_yticklabels([lab for _, lab in paired], fontsize=10.5)
ax2.set_xlabel("Paired PPO − baseline F1 (mean ± 95% CI, 10 seeds)", fontsize=10.5)
ax2.set_title("(b) RL beats every standard AL baseline on both backbones —\n"
              "and the gain is LARGER on the foundation model",
              fontsize=11.5, loc="left", fontweight="bold")
ax2.grid(True, alpha=0.3, axis="x"); ax2.legend(fontsize=10, loc="lower right")
ax2.set_xlim(-0.005, 0.085)

fig.suptitle("Reinforcement-learning calibration is backbone-agnostic — and helps the foundation model more",
             fontsize=11, color="#444")
fig.tight_layout()
out = Path("outputs/figures/fig20_rl_backbone.png")
fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
print(f"Saved {out}")
