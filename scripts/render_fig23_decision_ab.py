"""Fig 23 — decision-aligned reward vs pixel-F1 reward (10-seed paired A/B)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Prefer the 20-seed run if present (more honest, less noise-driven)
_UN20 = Path("outputs/layer3_ppo/decision_ab_unet_20s.json")
_AE20 = Path("outputs/layer3_ppo/decision_ab_ae_20s.json")
un = json.loads(_UN20.read_text() if _UN20.exists() else Path("outputs/layer3_ppo/decision_ab_unet.json").read_text())
ae = json.loads(_AE20.read_text() if _AE20.exists() else Path("outputs/layer3_ppo/decision_ab_ae.json").read_text())
N_SEEDS = un.get("n_seeds", 10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
arms = [("baseline", "#999"), ("pixel_arm", "#1f77b4"), ("decision_arm", "#d62728")]
labels = ["zero-shot\n(0.5 thr)", "pixel-F1\nreward", "decision\nreward"]
x = np.arange(2); w = 0.25


def get(d, arm, m):
    key = arm if arm != "baseline" else "base"
    return d[key][m]


# (a) pixel F1
for i, (a, c) in enumerate(arms):
    off = (i - 1) * w
    means = [get(un, a, "pixel_f1")["mean"], get(ae, a, "pixel_f1")["mean"]]
    err = [get(un, a, "pixel_f1")["ci95"][1] - get(un, a, "pixel_f1")["mean"],
           get(ae, a, "pixel_f1")["ci95"][1] - get(ae, a, "pixel_f1")["mean"]]
    err = [max(0.0, e) for e in err]
    ax1.bar(x + off, means, w, yerr=err, capsize=4, color=c, edgecolor='black', lw=0.5,
            label=labels[i], error_kw={"elinewidth": 1.2})
    for xi, m in zip(x + off, means):
        ax1.text(xi, m + 0.006, f"{m:.3f}", ha='center', fontsize=8.5)
ax1.set_xticks(x); ax1.set_xticklabels(["U-Net", "AlphaEarth"], fontsize=10.5, fontweight='bold')
ax1.set_ylabel("Test pixel F1", fontsize=11)
ax1.set_title(f"(a) Pixel F1 — pixel-reward best; decision-reward\n"
              f"sacrifices ~2–3 pp (paired-significant, p≤0.005, n={N_SEEDS} seeds)",
              fontsize=10.5, loc='left')
ax1.set_ylim(0.60, 0.82); ax1.legend(fontsize=9, loc='lower right'); ax1.grid(True, alpha=0.3, axis='y')

# (b) decision area error
for i, (a, c) in enumerate(arms):
    off = (i - 1) * w
    means = [get(un, a, "dec_err")["mean"], get(ae, a, "dec_err")["mean"]]
    err = [get(un, a, "dec_err")["ci95"][1] - get(un, a, "dec_err")["mean"],
           get(ae, a, "dec_err")["ci95"][1] - get(ae, a, "dec_err")["mean"]]
    err = [max(0.0, e) for e in err]
    ax2.bar(x + off, means, w, yerr=err, capsize=4, color=c, edgecolor='black', lw=0.5,
            error_kw={"elinewidth": 1.2})
    for xi, m in zip(x + off, means):
        ax2.text(xi, m + 0.5, f"{m:.2f}", ha='center', fontsize=8.5)
ax2.set_xticks(x); ax2.set_xticklabels(["U-Net", "AlphaEarth"], fontsize=10.5, fontweight='bold')
ax2.set_ylabel("Mean abs. relative area error", fontsize=11)
ax2.set_title(f"(b) Decision area error — direction backbone-dependent,\n"
              f"NOT yet significant at n={N_SEEDS}×4 (AE −22 %, U-Net +5 %)",
              fontsize=10.5, loc='left')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, max(get(un, "base", "dec_err")["ci95"][1], 18) * 1.05)

fig.suptitle(f"Reward shaping is a paired-significant control knob (a); whether a decision-aligned "
             f"reward net-improves the decision metric is not yet significant ({N_SEEDS}-seed paired A/B)",
             fontsize=10, color="#444")
fig.tight_layout()
out = Path("outputs/figures/fig23_decision_reward_ab.png")
fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
print(f"Saved {out}")
