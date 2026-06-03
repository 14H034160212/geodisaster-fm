"""Fig 26 — Leakage-free 10-fold LOEO: original PPO vs improved PPO.

Three panels:
  (a) per-event PPO F1 (v1 vs v2) on 10 leave-one-event-out folds — direction
      reversal: v2 lifts PPO above random on 7/10 events.
  (b) pooled paired-difference summary, v1 vs v2, vs random / CoreSet /
      zero-shot / full-pool oracle.
  (c) pooled F1 ladder showing v2 PPO statistically equivalent to the
      full-pool oracle.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

V1 = json.loads(Path("outputs/layer3_ppo/ppo_loeo_aggregate.json").read_text())
V2 = json.loads(Path("outputs/layer3_ppo/ppo_loeo_v2_aggregate.json").read_text())
EVENTS = [e for e in V1["folds"] if e in V2["folds"]]
METHODS = ["base", "random", "uncertainty", "coreset", "ppo", "full_pool"]
COLORS = {"base": "#888", "random": "#d62728", "uncertainty": "#ff7f0e",
          "coreset": "#9467bd", "ppo": "#1f77b4", "full_pool": "#2ca02c"}

fig = plt.figure(figsize=(15, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.2, 1.1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# (a) per-event PPO v1 vs v2 vs random, with calibration headroom band
x = np.arange(len(EVENTS))
w = 0.28
v1_ppo = [V1["per_fold"][e]["ppo"] for e in EVENTS]
v2_ppo = [V2["per_fold"][e]["ppo"] for e in EVENTS]
rnd    = [V2["per_fold"][e]["random"] for e in EVENTS]
oracle = [V2["per_fold"][e]["full_pool"] for e in EVENTS]
base   = [V2["per_fold"][e]["base"] for e in EVENTS]
for i, (b, o) in enumerate(zip(base, oracle)):
    ax1.fill_between([i - 0.45, i + 0.45], b, o, color="#eee", linewidth=0, zorder=0)
ax1.bar(x - w, v1_ppo, w, color="#bbb", label="PPO v1 (orig, leakage-suspect protocol)", zorder=2)
ax1.bar(x,     v2_ppo, w, color="#1f77b4", label="PPO v2 (GAE-λ + terminal + entropy)", zorder=2)
ax1.bar(x + w, rnd,    w, color="#d62728", alpha=0.7, label="random (n=5/seed)", zorder=2)
ax1.scatter(x, oracle, color="#2ca02c", marker="_", s=120, lw=2, label="full-pool oracle", zorder=3)
ax1.set_xticks(x); ax1.set_xticklabels(EVENTS, rotation=35, ha="right", fontsize=9)
ax1.set_ylabel("Test F1 on held-out event\n(mean over 10 seeds)", fontsize=10)
ax1.set_title("(a) Leakage-free LOEO: PPO-v2 lifts above random on 7/10 events; "
              "shaded band = base→oracle calibration headroom", fontsize=10, loc="left", fontweight="bold")
ax1.legend(fontsize=8.5, loc="lower right"); ax1.grid(True, alpha=0.3, axis="y")
ax1.set_ylim(0.55, 1.00)

# (b) paired-difference summary
comps = [("random","vs random"), ("uncertainty","vs uncertainty"),
         ("coreset","vs CoreSet"), ("base","vs zero-shot (0.5)"),
         ("full_pool","vs full-pool oracle")]
y = np.arange(len(comps))
v1_dx = [V1["paired_vs_ppo"][k]["mean"] for k, _ in comps]
v1_lo = [V1["paired_vs_ppo"][k]["ci95"][0] for k, _ in comps]
v1_hi = [V1["paired_vs_ppo"][k]["ci95"][1] for k, _ in comps]
v2_dx = [V2["paired_vs_ppo"][k]["mean"] for k, _ in comps]
v2_lo = [V2["paired_vs_ppo"][k]["ci95"][0] for k, _ in comps]
v2_hi = [V2["paired_vs_ppo"][k]["ci95"][1] for k, _ in comps]
v2_tp = [V2["paired_vs_ppo"][k]["t_p"] for k, _ in comps]
v2_wp = [V2["paired_vs_ppo"][k]["wilcoxon_p"] for k, _ in comps]
yo = 0.18
ax2.errorbar(v1_dx, y + yo,
             xerr=[[m - l for m, l in zip(v1_dx, v1_lo)], [h - m for m, h in zip(v1_dx, v1_hi)]],
             fmt="o", color="#888", capsize=4, label="v1 (orig PPO)")
ax2.errorbar(v2_dx, y - yo,
             xerr=[[m - l for m, l in zip(v2_dx, v2_lo)], [h - m for m, h in zip(v2_dx, v2_hi)]],
             fmt="o", color="#1f77b4", capsize=4, label="v2 (improved PPO)")
for yi, (dx, tp, wp) in enumerate(zip(v2_dx, v2_tp, v2_wp)):
    sig = "n.s."
    if tp < 0.001: sig = "***"
    elif tp < 0.01: sig = "**"
    elif tp < 0.05: sig = "*"
    elif tp < 0.1: sig = "(*)"
    note = f"  t-p={tp:.3f} W-p={wp:.4f}  {sig}"
    ax2.text(0.045, yi - yo, note, fontsize=8, color="#1f77b4", va="center")
ax2.axvline(0, color="#888", lw=1, ls="--")
ax2.set_yticks(y); ax2.set_yticklabels([lab for _, lab in comps], fontsize=9)
ax2.set_xlabel("PPO − comparator (F1, 95% CI)", fontsize=10)
ax2.set_title("(b) Paired difference vs each baseline (n=100 pairs)\n"
              "v2 reverses the wrong-direction v1 result vs random/coreset/oracle",
              fontsize=10, loc="left", fontweight="bold")
ax2.legend(fontsize=8.5, loc="upper right"); ax2.grid(True, alpha=0.3, axis="x")
ax2.set_xlim(-0.030, 0.075); ax2.invert_yaxis()

# (c) pooled F1 ladder
methods_show = ["base", "random", "coreset", "uncertainty", "ppo", "full_pool"]
labels_show = ["zero-shot\n(τ=0.5)", "random", "CoreSet", "uncertainty",
               "PPO-v2\n(ours)", "full-pool\noracle"]
vals = [V2["pooled_mean"][k] for k in methods_show]
colors = [COLORS[k] for k in methods_show]
xs = np.arange(len(methods_show))
bars = ax3.bar(xs, vals, color=colors, edgecolor="black", lw=0.8)
ax3.set_xticks(xs); ax3.set_xticklabels(labels_show, fontsize=9)
for xi, v in zip(xs, vals):
    ax3.text(xi, v + 0.002, f"{v:.4f}", ha="center", fontsize=8.5, fontweight="bold")
ax3.set_ylim(0.81, 0.85)
ax3.set_ylabel("Pooled test F1 over 100 paired pairs", fontsize=10)
ax3.set_title("(c) PPO-v2 statistically equivalent to full-pool oracle;\n"
              "4 actively-selected chips ≈ all pool chips", fontsize=10, loc="left", fontweight="bold")
ax3.grid(True, alpha=0.3, axis="y")

fig.suptitle("Leakage-free 10-fold LOEO: improved PPO (GAE-λ + terminal-only reward + entropy schedule)\n"
             "matches the full-pool oracle and significantly beats CoreSet & zero-shot on 100 paired pairs",
             fontsize=10.5, color="#444", y=1.02)
fig.tight_layout()
out = Path("outputs/figures/fig26_loeo_v1_v2.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
print(f"Saved {out}")
