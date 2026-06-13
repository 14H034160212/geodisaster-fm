"""Fig 29 + Fig 30 — the two new main figures for submission.

Fig 29 — The zero-label test (R4c). Panel (a): pooled F1 ladder showing
  EM (0.210) and BBSE (0.677) both below the uncorrected default (0.819)
  while every 4-label method sits near the oracle (0.84). Panel (b): the
  prior-shift diagnosis — per event, train prior vs true new prior vs
  pool-mean predicted probability; the score inflation (mean p >> both
  priors) is the distortion no zero-label method can see.

Fig 30 — The three-backbone ordering with 5-seed error bars. Panel (a):
  F1@tau* per backbone. Panel (b): calibration gain per backbone with
  error bars; ordering U-Net < Prithvi < DOFA.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ZL = json.loads(Path("outputs/layer3_ppo/zero_label_prior_correction.json").read_text())
GM = json.loads(Path("outputs/decision/gradient_multiseed.json").read_text())
UNET = json.loads(Path("outputs/leave_one_event_out_burnscars/summary.json").read_text())

# ---------------------------------------------------------------- Fig 29
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                               gridspec_kw={"width_ratios": [1, 1.3]})

methods = [("em", "Saerens EM\n(0 labels)", "#b3261e"),
           ("bbse", "BBSE\n(0 labels)", "#e07b39"),
           ("base", "uncorrected\nτ = 0.5", "#888"),
           ("random", "random\n(4 labels)", "#d62728"),
           ("ppo", "PPO\n(4 labels)", "#1f77b4"),
           ("full_pool", "full-pool\noracle", "#2ca02c")]
vals = [ZL["pooled_mean"][k] for k, _, _ in methods]
xs = np.arange(len(methods))
bars = ax1.bar(xs, vals, color=[c for _, _, c in methods], edgecolor="white")
for xi, v in zip(xs, vals):
    ax1.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax1.axhline(ZL["pooled_mean"]["base"], color="#555", ls="--", lw=1.1)
ax1.text(0.45, ZL["pooled_mean"]["base"] + 0.015, "doing nothing (τ = 0.5)",
         fontsize=8.5, color="#555")
ax1.set_xticks(xs)
ax1.set_xticklabels([l for _, l, _ in methods], fontsize=8.5)
ax1.set_ylabel("Pooled test F1 (200 paired pairs, LOEO)", fontsize=10)
ax1.set_ylim(0, 0.95)
ax1.set_title("(a) Zero-label prior corrections fail;\nany 4-label method recovers ≈99 % of oracle",
              fontsize=10.5, loc="left", fontweight="bold")
ax1.grid(True, alpha=0.3, axis="y")

events = list(ZL["per_region"].keys())
x = np.arange(len(events))
w = 0.27
train_pi = [ZL["per_region"][e]["train_prior"] for e in events]
true_pi = [ZL["per_region"][e]["diag_true_pool_prior"] for e in events]
mean_p = [ZL["per_region"][e]["diag_mean_pool_prob"] for e in events]
ax2.bar(x - w, train_pi, w, color="#888", label="training prior π")
ax2.bar(x, true_pi, w, color="#1c7f4f", label="TRUE new-event prior π′")
ax2.bar(x + w, mean_p, w, color="#b3261e", label="pool-mean predicted prob. p̄")
ax2.set_xticks(x)
ax2.set_xticklabels(events, rotation=35, ha="right", fontsize=8.5)
ax2.set_ylabel("probability", fontsize=10)
ax2.set_title("(b) The diagnosis: predicted probabilities (red) systematically exceed\n"
              "BOTH priors — score-distribution distortion, invisible to zero-label methods",
              fontsize=10.5, loc="left", fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle("The zero-label test: cross-disaster calibration drift is not pure label shift",
             fontsize=11.5, fontweight="bold", y=1.00)
fig.tight_layout()
out = Path("outputs/figures/fig29_zero_label_test.png")
fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
print(f"Saved {out}")

# ---------------------------------------------------------------- Fig 30
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

backbones = ["U-Net\n(task-trained)", "Prithvi\n(modality-matched)", "DOFA\n(generalist)"]
colors = ["#1f77b4", "#7a4cae", "#b3261e"]

unet_f1 = float(np.mean([UNET[y]["final_f1_at_best"] for y in UNET]))
f1_means = [unet_f1, GM["prithvi"]["overall_f1_best_mean"], GM["dofa"]["overall_f1_best_mean"]]
f1_stds = [0.0, GM["prithvi"]["overall_f1_best_std"], GM["dofa"]["overall_f1_best_std"]]
unet_gain = float(np.mean([UNET[y]["final_f1_at_best"] - UNET[y]["final_f1_at_0.5"] for y in UNET]))
g_means = [unet_gain, GM["prithvi"]["overall_gain_mean"], GM["dofa"]["overall_gain_mean"]]
g_stds = [0.0, GM["prithvi"]["overall_gain_std"], GM["dofa"]["overall_gain_std"]]

xs = np.arange(3)
ax1.bar(xs, f1_means, 0.55, yerr=f1_stds, capsize=6, color=colors, edgecolor="white")
for xi, (m, sd) in enumerate(zip(f1_means, f1_stds)):
    lab = f"{m:.3f}" + ("" if sd == 0 else f"\n±{sd:.3f}")
    ax1.text(xi, m + sd + 0.008, lab, ha="center", fontsize=9)
ax1.set_xticks(xs); ax1.set_xticklabels(backbones, fontsize=9.5)
ax1.set_ylabel("F1 @ τ* on held-out fire season (LOEO mean)", fontsize=10)
ax1.set_ylim(0.6, 0.95)
ax1.set_title("(a) Cross-event F1: no foundation model beats\nthe from-scratch U-Net (wildfires)",
              fontsize=10.5, loc="left", fontweight="bold")
ax1.grid(True, alpha=0.3, axis="y")

ax2.bar(xs, g_means, 0.55, yerr=g_stds, capsize=6, color=colors, edgecolor="white")
for xi, (m, sd) in enumerate(zip(g_means, g_stds)):
    lab = f"+{m:.4f}" + ("" if sd == 0 else f"\n±{sd:.4f}")
    ax2.text(xi, m + sd + 0.0006, lab, ha="center", fontsize=9)
ax2.set_xticks(xs); ax2.set_xticklabels(backbones, fontsize=9.5)
ax2.set_ylabel("Recoverable calibration gain (F1@τ* − F1@0.5)", fontsize=10)
ax2.set_title("(b) Calibration drift rises as task-match weakens\n(consistent ordering; intervals overlap — preliminary)",
              fontsize=10.5, loc="left", fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle("Three-backbone ordering on HLS Burn-Scars (5-seed head re-training; U-Net single full run)",
             fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
out = Path("outputs/figures/fig30_backbone_ordering.png")
fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
print(f"Saved {out}")
