"""Fig 1 — The H1-vs-H2 question and how our three tests discriminate.

Four-panel headline figure for the manuscript:

(a) Conceptual cartoon. Two prediction-score histograms per panel
    (positive class vs negative class). Under H1 (representation drift)
    the new-event histograms are distorted — overlap increases, ranking
    breaks. Under H2 (calibration drift) the per-class histograms keep
    their shape but slide along the score axis — the optimal threshold
    moves while the ranking is preserved.

(b) Test of H2(a) — pixel ranking transfers. We use F1@τ* (the
    region-optimal F1) as a proxy for the rank-based ceiling on each
    event; if H2 holds, F1@τ* should stay high across all events even
    when F1@0.5 collapses. Per-event scatter of F1@0.5 vs F1@τ*.

(c) Test of H2(b) — recalibration recovers F1. Same per-event bar
    chart: F1@0.5 (zero-shot) vs F1@τ* (region-optimal). The Pakistan
    lever is +0.184 F1, palu-tsunami is +0.235 in the xBD benchmark.

(d) Quantification of H2 — four labels reach the oracle ceiling.
    Pooled LOEO F1 across 100 paired pairs for six methods.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

CAL = json.loads(Path("outputs/decision/calibration_analysis.json").read_text())
V2  = json.loads(Path("outputs/layer3_ppo/ppo_loeo_v2_20s_aggregate.json").read_text())
ENS = json.loads(Path("outputs/layer3_ppo/ensemble_baseline_loeo.json").read_text())

fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30,
                      height_ratios=[1.0, 1.0])

# -----------------------------------------------------------------------------
# Panel (a) — H1 vs H2 conceptual cartoon
# -----------------------------------------------------------------------------
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_xlim(0, 10); ax_a.set_ylim(0, 6)
ax_a.set_xticks([]); ax_a.set_yticks([])
for spine in ax_a.spines.values():
    spine.set_visible(False)

def _gauss(x, mu, sig):
    return np.exp(-((x - mu)**2) / (2*sig**2)) / (sig * np.sqrt(2*np.pi))

x = np.linspace(0, 1, 200)

def _box(ax, x0, y0, w, h, title, color):
    ax.add_patch(patches.FancyBboxPatch((x0, y0), w, h,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor=color,
                                        linewidth=0, alpha=0.16))
    ax.text(x0 + w/2, y0 + h - 0.32, title, ha="center", va="top",
            fontsize=11, fontweight="bold", color=color)

# H1 panel
_box(ax_a, 0.2, 0.3, 4.5, 5.3, "H1 — representation drift", "#a83232")
# Training-event histograms (well-separated)
xs = np.linspace(0, 1, 100)
ax_train_neg = _gauss(xs, 0.30, 0.10)
ax_train_pos = _gauss(xs, 0.72, 0.08)
ax_a.fill_between(0.4 + xs*3.6, 0.85, 0.85 + ax_train_neg*0.10, color="#888", alpha=0.5)
ax_a.fill_between(0.4 + xs*3.6, 0.85, 0.85 + ax_train_pos*0.10, color="#1c7f4f", alpha=0.55)
ax_a.text(2.45, 1.85, "training event",
          ha="center", fontsize=8.5, color="#444")

# New-event histograms (distorted, big overlap → ranking broken)
ax_new_neg = _gauss(xs, 0.48, 0.18)
ax_new_pos = _gauss(xs, 0.55, 0.17)
ax_a.fill_between(0.4 + xs*3.6, 2.85, 2.85 + ax_new_neg*0.08, color="#888", alpha=0.5)
ax_a.fill_between(0.4 + xs*3.6, 2.85, 2.85 + ax_new_pos*0.08, color="#1c7f4f", alpha=0.55)
ax_a.text(2.45, 4.05, "new event (H1)\n— ranking broken —",
          ha="center", fontsize=8.5, color="#a83232", fontweight="bold")
ax_a.annotate("", xy=(2.45, 2.78), xytext=(2.45, 2.05),
              arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
ax_a.text(2.45, 0.75, "prediction score →", ha="center", fontsize=7.5, color="#555")

# H2 panel
_box(ax_a, 5.3, 0.3, 4.5, 5.3, "H2 — calibration drift", "#1f5fbe")
# Training-event histograms (same as H1)
ax_a.fill_between(5.5 + xs*3.6, 0.85, 0.85 + ax_train_neg*0.10, color="#888", alpha=0.5)
ax_a.fill_between(5.5 + xs*3.6, 0.85, 0.85 + ax_train_pos*0.10, color="#1c7f4f", alpha=0.55)
ax_a.text(7.55, 1.85, "training event",
          ha="center", fontsize=8.5, color="#444")
# Tau = 0.5 marker training
ax_a.axvline(5.5 + 0.5*3.6, ymin=0.85/6, ymax=2.0/6, color="#444", ls="--", lw=1.0)

# New-event histograms — same shape, slid along score axis
ax_new2_neg = _gauss(xs, 0.42, 0.10)   # shifted right vs training
ax_new2_pos = _gauss(xs, 0.82, 0.08)
ax_a.fill_between(5.5 + xs*3.6, 2.85, 2.85 + ax_new2_neg*0.10, color="#888", alpha=0.5)
ax_a.fill_between(5.5 + xs*3.6, 2.85, 2.85 + ax_new2_pos*0.10, color="#1c7f4f", alpha=0.55)
# Tau = 0.5 marker (now wrong)
ax_a.axvline(5.5 + 0.5*3.6, ymin=2.85/6, ymax=4.0/6, color="#a83232", ls="--", lw=1.0, alpha=0.7)
# Tau* marker (now correct)
ax_a.axvline(5.5 + 0.62*3.6, ymin=2.85/6, ymax=4.0/6, color="#1c7f4f", ls="-", lw=1.4)
ax_a.text(7.55, 4.05, "new event (H2)\n— ranking preserved,\nτ shifts —",
          ha="center", fontsize=8.5, color="#1f5fbe", fontweight="bold")
ax_a.annotate("", xy=(7.55, 2.78), xytext=(7.55, 2.05),
              arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
ax_a.text(5.5 + 0.5*3.6, 5.05, "old τ=0.5", ha="center", fontsize=7, color="#a83232")
ax_a.text(5.5 + 0.62*3.6, 5.35, "new τ*", ha="center", fontsize=7, color="#1c7f4f", fontweight="bold")
ax_a.text(7.55, 0.75, "prediction score →", ha="center", fontsize=7.5, color="#555")

ax_a.set_title("(a) Two competing hypotheses for cross-disaster generalisation failure",
               fontsize=11, fontweight="bold", loc="left", pad=8)

# -----------------------------------------------------------------------------
# Panel (b) — Test of H2(a): scatter F1@0.5 vs F1@τ* per event
# -----------------------------------------------------------------------------
ax_b = fig.add_subplot(gs[0, 1])
events = list(CAL["per_region"].keys())
f1_zero = np.array([CAL["per_region"][e]["f1_at_0.5"] for e in events])
f1_star = np.array([CAL["per_region"][e]["f1_at_best"] for e in events])

ax_b.scatter(f1_zero, f1_star, s=80, color="#1f5fbe", edgecolor="white", lw=0.6, zorder=3)
for e, x, y in zip(events, f1_zero, f1_star):
    dy = -0.025 if e in ("Pakistan",) else 0.015
    ha = "left" if e == "Pakistan" else "center"
    ax_b.annotate(e, (x, y), fontsize=8.5, color="#333",
                  textcoords="offset points", xytext=(4 if ha == "left" else 0, dy*100),
                  ha=ha)
ax_b.plot([0.4, 1.0], [0.4, 1.0], color="#888", ls="--", lw=1.0,
          label="y = x (no recalibration gain)")
ax_b.fill_between([0.4, 1.0], [0.4, 1.0], [1.0, 1.0],
                  color="#1f5fbe", alpha=0.05, zorder=1)
ax_b.set_xlim(0.5, 1.0); ax_b.set_ylim(0.5, 1.0)
ax_b.set_xlabel("F1 at default threshold τ = 0.5", fontsize=10)
ax_b.set_ylabel("F1 at region-optimal threshold τ*", fontsize=10)
ax_b.set_title("(b) Test of H2(a): ranking survives — F1@τ* high across all 10 events\n"
               "(every point above y = x; Pakistan +0.184 F1 from τ alone)",
               fontsize=10, fontweight="bold", loc="left")
ax_b.grid(True, alpha=0.3)
ax_b.legend(fontsize=9, loc="lower right")

# -----------------------------------------------------------------------------
# Panel (c) — Test of H2(b): F1@0.5 vs F1@τ* bar chart per event
# -----------------------------------------------------------------------------
ax_c = fig.add_subplot(gs[1, 0])
order = np.argsort(f1_zero)  # hardest first
events_s = [events[i] for i in order]
f1_zero_s = f1_zero[order]; f1_star_s = f1_star[order]
gain = f1_star_s - f1_zero_s

x = np.arange(len(events_s))
w = 0.40
ax_c.bar(x - w/2, f1_zero_s, w, color="#bbb", edgecolor="white", lw=0.5, label="F1 @ τ = 0.5")
ax_c.bar(x + w/2, f1_star_s, w, color="#1c7f4f", edgecolor="white", lw=0.5, label="F1 @ τ* (region-optimal)")
for xi, (z, s) in enumerate(zip(f1_zero_s, f1_star_s)):
    ax_c.text(xi, max(z, s) + 0.012, f"+{s - z:.3f}",
              ha="center", fontsize=7.5, color="#1c7f4f", fontweight="bold")
ax_c.set_xticks(x); ax_c.set_xticklabels(events_s, rotation=30, ha="right", fontsize=8.5)
ax_c.set_ylabel("F1 on held-out event", fontsize=10)
ax_c.set_ylim(0.5, 1.0)
ax_c.set_title("(c) Test of H2(b): recalibration recovers F1 on every event\n"
               "(Pakistan +0.184, Somalia +0.036, mean across 10 events +0.030)",
               fontsize=10, fontweight="bold", loc="left")
ax_c.legend(fontsize=9, loc="lower right")
ax_c.grid(True, alpha=0.3, axis="y")

# -----------------------------------------------------------------------------
# Panel (d) — Quantification of H2: 4 labels = oracle (LOEO pooled F1)
# -----------------------------------------------------------------------------
ax_d = fig.add_subplot(gs[1, 1])
methods = [("base", "zero-shot\n(τ=0.5)", "#888"),
           ("random", "random", "#d62728"),
           ("coreset", "CoreSet", "#9467bd"),
           ("ensemble", "ensemble\nuncertainty", "#7a4cae"),
           ("uncertainty", "uncertainty\n(entropy)", "#ff7f0e"),
           ("ppo", "PPO\n(ours, 4 chips)", "#1f77b4"),
           ("full_pool", "full-pool\noracle", "#2ca02c")]

vals = []
for k, _, _ in methods:
    if k == "ensemble":
        vals.append(ENS["pooled_mean_ensemble"])
    else:
        vals.append(V2["pooled_mean"][k])

colors = [m[2] for m in methods]
labels = [m[1] for m in methods]
xs = np.arange(len(methods))
bars = ax_d.bar(xs, vals, color=colors, edgecolor="white", lw=0.7)
# Highlight PPO and oracle
ppo_idx = next(i for i, m in enumerate(methods) if m[0] == "ppo")
oracle_idx = next(i for i, m in enumerate(methods) if m[0] == "full_pool")
bars[ppo_idx].set_edgecolor("#1f77b4"); bars[ppo_idx].set_linewidth(2)
bars[oracle_idx].set_edgecolor("#2ca02c"); bars[oracle_idx].set_linewidth(2)

for xi, v in zip(xs, vals):
    ax_d.text(xi, v + 0.0015, f"{v:.4f}", ha="center", fontsize=8)

ax_d.set_xticks(xs); ax_d.set_xticklabels(labels, fontsize=8.5)
ax_d.set_ylabel("Pooled test F1 (200 paired pairs, 20-seed LOEO)", fontsize=10)
ax_d.set_ylim(0.815, 0.85)
ax_d.set_title("(d) Quantifying H2: four labels recover ≈99 % of the oracle\n"
               "(PPO 0.834 vs oracle 0.840; entire active-selection family within 0.017 F1)",
               fontsize=10, fontweight="bold", loc="left")
ax_d.grid(True, alpha=0.3, axis="y")

# Connecting bracket between PPO and oracle
ymax_bar = max(vals[ppo_idx], vals[oracle_idx]) + 0.005
ax_d.plot([ppo_idx, ppo_idx, oracle_idx, oracle_idx],
          [vals[ppo_idx]+0.002, ymax_bar, ymax_bar, vals[oracle_idx]+0.002],
          color="#444", lw=1)
ax_d.text((ppo_idx + oracle_idx)/2, ymax_bar + 0.001,
          "PPO recovers ≈99 % of oracle\n(Δ = −0.007, t-p = 0.016)", ha="center", fontsize=8, color="#444")

fig.suptitle(
    "The cross-disaster generalisation problem is calibrational, not representational, "
    "and the calibration lever is four labels wide",
    fontsize=12.5, fontweight="bold", color="#222", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = Path("outputs/figures/fig1_h1_h2_concept.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
print(f"Saved {out}")
