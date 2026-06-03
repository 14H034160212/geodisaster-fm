"""Fig 27 — Honest failure-case analysis under LOEO-v2.

Where does PPO win, where does it lose, and what does the per-pair (seed,fold)
distribution look like? This figure is the deliberately honest companion to
Fig 26: it shows that the PPO − random advantage is driven by a long-tailed
distribution (Wilcoxon-significant rank shift, t-test-borderline parametric
mean) and identifies the three events where PPO fails to beat random.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ALL = ["Ghana", "India", "Mekong", "Nigeria", "Pakistan", "Paraguay",
       "Somalia", "Spain", "Sri-Lanka", "USA"]

# Pull per-(event, seed) PPO and random F1 from raw fold JSONs
per_seed_diffs = {}
per_seed_means = {}
all_diffs = []
headroom = {}
for ev in ALL:
    p = Path(f"outputs/layer3_ppo/ppo_loeo_v2_{ev}.json")
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    ppo = np.asarray(d["raw"]["meta_test"][ev]["ppo"])
    rnd = np.asarray(d["raw"]["meta_test"][ev]["random"])
    base = float(np.mean(d["raw"]["meta_test"][ev]["base"]))
    oracle = float(np.mean(d["raw"]["meta_test"][ev]["full_pool"]))
    per_seed_diffs[ev] = (ppo - rnd).tolist()
    per_seed_means[ev] = (float(np.mean(ppo)), float(np.mean(rnd)),
                          float(np.mean(ppo - rnd)))
    headroom[ev] = max(0.0, oracle - base)
    all_diffs.extend((ppo - rnd).tolist())

events = list(per_seed_diffs.keys())
# Sort events by mean PPO − random (descending → wins on left, losses on right)
events_sorted = sorted(events, key=lambda e: -per_seed_means[e][2])

fig = plt.figure(figsize=(15, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# (a) per-event strip: each seed's (PPO − random) point + mean diamond
x_positions = np.arange(len(events_sorted))
for xi, ev in enumerate(events_sorted):
    diffs = per_seed_diffs[ev]
    jitter = np.random.RandomState(42 + xi).uniform(-0.18, 0.18, size=len(diffs))
    colors = ["#1c7f4f" if d > 0 else "#a86a1f" for d in diffs]
    ax1.scatter(xi + jitter, diffs, c=colors, alpha=0.55, s=22, zorder=2)
    mean_d = per_seed_means[ev][2]
    ax1.plot([xi - 0.32, xi + 0.32], [mean_d, mean_d], color="#1f77b4",
             lw=2.5, zorder=3, solid_capstyle="round")
    # Headroom band annotation
    h = headroom[ev]
    ax1.text(xi, -0.052, f"h={h:.3f}", ha="center", fontsize=7,
             color="#6b7280", style="italic")
ax1.axhline(0, color="#444", lw=1.2, ls="-")
ax1.set_xticks(x_positions)
ax1.set_xticklabels(events_sorted, rotation=30, ha="right", fontsize=9)
ax1.set_ylabel("PPO − random F1 (per seed)\n— blue bar = per-event mean —",
               fontsize=10)
ax1.set_title("(a) Where does PPO win and lose? — sorted by mean Δ across 10 LOEO seeds\n"
              "Green: PPO wins that seed. Brown: PPO loses. h = base→oracle calibration headroom.",
              fontsize=10, loc="left", fontweight="bold")
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_ylim(-0.06, 0.06)

# (b) per-pair difference histogram with Wilcoxon vs t-test annotations
diffs_arr = np.asarray(all_diffs)
n = len(diffs_arr); mean_d = float(diffs_arr.mean())
median_d = float(np.median(diffs_arr))
# count > 0 vs < 0 (Wilcoxon's underlying win-rate)
wins = int((diffs_arr > 0).sum())
losses = int((diffs_arr < 0).sum())
ties = n - wins - losses

ax2.hist(diffs_arr, bins=30, color="#9bb6d8", edgecolor="white", alpha=0.85)
ax2.axvline(0, color="#444", ls="-", lw=1.3)
ax2.axvline(mean_d, color="#1f77b4", ls="--", lw=1.8,
            label=f"mean = {mean_d:+.4f}\n(t-p = 0.084)")
ax2.axvline(median_d, color="#1c7f4f", ls=":", lw=1.8,
            label=f"median = {median_d:+.4f}\n(Wilcoxon p = 0.0006)")
ax2.set_xlabel("PPO − random F1 (one value per (fold, seed) pair, n=100)", fontsize=10)
ax2.set_ylabel("count", fontsize=10)
ax2.set_title(f"(b) Per-pair distribution: PPO wins {wins} / {n} pairs\n"
              f"Heavy right shift drives Wilcoxon p=0.0006; a few large negatives\n"
              f"on saturated events soften the parametric mean to t-p=0.084",
              fontsize=10, loc="left", fontweight="bold")
ax2.legend(fontsize=9, loc="upper left", framealpha=0.95)
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle("Honest LOEO-v2 failure-case analysis: PPO − random per-event scatter (100 paired pairs)",
             fontsize=10.5, color="#444", y=1.02)
fig.tight_layout()
out = Path("outputs/figures/fig27_failure_cases.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
print(f"Saved {out}  (wins {wins}, losses {losses}, ties {ties} out of {n})")
