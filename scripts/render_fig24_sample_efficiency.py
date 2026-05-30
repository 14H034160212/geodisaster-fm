"""Fig 24 — Sample-efficiency curve of the Active Calibration PPO policy.

PPO's gain over baselines is largest at the smallest label budget and shrinks
as the budget grows — the canonical label-efficient pattern. PPO at budget=1
matches or beats all three active-learning baselines at budget=8.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BUDGETS = [1, 2, 4, 8]
arms = [("base",         "zero-shot (0.5)",   "#999", "--"),
        ("random",       "random",            "#d62728", "-"),
        ("uncertainty",  "uncertainty",       "#ff7f0e", "-"),
        ("coreset",      "coreset",           "#9467bd", "-"),
        ("ppo",          "PPO (ours)",        "#1f77b4", "-"),
        ("full_pool",    "full-pool oracle",  "#2ca02c", "--")]

means = {a: [] for a, _, _, _ in arms}
ci_lo, ci_hi = {a: [] for a, _, _, _ in arms}, {a: [] for a, _, _, _ in arms}
paired = []
for b in BUDGETS:
    d = json.loads(Path(f"outputs/layer3_ppo/ppo_sig_b{b}.json").read_text())
    a = d["aggregate"]
    for k, _, _, _ in arms:
        if k in a:
            means[k].append(a[k]["mean"])
            ci_lo[k].append(a[k]["ci95"][0]); ci_hi[k].append(a[k]["ci95"][1])
        else:
            means[k].append(np.nan); ci_lo[k].append(np.nan); ci_hi[k].append(np.nan)
    paired.append(d["paired"]["ppo_vs_random"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.2, 1]})

# (a) per-method curves
for k, lab, col, ls in arms:
    if all(np.isnan(means[k])):
        continue
    ax1.plot(BUDGETS, means[k], ls, color=col, lw=2, marker="o", ms=7, label=lab)
    if ls == "-":
        lo = np.array(ci_lo[k]); hi = np.array(ci_hi[k])
        ax1.fill_between(BUDGETS, lo, hi, color=col, alpha=0.10, linewidth=0)
ax1.set_xscale("log", base=2); ax1.set_xticks(BUDGETS); ax1.set_xticklabels(BUDGETS)
ax1.set_xlabel("Label budget (number of chips)", fontsize=11)
ax1.set_ylabel("Test F1 after calibration (mean ± 95% CI over 10 seeds)", fontsize=11)
ax1.set_title("(a) Sample-efficiency curve — PPO at budget=1 ≥ baselines at budget=8",
              fontsize=11, loc="left", fontweight="bold")
ax1.legend(fontsize=9, loc="lower right"); ax1.grid(True, alpha=0.3, which="both")
ax1.set_ylim(0.70, 0.80)

# (b) PPO − random gain shrinks as budget grows
gains = [p["mean"] for p in paired]
lo = [p["ci95"][0] for p in paired]; hi = [p["ci95"][1] for p in paired]
ax2.errorbar(BUDGETS, gains,
             yerr=[[g - l for g, l in zip(gains, lo)], [h - g for g, h in zip(gains, hi)]],
             fmt="o-", color="#1c7f4f", ms=10, lw=2, capsize=5, elinewidth=1.6)
for b, g, p in zip(BUDGETS, gains, paired):
    ax2.text(b, g + 0.005, f"+{g:.3f}\np={p['t_p']:.3f}",
             ha="center", fontsize=9, color="#1c7f4f", fontweight="bold")
ax2.axhline(0, ls="--", color="#888", lw=1.2)
ax2.set_xscale("log", base=2); ax2.set_xticks(BUDGETS); ax2.set_xticklabels(BUDGETS)
ax2.set_xlabel("Label budget", fontsize=11)
ax2.set_ylabel("Paired PPO − random F1 gain", fontsize=11)
ax2.set_title("(b) PPO's edge is biggest where labels are scarcest\n"
              "(all four budgets paired-significant)",
              fontsize=11, loc="left", fontweight="bold")
ax2.grid(True, alpha=0.3); ax2.set_ylim(-0.005, 0.08)

fig.suptitle("Active Calibration PPO is canonically label-efficient: largest gain at smallest budget; PPO@1 ≥ baselines@8",
             fontsize=10.5, color="#444")
fig.tight_layout()
out = Path("outputs/figures/fig24_sample_efficiency.png")
fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
print(f"Saved {out}")
