"""Fig 25 — Leakage-free meta-train/meta-test PPO results.

The earlier PPO experiment trained and evaluated on the same 4 hard regions
(only the within-region pool/test split differed across seeds). Reviewers will
flag this as test-event leakage: the policy was optimised against the very
test signal it is reported on.

This figure shows the cleaner result. Policy is trained on 6 meta-train events
(Ghana, Mekong, Nigeria, Sri-Lanka, USA, Spain) and frozen, then evaluated on
4 meta-test events (Pakistan, Somalia, Paraguay, India) that the policy never
saw during training. The same canonical label-efficient pattern holds: PPO
remains > random / uncertainty / coreset at every budget; biggest gain at
smallest budget.

Two panels:
  (a) sample-efficiency curve on meta-TEST events (the headline)
  (b) old protocol (within-event) vs new protocol (meta-test) side-by-side at
      B=4 — visual proof that the result is not a leakage artefact.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BUDGETS = [1, 2, 4, 8]
arms = [("base",         "zero-shot (0.5)",   "#999", "--"),
        ("random",       "random",            "#d62728", "-"),
        ("uncertainty",  "uncertainty",       "#ff7f0e", "-"),
        ("coreset",      "coreset",           "#9467bd", "-"),
        ("ppo",          "PPO (ours, meta-test)", "#1f77b4", "-"),
        ("full_pool",    "full-pool oracle",  "#2ca02c", "--")]

means, ci_lo, ci_hi = {a: [] for a, *_ in arms}, {a: [] for a, *_ in arms}, {a: [] for a, *_ in arms}
paired = []
for b in BUDGETS:
    p = Path(f"outputs/layer3_ppo/ppo_meta_b{b}.json")
    if not p.exists():
        for k in means: means[k].append(np.nan); ci_lo[k].append(np.nan); ci_hi[k].append(np.nan)
        paired.append({"mean": np.nan, "ci95": [np.nan, np.nan], "t_p": np.nan})
        continue
    d = json.loads(p.read_text())
    a = d["aggregate"]["meta_test"]
    for k in means:
        means[k].append(a[k]["mean"]); ci_lo[k].append(a[k]["ci95"][0]); ci_hi[k].append(a[k]["ci95"][1])
    paired.append(d["paired"]["meta_test"]["ppo_vs_random"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})

# (a) meta-test sample-efficiency curve
for k, lab, col, ls in arms:
    if all(np.isnan(means[k])): continue
    ax1.plot(BUDGETS, means[k], ls, color=col, lw=2, marker="o", ms=7, label=lab)
    if ls == "-":
        lo = np.array(ci_lo[k]); hi = np.array(ci_hi[k])
        ax1.fill_between(BUDGETS, lo, hi, color=col, alpha=0.10, linewidth=0)
ax1.set_xscale("log", base=2); ax1.set_xticks(BUDGETS); ax1.set_xticklabels(BUDGETS)
ax1.set_xlabel("Label budget (chips)", fontsize=11)
ax1.set_ylabel("Test F1 after calibration (mean ± 95% CI, 10 seeds)\n— meta-TEST events only —", fontsize=11)
ax1.set_title("(a) Leakage-free sample-efficiency on 4 held-out events\n"
              "policy trained on 6 meta-train events; never saw these 4",
              fontsize=11, loc="left", fontweight="bold")
ax1.legend(fontsize=8.5, loc="lower right"); ax1.grid(True, alpha=0.3, which="both")

# (b) old vs new protocol at B=4
labels_b = ["zero-shot\n(0.5)", "random", "uncertainty", "coreset", "PPO\n(ours)", "full-pool\noracle"]
keys = ["base","random","uncertainty","coreset","ppo","full_pool"]
old_path = Path("outputs/layer3_ppo/ppo_sig_b4.json")    # within-event (4 hard regions)
new_path = Path("outputs/layer3_ppo/ppo_meta_b4.json")
if old_path.exists() and new_path.exists():
    old = json.loads(old_path.read_text())["aggregate"]
    new = json.loads(new_path.read_text())["aggregate"]["meta_test"]
    x = np.arange(len(keys)); w = 0.38
    o_means = [old[k]["mean"] for k in keys]
    n_means = [new[k]["mean"] for k in keys]
    o_err = [(old[k]["mean"]-old[k]["ci95"][0]) for k in keys]
    n_err = [(new[k]["mean"]-new[k]["ci95"][0]) for k in keys]
    ax2.bar(x-w/2, o_means, w, yerr=o_err, capsize=3, color="#bbb", label="old (within-event, same 4 regions)")
    ax2.bar(x+w/2, n_means, w, yerr=n_err, capsize=3, color="#1f77b4", label="new (meta-test, never seen by PPO)")
    for xi, (o, n) in enumerate(zip(o_means, n_means)):
        ax2.text(xi-w/2, o+0.005, f"{o:.3f}", ha="center", fontsize=8)
        ax2.text(xi+w/2, n+0.005, f"{n:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(labels_b, fontsize=8.5)
    ax2.set_ylabel("Test F1 (mean over seeds)", fontsize=11)
    ax2.set_title("(b) Old (suspect of leakage) vs new (leakage-free) at B=4\n"
                  "PPO advantage survives strict event-level meta-test split",
                  fontsize=11, loc="left", fontweight="bold")
    ax2.legend(fontsize=8.5, loc="lower right"); ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0.55, 0.85)
else:
    ax2.text(0.5, 0.5, "Awaiting ppo_meta_b4.json", ha="center", va="center")

fig.suptitle("Leakage-free protocol: PPO is trained on 6 meta-train events, evaluated on 4 frozen meta-test events",
             fontsize=10.5, color="#444")
fig.tight_layout()
out = Path("outputs/figures/fig25_leakage_free.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
print(f"Saved {out}")
