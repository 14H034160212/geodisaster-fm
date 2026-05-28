"""Render the two new consolidation figures:
  fig15 — xBD cross-hazard building-localization generalization (leave-one-hazard-out)
  fig16 — calibration > structure (xBD building-damage decision: B3 beats SDI)
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def fig_cross_hazard(out="outputs/figures/fig15_xbd_cross_hazard.png"):
    d = json.loads(Path("outputs/xbd_localization/results.json").read_text())
    d = sorted(d, key=lambda r: -r["f1"])
    names = [r["test_hazard"].replace("hurricane-", "hurr-") for r in d]
    f1 = [r["f1"] for r in d]
    colors = ["#1c7f4f" if v >= 0.55 else "#d62728" for v in f1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(names)), f1, color=colors, edgecolor="black", lw=0.6)
    for i, v in enumerate(f1):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=10)
    mean = np.mean(f1)
    ax.axhline(mean, ls="--", color="#555", lw=1.2, label=f"mean F1 = {mean:.3f}")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9, rotation=15)
    ax.set_ylabel("Held-out hazard building-localization F1", fontsize=11)
    ax.set_title("Cross-hazard generalization on xBD (leave-one-hazard-out)\n"
                 "geophysical events transfer (0.59–0.64); hurricanes are hardest (0.30–0.43)",
                 fontsize=11, loc="left")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y"); ax.set_ylim(0, 0.72)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    return out


def fig_calib_vs_struct(out="outputs/figures/fig16_calibration_vs_structure.png"):
    d = json.loads(Path("outputs/xbd_damage_sdi/sdi_offline_results.json").read_text())
    m = d["methods"]
    order = ["B2_any_intersection", "B1_raw_threshold", "SDI_potts",
             "SDI_attractive", "B3_prob_threshold*"]
    labels = ["B2 any-\nintersect", "B1 raw-\nthreshold", "SDI\nPotts",
              "SDI\nattractive", "B3 calibrated\nthreshold*"]
    f1 = [m[k]["f1"] for k in order]
    colors = ["#bbbbbb", "#d62728", "#9467bd", "#ff7f0e", "#1f77b4"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(range(len(order)), f1, color=colors, edgecolor="black", lw=0.6)
    for i, v in enumerate(f1):
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Damaged-building F1 vs ground truth", fontsize=11)
    ax.set_title("Calibration > structure: on xBD building damage, a simple calibrated\n"
                 "threshold (B3) matches/beats structured inference (SDI)",
                 fontsize=11, loc="left")
    ax.grid(True, alpha=0.3, axis="y"); ax.set_ylim(0, max(f1) * 1.18)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    return out


if __name__ == "__main__":
    print("Saved:", fig_cross_hazard())
    print("Saved:", fig_calib_vs_struct())
