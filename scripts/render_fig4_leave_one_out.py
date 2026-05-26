"""Fig 4 — Sen1Floods11 leave-one-region-out heatmap (CrossEarth-style).

Renders a single panel that shows per-region difficulty of cross-domain transfer:
each row = a region used as test (the other 9 trained on it). Sorted by F1
descending so the difficulty gradient is visible at a glance.

Annotates the average + the spread; flags the hardest region.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(results_json: str | Path, out_path: str | Path) -> Path:
    results = json.loads(Path(results_json).read_text())
    results = sorted(results, key=lambda r: r["f1"], reverse=True)
    regions = [r["test_region"] for r in results]
    f1 = np.array([r["f1"] for r in results])
    iou = np.array([r["iou"] for r in results])
    auprc = np.array([r["auprc"] for r in results])
    prec = np.array([r["precision"] for r in results])
    rec = np.array([r["recall"] for r in results])

    fig, (ax_bar, ax_hm) = plt.subplots(1, 2, figsize=(15, 6),
                                       gridspec_kw={"width_ratios": [1.3, 1]})

    # --- Left: F1 / IoU / AUPRC bars per region ---
    x = np.arange(len(regions))
    width = 0.27
    ax_bar.bar(x - width, f1,    width, color="#1f77b4", label="F1")
    ax_bar.bar(x,         iou,   width, color="#ff7f0e", label="IoU")
    ax_bar.bar(x + width, auprc, width, color="#2ca02c", label="AUPRC")
    for i, v in enumerate(f1):
        ax_bar.text(i - width, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    ax_bar.axhline(f1.mean(), ls="--", color="#1f77b4", alpha=0.5,
                   label=f"avg F1 = {f1.mean():.3f}")
    ax_bar.axhline(iou.mean(), ls="--", color="#ff7f0e", alpha=0.5,
                   label=f"avg IoU = {iou.mean():.3f}")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(regions, rotation=30, ha="right")
    ax_bar.set_ylabel("Metric on held-out region", fontsize=11)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title("Leave-one-region-out generalization — U-Net (S1 + S2)",
                     fontsize=12, loc="left", fontweight="bold")
    ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.legend(loc="lower left", fontsize=9, ncol=3)

    # Annotate easiest/hardest
    ax_bar.annotate("easiest", xy=(0, f1[0]), xytext=(0, 1.0),
                    ha="center", fontsize=9, color="#2ca02c",
                    arrowprops=dict(arrowstyle="->", color="#2ca02c"))
    ax_bar.annotate("hardest", xy=(len(regions)-1, f1[-1]),
                    xytext=(len(regions)-1, 0.9), ha="center", fontsize=9,
                    color="#d62728",
                    arrowprops=dict(arrowstyle="->", color="#d62728"))

    # --- Right: heatmap (regions × metrics) ---
    metrics_matrix = np.stack([f1, iou, prec, rec, auprc], axis=1)
    metric_labels = ["F1", "IoU", "Precision", "Recall", "AUPRC"]
    im = ax_hm.imshow(metrics_matrix, cmap="viridis", vmin=0.3, vmax=1.0, aspect="auto")
    ax_hm.set_xticks(np.arange(len(metric_labels)))
    ax_hm.set_xticklabels(metric_labels)
    ax_hm.set_yticks(np.arange(len(regions)))
    ax_hm.set_yticklabels(regions)
    for i in range(metrics_matrix.shape[0]):
        for j in range(metrics_matrix.shape[1]):
            v = metrics_matrix[i, j]
            ax_hm.text(j, i, f"{v:.2f}",
                       ha="center", va="center",
                       color="white" if v < 0.6 else "black", fontsize=9)
    plt.colorbar(im, ax=ax_hm, fraction=0.04, pad=0.04)
    ax_hm.set_title("Per-metric heatmap", fontsize=12, loc="left", fontweight="bold")

    fig.suptitle(
        f"Sen1Floods11 leave-one-region-out: 10 hold-outs × U-Net (S1+S2)  "
        f"|  avg F1={f1.mean():.3f}  IoU={iou.mean():.3f}  "
        f"AUPRC={auprc.mean():.3f}  |  spread F1: {f1.min():.2f}–{f1.max():.2f}",
        fontsize=10.5, color="#444"
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="outputs/leave_one_region_out/results.json")
    p.add_argument("--out",     default="outputs/figures/fig4_leave_one_region_out.png")
    args = p.parse_args()
    out = render(args.results, args.out)
    print(f"Saved: {out}")
