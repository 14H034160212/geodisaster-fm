"""Figure 1 — paradigm comparison: traditional disaster RS vs GeoDisaster-FM.

Schematic figure (no data input required). Two flow diagrams stacked.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def render(out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    boxes_old = [
        "raw event imagery",
        "manual labeling\n(thousands of hours)",
        "train segmentation\nfrom scratch",
        "per-event\nimpact map",
    ]
    boxes_new = [
        "raw event imagery\n+ AlphaEarth embedding",
        "few-shot adaptation\n(0.1–10% labels)",
        "transferable\nGeoDisaster-FM",
        "impact map +\ndecision metrics",
    ]
    for ax, boxes, title, color in [
        (axes[0], boxes_old, "Traditional disaster remote sensing", "#bdbdbd"),
        (axes[1], boxes_new, "GeoDisaster-FM", "#1f77b4"),
    ]:
        ax.set_xlim(0, len(boxes))
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.axis("off")
        for i, text in enumerate(boxes):
            ax.add_patch(mpatches.FancyBboxPatch(
                (i + 0.05, 0.2), 0.85, 0.6,
                boxstyle="round,pad=0.02", linewidth=1, edgecolor="black", facecolor=color,
            ))
            ax.text(i + 0.475, 0.5, text, ha="center", va="center", fontsize=10)
            if i < len(boxes) - 1:
                ax.annotate("", xy=(i + 1.05, 0.5), xytext=(i + 0.9, 0.5),
                            arrowprops=dict(arrowstyle="->", linewidth=1.5))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
