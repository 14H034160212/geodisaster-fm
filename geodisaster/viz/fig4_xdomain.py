"""Figure 4 — cross-domain generalization matrix heatmap."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render(csv_path: str | Path, out_path: str | Path, metric: str = "test/f1") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    if metric not in df.columns:
        metric = next((c for c in df.columns if c.endswith("f1")), df.columns[-1])

    pivot = df.pivot_table(index="train", columns="test", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(0.9 * pivot.shape[1] + 4, 0.6 * pivot.shape[0] + 3))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label=metric)
    ax.set_xlabel("Test domain")
    ax.set_ylabel("Train domain")
    ax.set_title("Cross-domain generalization")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
