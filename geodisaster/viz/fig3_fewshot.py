"""Figure 3 — few-shot label-fraction performance curves.

Input: per-run CSV from `geodisaster.experiments.run_few_shot`. Multiple model
runs can be overlaid by passing a dict of CSV paths.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def render(
    results: dict[str, str | Path],
    out_path: str | Path,
    metric: str = "test/f1",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, csv_path in results.items():
        df = pd.read_csv(csv_path)
        col = metric if metric in df.columns else next((c for c in df.columns if c.endswith("f1")), None)
        if col is None:
            continue
        agg = df.groupby("label_fraction")[col].agg(["mean", "std"]).reset_index()
        ax.plot(agg["label_fraction"], agg["mean"], marker="o", label=label)
        ax.fill_between(agg["label_fraction"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("Label fraction")
    ax.set_ylabel(metric)
    ax.set_title("Few-shot label efficiency")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
