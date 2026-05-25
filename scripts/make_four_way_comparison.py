"""Render the four-way Sen1Floods11 cross-region comparison figure.

Overlays:
    - U-Net SAR-only (single point @ 100% labels, F1=0.618)
    - U-Net SAR+Optical (full label-fraction sweep)
    - AlphaEarth+S1 (full label-fraction sweep)
    - AlphaEarth+S1 @ 100% (single point, F1=0.610)

The headline question: does AlphaEarth beat U-Net at LOW label fractions
(Nature proposal §H1) even if it loses at 100% labels?
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def render(unet_csv: str, ae_csv: str, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    unet = pd.read_csv(unet_csv)
    ae = pd.read_csv(ae_csv)

    fig, (ax_f1, ax_aux) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: F1 curves ---
    ax_f1.plot(unet["label_fraction"] * 100, unet["test/f1"], "o-",
               color="#1f77b4", lw=2.2, ms=9,
               label="U-Net SAR+Optical (S1+S2, 15 ch)")
    ax_f1.plot(ae["label_fraction"] * 100, ae["test/f1"], "s-",
               color="#d62728", lw=2.2, ms=9,
               label="AlphaEarth + S1 (foundation 64-d + SAR)")
    ax_f1.axhline(0.6182, ls="--", color="#7f7f7f", lw=1.5,
                  label="U-Net SAR-only @ 100% labels (F1=0.618)")
    ax_f1.set_xscale("log")
    ax_f1.set_xlabel("Training label fraction (%)", fontsize=11)
    ax_f1.set_ylabel("Test F1 on held-out region (USA)", fontsize=11)
    ax_f1.set_title("Sen1Floods11 cross-region label efficiency",
                    fontsize=12, loc="left", fontweight="bold")
    ax_f1.grid(True, alpha=0.3, which="both")
    ax_f1.legend(loc="lower right", fontsize=9)

    # Best-of comparison annotation: crossover or no?
    best_unet = unet.loc[unet["test/f1"].idxmax()]
    best_ae = ae.loc[ae["test/f1"].idxmax()]
    crossover_pct = None
    # Look for where AE >= U-Net (if ever)
    merged = pd.merge(unet, ae, on="label_fraction", suffixes=("_unet", "_ae"))
    for _, row in merged.iterrows():
        if row["test/f1_ae"] >= row["test/f1_unet"]:
            crossover_pct = row["label_fraction"] * 100
            break

    if crossover_pct is not None:
        ax_f1.annotate(
            f"AE foundation\ncatches up at\n{crossover_pct:.0f}%",
            xy=(crossover_pct, ae.loc[ae["label_fraction"] == crossover_pct/100, "test/f1"].values[0]),
            xytext=(crossover_pct * 3, 0.70),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.2),
            fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightgreen", ec="green", alpha=0.6),
        )
    else:
        ax_f1.annotate(
            "U-Net SAR+Optical wins\nat every label fraction",
            xy=(50, 0.83), xytext=(50, 0.92),
            ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.9),
        )

    # --- Right: AUPRC + IoU side-by-side per fraction ---
    width = 0.35
    fracs = unet["label_fraction"].values
    x = list(range(len(fracs)))
    ax_aux.bar([xi - width/2 for xi in x], unet["test/iou"], width,
               color="#1f77b4", label="U-Net SAR+Optical IoU")
    ax_aux.bar([xi + width/2 for xi in x], ae["test/iou"], width,
               color="#d62728", label="AlphaEarth+S1 IoU")
    ax_aux.set_xticks(x)
    ax_aux.set_xticklabels([f"{int(f*100)}%" for f in fracs])
    ax_aux.set_xlabel("Training label fraction", fontsize=11)
    ax_aux.set_ylabel("Test IoU on USA", fontsize=11)
    ax_aux.set_title("Per-fraction IoU side by side",
                     fontsize=12, loc="left", fontweight="bold")
    ax_aux.grid(True, alpha=0.3, axis="y")
    ax_aux.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "Foundation prior vs multi-modal baseline — Sen1Floods11 (train: 8 regions, val: Spain, test: USA, 69 chips)",
        fontsize=11, color="#555555"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def write_table(unet_csv: str, ae_csv: str, out_path: str | Path):
    unet = pd.read_csv(unet_csv)
    ae = pd.read_csv(ae_csv)
    table = []
    for _, r in unet.iterrows():
        table.append({
            "model": "U-Net (S1+S2)",
            "label_pct": f"{int(r['label_fraction']*100)}%",
            "n_train": int(r["n_train"]),
            "f1":        round(float(r["test/f1"]), 4),
            "iou":       round(float(r["test/iou"]), 4),
            "precision": round(float(r["test/precision"]), 4),
            "recall":    round(float(r["test/recall"]), 4),
        })
    for _, r in ae.iterrows():
        table.append({
            "model": "AlphaEarth + S1",
            "label_pct": f"{int(r['label_fraction']*100)}%",
            "n_train": int(r["n_train"]),
            "f1":        round(float(r["test/f1"]), 4),
            "iou":       round(float(r["test/iou"]), 4),
            "precision": round(float(r["test/precision"]), 4),
            "recall":    round(float(r["test/recall"]), 4),
        })
    # Reference point: U-Net SAR-only
    table.append({
        "model": "U-Net (S1-only) — reference",
        "label_pct": "100%", "n_train": 252,
        "f1": 0.6182, "iou": 0.4461, "precision": 0.5230, "recall": 0.7520,
    })
    Path(out_path).write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--unet-csv", default="outputs/few_shot_unet_s1s2/few_shot_results.csv")
    p.add_argument("--ae-csv",   default="outputs/few_shot_ae_s1/few_shot_results.csv")
    p.add_argument("--out",      default="outputs/figures/fig3_four_way_comparison.png")
    p.add_argument("--table",    default="outputs/four_way_results_table.json")
    args = p.parse_args()
    out = render(args.unet_csv, args.ae_csv, args.out)
    write_table(args.unet_csv, args.ae_csv, args.table)
    print(f"Figure: {out}")
    print(f"Table:  {args.table}")
