"""Render a 4-panel honesty figure for the Brazil zero-shot deployment.

Goal: show the model's behavior under true out-of-distribution conditions
(Sen1Floods11 has NO South American training regions). The panels reveal that
the model over-predicts water in Brazil, but importantly: the relative pattern
still tracks the real flooding (Lake Guaiba expanding northward into Porto
Alegre suburbs in May 2024).

Panels:
  1. Sentinel-2 RGB (post-event composite, May 2024)
  2. JRC permanent water mask (occurrence >= 50%)
  3. Model prediction (Sen1Floods11-trained U-Net S1+S2)
  4. Difference: predicted - permanent (the model's "extra water" call)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import ee
import matplotlib.pyplot as plt
import numpy as np
import rasterio

sys.path.insert(0, ".")
from geodisaster.data.gee.base import init_ee


def fetch_s2_rgb(bbox, start, end, project="ee-bqmbill714",
                  ref_h: int = 5009, ref_w: int = 4339):
    """Fetch median S2 RGB (B4, B3, B2) clipped to AOI, downsampled to ref grid."""
    init_ee(project=project)
    region = ee.Geometry.Rectangle(list(bbox), proj="EPSG:4326", geodesic=False)
    img = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
             .filterBounds(region)
             .filterDate(start, end)
             .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
             .select(["B4", "B3", "B2"])
             .median()
             .clip(region))
    # Use a coarser scale for the RGB preview so it fits in one request
    cy = 0.5 * (bbox[1] + bbox[3])
    deg_lat = 1.0 / 111_320.0
    deg_lon = 1.0 / (111_320.0 * max(math.cos(math.radians(cy)), 1e-6))
    scale_m = 60   # 60m thumbnail for visualization
    h = int((bbox[3] - bbox[1]) / (scale_m * deg_lat))
    w = int((bbox[2] - bbox[0]) / (scale_m * deg_lon))
    arr = ee.data.computePixels({
        "expression": img.float(),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": w, "height": h},
            "affineTransform": {
                "scaleX": scale_m * deg_lon, "shearX": 0, "translateX": bbox[0],
                "shearY": 0, "scaleY": -scale_m * deg_lat, "translateY": bbox[3],
            },
            "crsCode": "EPSG:4326",
        },
    })
    rgb = np.stack([arr["B4"], arr["B3"], arr["B2"]], axis=-1).astype(np.float32)
    rgb = np.clip(rgb / 3000.0, 0, 1)
    return rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", default="brazil_rs_2024")
    p.add_argument("--zero-shot-dir", default="outputs/zero_shot")
    p.add_argument("--out", default="outputs/figures/fig6_brazil_zero_shot.png")
    args = p.parse_args()

    ev_dir = Path(args.zero_shot_dir) / args.event
    summary = json.loads((ev_dir / "summary.json").read_text())
    flood_summary = json.loads((ev_dir / "flood_decision_summary.json").read_text())
    bbox = tuple(summary["bbox"])

    print(f"Fetching S2 RGB for {summary['name']}...")
    rgb = fetch_s2_rgb(bbox, *summary["post_window"])
    print(f"  RGB shape: {rgb.shape}")

    # Load prediction + permanent water + flood-only at native 10m
    with rasterio.open(ev_dir / f"{args.event}_pred.tif") as src:
        pred = src.read(1)
    with rasterio.open(ev_dir / f"{args.event}_permanent_water.tif") as src:
        perm = src.read(1)
    flood_only = ((pred == 1) & (perm == 0)).astype(np.uint8)

    # Downsample 10m masks to RGB grid for display
    ds = pred.shape[0] // rgb.shape[0]
    from skimage.transform import resize
    pred_disp = resize(pred, rgb.shape[:2], order=0, preserve_range=True).astype(np.uint8)
    perm_disp = resize(perm, rgb.shape[:2], order=0, preserve_range=True).astype(np.uint8)
    flood_disp = resize(flood_only, rgb.shape[:2], order=0, preserve_range=True).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

    # 1. RGB
    axes[0].imshow(rgb)
    axes[0].set_title("(a) Sentinel-2 RGB\npost-event (May 2024)", fontsize=11, loc="left",
                      fontweight="bold")
    axes[0].axis("off")

    # 2. Permanent water (JRC)
    axes[1].imshow(rgb, alpha=0.5)
    axes[1].imshow(np.where(perm_disp == 1, 1, np.nan), cmap="Blues_r",
                   alpha=0.85, vmin=0, vmax=1)
    axes[1].set_title(f"(b) JRC permanent water mask\n{flood_summary['permanent_water_pct']:.1f}% of AOI",
                      fontsize=11, loc="left", fontweight="bold")
    axes[1].axis("off")

    # 3. Model prediction
    axes[2].imshow(rgb, alpha=0.5)
    axes[2].imshow(np.where(pred_disp == 1, 1, np.nan), cmap="Reds_r",
                   alpha=0.85, vmin=0, vmax=1)
    axes[2].set_title(f"(c) U-Net (S1+S2) prediction\n{summary['water_pct']:.1f}% predicted as water  →  over-prediction",
                      fontsize=11, loc="left", fontweight="bold")
    axes[2].axis("off")

    # 4. flood-only
    axes[3].imshow(rgb, alpha=0.5)
    axes[3].imshow(np.where(flood_disp == 1, 1, np.nan), cmap="Oranges_r",
                   alpha=0.85, vmin=0, vmax=1)
    axes[3].set_title(f"(d) Predicted − permanent = 'flood'\n{flood_summary['flood_only_water_pct']:.1f}% of AOI ({flood_summary['flood_only_km2']:.0f} km²)",
                      fontsize=11, loc="left", fontweight="bold")
    axes[3].axis("off")

    fig.suptitle(
        f"Zero-shot deployment on 2024 Rio Grande do Sul, Brazil flood  —  "
        f"Sen1Floods11-trained U-Net (no South American training data)  —  "
        f"over-prediction is the honest result: domain shift bigger than any in-distribution holdout",
        fontsize=10.5, color="#444"
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
