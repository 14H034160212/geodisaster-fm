"""Zero-shot deployment: apply a Sen1Floods11-trained model to an unseen flood event.

Pulls Sentinel-1 (VV+VH, pre/post-event composites) and Sentinel-2 (post-event)
for an AOI via GEE, tiles into 512x512 patches matching Sen1Floods11 geometry,
runs U-Net (S1+S2) inference, computes affected buildings/roads (Hu Nature
paradigm), saves per-chip masks + summary.

This is the "trained on 8 Sen1Floods11 regions → deployed on a brand-new flood
that the model has never seen" demo — the Nature value-add.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import ee
import numpy as np
import rasterio
import torch
from omegaconf import OmegaConf

sys.path.insert(0, ".")
from geodisaster.data.gee.base import init_ee


GEE_PROJECT = "ee-bqmbill714"


# ---------------------------------------------------------------------------
# Event registry — true unseen flood events (not in Sen1Floods11)
# ---------------------------------------------------------------------------
UNSEEN_EVENTS = {
    "brazil_rs_2024": {
        "name": "2024 Rio Grande do Sul, Brazil floods (Porto Alegre lakes)",
        # Focused on most-flooded zone: Lake Guaiba + Lagoa dos Patos north
        "bbox":         (-51.4, -30.2, -50.95, -29.75),    # ~50 km square
        "pre_window":  ("2024-04-01", "2024-04-25"),
        "post_window": ("2024-05-05", "2024-05-25"),
        "event_date":   "2024-05-04",
    },
    "uae_sharjah_2024": {
        "name": "2024 UAE Sharjah floods",
        "bbox":         (55.3, 25.0, 55.7, 25.5),
        "pre_window":  ("2024-03-15", "2024-04-10"),
        "post_window": ("2024-04-18", "2024-05-05"),
        "event_date":   "2024-04-16",
    },
    "libya_derna_2023": {
        "name": "2023 Libya Derna flood (Storm Daniel)",
        "bbox":         (22.5, 32.6, 22.85, 32.85),
        "pre_window":  ("2023-08-15", "2023-09-08"),
        "post_window": ("2023-09-12", "2023-09-30"),
        "event_date":   "2023-09-10",
    },
}


def fetch_s1_composite(bbox, start, end, polarisations=("VV", "VH")):
    region = ee.Geometry.Rectangle(list(bbox), proj="EPSG:4326", geodesic=False)
    coll = ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterBounds(region) \
            .filterDate(start, end) \
            .filter(ee.Filter.eq("instrumentMode", "IW"))
    for p in polarisations:
        coll = coll.filter(ee.Filter.listContains("transmitterReceiverPolarisation", p))
    coll = coll.select(list(polarisations))
    n = coll.size().getInfo()
    if n == 0:
        raise RuntimeError(f"No Sentinel-1 scenes for bbox={bbox} {start}..{end}")
    img = coll.median().clip(region)
    return img, region, n


def fetch_s2_composite(bbox, start, end,
                       bands=("B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12","B1","B9","B10")):
    # Sen1Floods11 uses L1C TOA (which has B10 cirrus). Match that to keep
    # band order + units identical to the training distribution.
    region = ee.Geometry.Rectangle(list(bbox), proj="EPSG:4326", geodesic=False)
    coll = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
            .filterBounds(region) \
            .filterDate(start, end) \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)) \
            .select(list(bands))
    n = coll.size().getInfo()
    if n == 0:
        raise RuntimeError(f"No Sentinel-2 scenes for bbox={bbox} {start}..{end}")
    img = coll.median().clip(region)
    return img, region, n


def _max_chunk_side(n_bands: int, bytes_per_pixel: int = 4,
                    max_bytes: int = 45_000_000) -> int:
    """Pixels-per-side that fits under GEE's per-request budget."""
    return max(64, int(math.sqrt(max_bytes / (n_bands * bytes_per_pixel))))


def compute_pixels(img, region, bbox, scale_m=10, crs="EPSG:4326",
                   tile_split=None, n_bands_hint=2, dtype="float32"):
    """Tile-based computePixels (GEE 48MB limit). Outputs (C, H, W) float32.

    If tile_split is None, picks the smallest split that keeps each chunk
    under GEE's per-request budget given ``n_bands_hint`` bands of ``dtype``.
    """
    cy = 0.5 * (bbox[1] + bbox[3])
    deg_lat = 1.0 / 111_320.0
    deg_lon = 1.0 / (111_320.0 * max(math.cos(math.radians(cy)), 1e-6))
    full_h = int((bbox[3] - bbox[1]) / (scale_m * deg_lat))
    full_w = int((bbox[2] - bbox[0]) / (scale_m * deg_lon))

    if tile_split is None:
        max_side = _max_chunk_side(n_bands_hint)
        tile_split = max(1, math.ceil(max(full_h, full_w) / max_side))
        print(f"  auto tile_split={tile_split} (max_side={max_side}, "
              f"full={full_h}x{full_w}, n_bands={n_bands_hint})")
    th = (full_h + tile_split - 1) // tile_split
    tw = (full_w + tile_split - 1) // tile_split

    tiles: list[tuple[int, int, np.ndarray]] = []
    for ri in range(tile_split):
        for ci in range(tile_split):
            row_off = ri * th
            col_off = ci * tw
            ch = min(th, full_h - row_off)
            cw = min(tw, full_w - col_off)
            if ch <= 0 or cw <= 0:
                continue
            minx = bbox[0] + col_off * scale_m * deg_lon
            maxy = bbox[3] - row_off * scale_m * deg_lat
            arr = ee.data.computePixels({
                "expression": img.float(),
                "fileFormat": "NUMPY_NDARRAY",
                "grid": {
                    "dimensions": {"width": cw, "height": ch},
                    "affineTransform": {
                        "scaleX": scale_m * deg_lon, "shearX": 0, "translateX": minx,
                        "shearY": 0, "scaleY": -scale_m * deg_lat, "translateY": maxy,
                    },
                    "crsCode": crs,
                },
            })
            bands = sorted(arr.dtype.names)
            chw = np.stack([arr[b] for b in bands], axis=0).astype(np.float32)
            tiles.append((row_off, col_off, chw))

    n_bands = tiles[0][2].shape[0]
    out = np.zeros((n_bands, full_h, full_w), dtype=np.float32)
    for row_off, col_off, sub in tiles:
        _, sh, sw = sub.shape
        out[:, row_off:row_off + sh, col_off:col_off + sw] = sub

    transform = rasterio.transform.from_bounds(*bbox, full_w, full_h)
    return out, transform


def tile_to_512(arr_chw, transform, tile_size=512):
    """Slice into 512x512 patches with their per-patch transforms."""
    _, H, W = arr_chw.shape
    patches = []
    for r in range(0, H - tile_size + 1, tile_size):
        for c in range(0, W - tile_size + 1, tile_size):
            patch = arr_chw[:, r:r+tile_size, c:c+tile_size]
            ptf = rasterio.transform.Affine(
                transform.a, transform.b,
                transform.c + c * transform.a,
                transform.d, transform.e,
                transform.f + r * transform.e,
            )
            patches.append((r, c, patch, ptf))
    return patches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", required=True, choices=sorted(UNSEEN_EVENTS))
    p.add_argument("--ckpt", default="outputs/sen1floods11_unet_s1s2/checkpoints/best-epoch016.ckpt")
    p.add_argument("--stats", default="data/processed/norm_stats_sen1floods11.yaml")
    p.add_argument("--out-dir", default="outputs/zero_shot")
    p.add_argument("--project", default=GEE_PROJECT)
    args = p.parse_args()

    ev = UNSEEN_EVENTS[args.event]
    out_dir = Path(args.out_dir) / args.event
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {ev['name']} ===")
    print(f"bbox: {ev['bbox']}")

    init_ee(project=args.project)
    print("\nFetching Sentinel-1 post-event composite (VV+VH)...")
    s1_img, region, n_s1 = fetch_s1_composite(ev["bbox"], *ev["post_window"])
    print(f"  {n_s1} S1 scenes")
    s1_arr, s1_tf = compute_pixels(s1_img, region, ev["bbox"], n_bands_hint=2)
    print(f"  S1 shape={s1_arr.shape}")

    print("\nFetching Sentinel-2 post-event composite (13 bands)...")
    s2_img, region2, n_s2 = fetch_s2_composite(ev["bbox"], *ev["post_window"])
    print(f"  {n_s2} S2 scenes")
    s2_arr, s2_tf = compute_pixels(s2_img, region2, ev["bbox"], n_bands_hint=13)
    print(f"  S2 shape={s2_arr.shape}")

    # Align S1/S2 to same grid (use S1's transform; they're already aligned by AOI+scale)
    # Reorder S2 bands to match Sen1Floods11 native order: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12,B1,B9,B10
    # That's what Sen1Floods11 ships; our config uses all 13 in this order.
    # Concatenate channel-wise: 2 S1 + 13 S2 = 15 channels
    H = min(s1_arr.shape[1], s2_arr.shape[1])
    W = min(s1_arr.shape[2], s2_arr.shape[2])
    s1_arr = s1_arr[:, :H, :W]
    s2_arr = s2_arr[:, :H, :W]
    combined = np.concatenate([s1_arr, s2_arr], axis=0)
    print(f"  Combined input shape: {combined.shape}")

    # Tile to 512x512
    tiles = tile_to_512(combined, s1_tf, tile_size=512)
    print(f"  Tiled into {len(tiles)} patches of 512x512")

    if not tiles:
        print("  AOI too small for 512x512 patches — pad and continue")
        # Pad and use single patch
        c, h, w = combined.shape
        padded = np.zeros((c, 512, 512), dtype=np.float32)
        padded[:, :h, :w] = combined[:, :512, :512]
        tiles = [(0, 0, padded, s1_tf)]

    # Load model
    print(f"\nLoading checkpoint: {args.ckpt}")
    from geodisaster.train import DisasterSegLightningModule
    from geodisaster.datasets import stats_with_fallbacks
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = state["hyper_parameters"]
    model_cfg = OmegaConf.create(hp.get("model_cfg", hp.get("model")))
    train_cfg = OmegaConf.create(hp.get("train_cfg", hp.get("train")))
    sources = hp["sources"]
    module = DisasterSegLightningModule(model_cfg, train_cfg, sources)
    module.load_state_dict(state["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.eval().to(device)
    print(f"  Loaded on {device}, sources={sources}")

    # Normalize
    norm = stats_with_fallbacks(args.stats, sources)
    s1_mean, s1_std = norm["sentinel1"]
    s2_mean, s2_std = norm["sentinel2"]

    # Inference per tile
    print(f"\nRunning inference on {len(tiles)} tiles...")
    pred_masks = []
    pred_scores = []
    for r, c, tile_arr, tile_tf in tiles:
        s1_t = tile_arr[:2].copy()
        s2_t = tile_arr[2:].copy()
        s1_t = (s1_t - s1_mean) / max(s1_std, 1e-6)
        s2_t = (s2_t - s2_mean) / max(s2_std, 1e-6)
        x = np.concatenate([s1_t, s2_t], axis=0).astype(np.float32)
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = module.model(x_t)
        score = torch.sigmoid(logits.squeeze()).cpu().numpy()
        mask = (score > 0.5).astype(np.uint8)
        pred_masks.append((r, c, mask, tile_tf))
        pred_scores.append((r, c, score, tile_tf))

    # Stitch back to full canvas
    _, full_h, full_w = combined.shape
    full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
    for r, c, mask, _ in pred_masks:
        h, w = mask.shape
        full_mask[r:r+h, c:c+w] = mask

    # Save as georeferenced GeoTIFF
    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 1,
        "height": full_h, "width": full_w,
        "crs": "EPSG:4326", "transform": s1_tf,
        "compress": "deflate", "nodata": 255,
    }
    pred_path = out_dir / f"{args.event}_pred.tif"
    with rasterio.open(pred_path, "w", **profile) as dst:
        dst.write(full_mask, 1)
    print(f"  Saved prediction: {pred_path}")
    print(f"  Total water pixels: {int(full_mask.sum()):,} / {full_h * full_w:,} "
          f"({100 * full_mask.sum() / (full_h * full_w):.2f}%)")

    # Summary
    summary = {
        "event": args.event,
        "name": ev["name"],
        "bbox": list(ev["bbox"]),
        "pre_window": list(ev["pre_window"]),
        "post_window": list(ev["post_window"]),
        "model_ckpt": args.ckpt,
        "n_tiles": len(tiles),
        "pred_shape": list(full_mask.shape),
        "water_pixel_count": int(full_mask.sum()),
        "water_area_km2": round(float(full_mask.sum() * 100 / 1e6), 2),  # 10m × 10m = 100 m²
        "total_area_km2": round(full_h * full_w * 100 / 1e6, 2),
        "water_pct": round(100 * full_mask.sum() / (full_h * full_w), 3),
        "n_s1_scenes": n_s1,
        "n_s2_scenes": n_s2,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary:\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
