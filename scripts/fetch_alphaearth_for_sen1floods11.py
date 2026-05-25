"""Fetch a pixel-aligned 64-d AlphaEarth chip for every Sen1Floods11 patch.

For each Sen1Floods11 chip:
    1. Read the S1 file's CRS / affine / shape so the AE chip aligns exactly.
    2. Pick the pre-event AlphaEarth year (event_year - 1, or 2017 if earlier).
    3. ``ee.data.computePixels`` with the chip's grid -> structured numpy array.
    4. Convert (H, W) struct -> (64, H, W) float32 and save as
       ``<patch_id>__alphaearth.npy`` alongside the existing files.
    5. Update each event's ``manifest.json`` so the new source is discoverable.

Parallel with ThreadPoolExecutor; ~1-2 s per chip means ~3-5 min for 446 chips.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import time
from pathlib import Path

import ee
import numpy as np
import rasterio


# Per-region rough event year (matches sen1floods11_bridge.REGION_EVENT_YEAR).
# We use event_year - 1 as the "pre-event" year. AlphaEarth annual coverage
# starts in 2017; anything before clamps to 2017.
REGION_EVENT_YEAR = {
    "Bolivia":   2018, "Ghana":     2018, "India":     2016,
    "Mekong":    2018, "Nigeria":   2018, "Pakistan":  2017,
    "Paraguay":  2018, "Somalia":   2018, "Spain":     2019,
    "Sri-Lanka": 2017, "USA":       2016,
}
COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"


def _ae_year_for(region: str) -> int:
    return max(REGION_EVENT_YEAR.get(region, 2018) - 1, 2017)


def _structured_to_chw(arr: np.ndarray) -> np.ndarray:
    """ee.data.computePixels returns an (H, W) array with one dtype field per
    band. Convert to (C, H, W) float32 with C = band count."""
    bands = sorted(arr.dtype.names)  # A00..A63
    chw = np.stack([arr[b] for b in bands], axis=0).astype(np.float32)
    return chw


def _compute_tile(img, tf, crs, col_off, row_off, tile_w, tile_h):
    """Fetch a single sub-tile via computePixels (server casts to float32 to halve bytes)."""
    sub_translate_x = tf.c + col_off * tf.a + row_off * tf.b
    sub_translate_y = tf.f + col_off * tf.d + row_off * tf.e
    arr = ee.data.computePixels({
        "expression": img.float(),  # cast 64-d float64 -> float32 server-side
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": tile_w, "height": tile_h},
            "affineTransform": {
                "scaleX": tf.a, "shearX": tf.b, "translateX": sub_translate_x,
                "shearY": tf.d, "scaleY": tf.e, "translateY": sub_translate_y,
            },
            "crsCode": crs.to_string(),
        },
    })
    return _structured_to_chw(arr)


def fetch_chip(s1_path: Path, year: int, tile_split: int = 2) -> np.ndarray:
    """Fetch AlphaEarth aligned to ``s1_path``.

    Splits the chip into ``tile_split`` x ``tile_split`` quadrants so each
    request stays under GEE's 48 MB / computePixels limit. For 512x512x64-d
    float32: split=2 -> 256x256x64x4 = 16 MB per request.
    """
    with rasterio.open(s1_path) as src:
        crs = src.crs
        tf = src.transform
        h, w = src.height, src.width
        bounds = src.bounds

    region = ee.Geometry.Rectangle(
        [bounds.left, bounds.bottom, bounds.right, bounds.top],
        proj=crs.to_string(), geodesic=False,
    )
    img = ee.ImageCollection(COLLECTION) \
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01") \
            .filterBounds(region) \
            .mosaic() \
            .clip(region)

    tile_h = (h + tile_split - 1) // tile_split
    tile_w = (w + tile_split - 1) // tile_split
    chips: list[tuple[int, int, np.ndarray]] = []
    for ri in range(tile_split):
        for ci in range(tile_split):
            row_off = ri * tile_h
            col_off = ci * tile_w
            cur_h = min(tile_h, h - row_off)
            cur_w = min(tile_w, w - col_off)
            sub = _compute_tile(img, tf, crs, col_off, row_off, cur_w, cur_h)
            chips.append((row_off, col_off, sub))

    # Stitch into the full canvas
    c = chips[0][2].shape[0]
    out = np.zeros((c, h, w), dtype=np.float32)
    for row_off, col_off, sub in chips:
        ch, hh, ww = sub.shape
        out[:, row_off:row_off + hh, col_off:col_off + ww] = sub
    return out


def process_one(s1_path: Path, out_path: Path, region: str) -> tuple[str, str]:
    if out_path.exists() and out_path.stat().st_size > 0:
        return ("skip", str(out_path))
    try:
        year = _ae_year_for(region)
        ae = fetch_chip(s1_path, year)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, ae)
        return ("ok", str(out_path))
    except Exception as e:
        return ("err", f"{out_path.name}: {type(e).__name__}: {str(e)[:140]}")


def update_manifests(patches_root: Path):
    updated = 0
    for event_dir in patches_root.glob("sen1floods11_*"):
        m_path = event_dir / "manifest.json"
        if not m_path.exists():
            continue
        m = json.loads(m_path.read_text())
        changed = False
        for p in m["patches"]:
            patch_id = p["patch_id"]
            ae_path = event_dir / f"{patch_id}__alphaearth.npy"
            if ae_path.exists() and p["sources"].get("alphaearth") != str(ae_path):
                p["sources"]["alphaearth"] = str(ae_path)
                changed = True
        if changed:
            m_path.write_text(json.dumps(m, indent=2))
            updated += 1
    return updated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s1-root",
                   default="data/external/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand")
    p.add_argument("--patches-root", default="data/processed/patches")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--project", default="ee-bqmbill714")
    args = p.parse_args()

    ee.Initialize(project=args.project,
                  opt_url="https://earthengine-highvolume.googleapis.com")
    s1_root = Path(args.s1_root)
    patches_root = Path(args.patches_root)

    # Build job list from existing per-event manifests (so we honour Bolivia exclusion etc.)
    jobs: list[tuple[Path, Path, str]] = []
    for event_dir in sorted(patches_root.glob("sen1floods11_*")):
        region = event_dir.name.replace("sen1floods11_", "")
        m = json.loads((event_dir / "manifest.json").read_text())
        for patch in m["patches"]:
            patch_id = patch["patch_id"]
            s1_path = s1_root / f"{patch_id}.tif"
            if not s1_path.exists():
                continue
            out_path = event_dir / f"{patch_id}__alphaearth.npy"
            jobs.append((s1_path, out_path, region))
    print(f"Queued {len(jobs)} AlphaEarth fetches across "
          f"{len(set(j[2] for j in jobs))} regions  workers={args.workers}")

    t0 = time.time()
    n_ok = n_skip = n_err = 0
    errs: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, s1, out, region) for s1, out, region in jobs]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            kind, msg = fut.result()
            if kind == "ok":
                n_ok += 1
            elif kind == "skip":
                n_skip += 1
            else:
                n_err += 1
                if len(errs) < 5:
                    errs.append(msg)
            if i % 25 == 0 or i == len(futs):
                dt = time.time() - t0
                rate = i / dt if dt > 0 else 0
                print(f"  [{i:>4}/{len(futs)}] ok={n_ok} skip={n_skip} err={n_err} "
                      f"({rate:.1f} chips/s, eta={(len(futs)-i)/max(rate,0.1):.0f}s)")
    if errs:
        print("\nFirst errors:")
        for e in errs:
            print(f"  - {e}")

    updated = update_manifests(patches_root)
    print(f"\nUpdated {updated} manifests with alphaearth source.")


if __name__ == "__main__":
    main()
