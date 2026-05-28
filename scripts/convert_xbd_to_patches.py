"""Convert xBD (xView2) post-disaster imagery + damage targets into the
GeoDisaster-FM patch format, so the existing training pipeline can run on it.

Each xBD post image is 1024x1024 RGB; the matching target raster encodes
building damage (0=background, 1=no-damage, 2=minor, 3=major, 4=destroyed).

For the first multi-hazard experiment we do BUILDING LOCALIZATION: a binary
mask (any building footprint = 1) segmented from the post-disaster optical
image. Each 1024x1024 image is tiled into 4x 512x512 patches to match the
Sen1Floods11 patch size. One xBD "disaster" becomes one event dir
``data/processed/patches/xbd_<disaster>/`` with a manifest the loader reads.

Damage class is preserved in the label too (we save the raw class mask as
``__damage.npy``) so a 4-way damage-classification task can reuse the patches.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

DISASTERS = ["hurricane-harvey", "hurricane-florence", "guatemala-volcano",
             "mexico-earthquake", "palu-tsunami"]


def _tiles(h, w, size):
    for r in range(0, h - size + 1, size):
        for c in range(0, w - size + 1, size):
            yield r, c


def convert_disaster(disaster: str, xbd_root: Path, out_root: Path, size: int = 512,
                     hazard: str = "") -> dict | None:
    img_dir = xbd_root / "train" / "images"
    tgt_dir = xbd_root / "train" / "targets"
    posts = sorted(img_dir.glob(f"{disaster}_*_post_disaster.png"))
    if not posts:
        print(f"  {disaster}: no post images found"); return None
    out_dir = out_root / f"xbd_{disaster}"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    px_sum = np.zeros(3); px_sqsum = np.zeros(3); px_n = 0
    for img_path in posts:
        stem = img_path.stem  # e.g. hurricane-harvey_00000123_post_disaster
        tgt_path = tgt_dir / f"{stem}_target.png"
        if not tgt_path.exists():
            continue
        img = np.asarray(Image.open(img_path).convert("RGB"))          # H,W,3
        tgt = np.asarray(Image.open(tgt_path))                          # H,W
        if tgt.ndim == 3:
            tgt = tgt[..., 0]
        h, w = tgt.shape
        for r, c in _tiles(h, w, size):
            # store optical scaled to [0,1] (3,S,S); z-scored later by the loader
            opt = img[r:r + size, c:c + size, :].transpose(2, 0, 1).astype(np.float32) / 255.0
            dmg = tgt[r:r + size, c:c + size].astype(np.uint8)          # 0..4
            lab = (dmg > 0).astype(np.uint8)                            # binary building
            pid = f"{stem}_r{r}_c{c}"
            opt_p = out_dir / f"{pid}__optical.npy"
            lab_p = out_dir / f"{pid}__label.npy"
            dmg_p = out_dir / f"{pid}__damage.npy"
            np.save(opt_p, opt); np.save(lab_p, lab); np.save(dmg_p, dmg)
            px_sum += opt.reshape(3, -1).sum(1)
            px_sqsum += (opt.reshape(3, -1) ** 2).sum(1)
            px_n += opt.shape[1] * opt.shape[2]
            records.append({
                "patch_id": pid, "row": r, "col": c, "size": size,
                "sources": {"optical": str(opt_p)},
                "label_path": str(lab_p),
                "damage_path": str(dmg_p),
                "pos_fraction": float(lab.mean()),
            })
    manifest = {
        "event_id": f"xbd_{disaster}", "hazard": hazard or disaster,
        "patch_size": size, "stride": size, "n_patches": len(records),
        "ref_source": "optical", "patches": records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    mean = (px_sum / max(px_n, 1)); var = (px_sqsum / max(px_n, 1)) - mean ** 2
    std = np.sqrt(np.clip(var, 1e-8, None))
    print(f"  {disaster}: {len(records)} patches from {len(posts)} post images "
          f"| optical mean(/255)={mean.round(3).tolist()} std={std.round(3).tolist()}")
    return {"n": len(records), "px_sum": px_sum, "px_sqsum": px_sqsum, "px_n": px_n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xbd-root", default="data/raw/xbd")
    p.add_argument("--out-root", default="data/processed/patches")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--stats-out", default="data/processed/norm_stats_xbd.yaml")
    p.add_argument("--disasters", nargs="*", default=DISASTERS)
    args = p.parse_args()

    xbd_root = Path(args.xbd_root)
    g_sum = np.zeros(3); g_sq = np.zeros(3); g_n = 0
    for d in args.disasters:
        r = convert_disaster(d, xbd_root, Path(args.out_root), args.size,
                             hazard=d.split("-")[-1])
        if r:
            g_sum += r["px_sum"]; g_sq += r["px_sqsum"]; g_n += r["px_n"]
    if g_n > 0:
        gmean = g_sum / g_n
        gstd = np.sqrt(np.clip(g_sq / g_n - gmean ** 2, 1e-8, None))
        # single scalar mean/std across channels for the loader's per-source z-score
        mean_scalar = float(gmean.mean()); std_scalar = float(gstd.mean())
        Path(args.stats_out).write_text(
            f"optical:\n  mean: {mean_scalar:.5f}\n  std: {std_scalar:.5f}\n")
        print(f"Wrote {args.stats_out}: optical mean={mean_scalar:.4f} std={std_scalar:.4f}")


if __name__ == "__main__":
    main()
