"""Build pre+post change-detection patches for xBD building localization.

For each post image we stack the matching PRE-disaster image: optical = 6 channels
[pre R,G,B, post R,G,B] (each /255). Label = building footprint (target >= 1).
Writes xbdpp_<disaster>/ so a 6-channel U-Net can exploit pre/post change — the
known #1 lever for xBD that our post-only model lacked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMG = Path("data/raw/xbd/train/images")
TGT = Path("data/raw/xbd/train/targets")
OUT = Path("data/processed/patches")
DISASTERS = ["hurricane-harvey", "hurricane-florence", "mexico-earthquake",
             "palu-tsunami", "guatemala-volcano"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disasters", nargs="*", default=DISASTERS)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    S = args.size
    for d in args.disasters:
        posts = sorted(IMG.glob(f"{d}_*_post_disaster.png"))
        out_dir = OUT / f"xbdpp_{d}"; out_dir.mkdir(parents=True, exist_ok=True)
        recs = []
        for pp in posts:
            stem = pp.stem  # {d}_{id}_post_disaster
            pre = IMG / f"{stem.replace('_post_disaster', '_pre_disaster')}.png"
            tgt = TGT / f"{stem}_target.png"
            if not (pre.exists() and tgt.exists()):
                continue
            post_a = np.asarray(Image.open(pp).convert("RGB"))
            pre_a = np.asarray(Image.open(pre).convert("RGB"))
            t = np.asarray(Image.open(tgt)); t = t[..., 0] if t.ndim == 3 else t
            h, w = t.shape
            for r in range(0, h - S + 1, S):
                for c in range(0, w - S + 1, S):
                    pre_t = pre_a[r:r+S, c:c+S].transpose(2, 0, 1).astype(np.float32) / 255.0
                    post_t = post_a[r:r+S, c:c+S].transpose(2, 0, 1).astype(np.float32) / 255.0
                    opt6 = np.concatenate([pre_t, post_t], axis=0).astype(np.float16)  # 6,S,S (fp16: disk)
                    lab = (t[r:r+S, c:c+S] >= 1).astype(np.uint8)          # building
                    pid = f"{stem}_r{r}_c{c}"
                    op = out_dir / f"{pid}__optical.npy"; lp = out_dir / f"{pid}__label.npy"
                    np.save(op, opt6); np.save(lp, lab)
                    recs.append({"patch_id": pid, "row": r, "col": c, "size": S,
                                 "sources": {"optical": str(op)}, "label_path": str(lp),
                                 "pos_fraction": float(lab.mean())})
        (out_dir / "manifest.json").write_text(json.dumps(
            {"event_id": f"xbdpp_{d}", "hazard": d, "patch_size": S, "stride": S,
             "n_patches": len(recs), "ref_source": "optical",
             "note": "optical = 6ch [pre RGB, post RGB]", "patches": recs}, indent=2))
        print(f"  {d}: {len(recs)} pre+post patches")


if __name__ == "__main__":
    main()
