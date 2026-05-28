"""Build binary 'damaged-building' training patches from the existing xBD
patches: label = 1 where damage class >= 3 (major or destroyed), else 0.

Writes xbddmg_<disaster>/ event dirs whose manifest reuses the existing optical
.npy and points label_path at a new __dmg.npy. Lets the standard training
pipeline learn a post-image damage-evidence model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("data/processed/patches")
DISASTERS = ["hurricane-harvey", "hurricane-florence", "mexico-earthquake",
             "palu-tsunami", "guatemala-volcano"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--disasters", nargs="*", default=DISASTERS)
    p.add_argument("--damage-thresh", type=int, default=3, help="class >= this = damaged")
    args = p.parse_args()

    for d in args.disasters:
        src = ROOT / f"xbd_{d}"
        man_p = src / "manifest.json"
        if not man_p.exists():
            print(f"  skip {d}: no manifest"); continue
        man = json.loads(man_p.read_text())
        out_dir = ROOT / f"xbddmg_{d}"; out_dir.mkdir(parents=True, exist_ok=True)
        recs = []
        n_pos_px = n_px = 0
        for r in man["patches"]:
            dmg_path = r.get("damage_path")
            if not dmg_path or not Path(dmg_path).exists():
                continue
            dmg = np.load(dmg_path)
            lab = (dmg >= args.damage_thresh).astype(np.uint8)
            n_pos_px += int(lab.sum()); n_px += lab.size
            lab_p = out_dir / f"{r['patch_id']}__dmg.npy"
            np.save(lab_p, lab)
            recs.append({"patch_id": r["patch_id"], "row": r.get("row", 0),
                         "col": r.get("col", 0), "size": r.get("size", 512),
                         "sources": {"optical": r["sources"]["optical"]},
                         "label_path": str(lab_p),
                         "pos_fraction": float(lab.mean())})
        (out_dir / "manifest.json").write_text(json.dumps(
            {"event_id": f"xbddmg_{d}", "hazard": man.get("hazard", d),
             "patch_size": 512, "stride": 512, "n_patches": len(recs),
             "ref_source": "optical", "label": f"damage>={args.damage_thresh}",
             "patches": recs}, indent=2))
        frac = 100 * n_pos_px / max(n_px, 1)
        print(f"  {d}: {len(recs)} patches | damaged-pixel fraction {frac:.3f}%")


if __name__ == "__main__":
    main()
