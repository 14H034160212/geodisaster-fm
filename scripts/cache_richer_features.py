"""Phase 1.2 — enrich chip cache with 5 additional features (no GPU needed).

Original cache stores 5 per-chip features:
  [pr.mean, pr.std, (pr>0.5).mean, ent.mean, ent.std]

Adds 5 more, all computed from already-cached per-pixel probabilities:
  decision_frontier_fraction  : fraction of pixels with 0.3 < p < 0.7
                                — directly informative about which chips will
                                  most-change under threshold tuning
  p_q10, p_q25, p_q75, p_q90 : probability distribution quantiles
                                — replaces "mean + std" with shape-aware info

Output: outputs/layer3_ppo/chip_cache_all10_rich.pkl
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np

SRC = Path("outputs/layer3_ppo/chip_cache_all10.pkl")
OUT = Path("outputs/layer3_ppo/chip_cache_all10_rich.pkl")


def main():
    caches = pickle.loads(SRC.read_bytes())
    out = {}
    for r, v in caches.items():
        probs_list = v["probs"]
        extra = []
        for pr in probs_list:
            # decision-frontier proximity
            frontier_frac = float(((pr > 0.3) & (pr < 0.7)).mean())
            # quantiles
            q10, q25, q75, q90 = np.quantile(pr, [0.10, 0.25, 0.75, 0.90]).astype(float)
            extra.append([frontier_frac, q10, q25, q75, q90])
        extra = np.asarray(extra, dtype=np.float32)
        rich = np.concatenate([v["feats_raw"], extra], axis=1)
        out[r] = {"probs": v["probs"], "labels": v["labels"],
                  "feats_raw": rich, "n": v["n"]}
        print(f"  {r:10s} n={v['n']} feats: {v['feats_raw'].shape[1]} → {rich.shape[1]}")
    OUT.write_bytes(pickle.dumps(out))
    print(f"\nSaved {OUT}  ({OUT.stat().st_size/1e6:.0f} MB)")
    # sanity: check no NaN
    nans = sum(np.isnan(out[r]["feats_raw"]).sum() for r in out)
    print(f"  NaN cells: {nans}")


if __name__ == "__main__":
    main()
