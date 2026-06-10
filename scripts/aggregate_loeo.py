"""Aggregate LOEO fold JSONs into a pooled paired statistical test.

Supports both protocol versions:
  - old (200 updates, step-level reward, no GAE): ppo_loeo_<EVENT>.json
  - v2 (300 updates, terminal_pixel reward, GAE-λ + entropy schedule):
    ppo_loeo_v2_<EVENT>.json

Pass --variant v2 to aggregate the improved-PPO results.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import scipy.stats as sps

ALL = ["Ghana", "India", "Mekong", "Nigeria", "Pakistan", "Paraguay",
       "Somalia", "Spain", "Sri-Lanka", "USA"]
LOG = Path("outputs/layer3_ppo")
METHODS = ["base", "random", "uncertainty", "coreset", "ppo", "full_pool"]


def paired(diffs):
    d = np.asarray(diffs, float); n = len(d); m = float(d.mean())
    if n < 2 or d.std(ddof=1) == 0:
        return {"mean": m, "ci95": [m, m], "se": 0.0, "t_p": 1.0, "wilcoxon_p": 1.0, "n": n}
    se = float(d.std(ddof=1) / np.sqrt(n))
    tc = float(sps.t.ppf(0.975, n - 1))
    ci = [m - tc * se, m + tc * se]
    t_p = float(sps.ttest_1samp(d, 0.0).pvalue)
    try:
        w_p = float(sps.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        w_p = 1.0
    return {"mean": m, "ci95": ci, "se": se, "t_p": t_p, "wilcoxon_p": w_p, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="", choices=["", "v2", "v3", "v2_20s"],
                    help="'' = original (ppo_loeo_<EVENT>.json); 'v2' = improved PPO "
                         "(GAE + terminal + entropy schedule).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suffix = f"_{args.variant}" if args.variant else ""
    pooled = {k: [] for k in METHODS}
    per_fold = {}
    folds_loaded = []
    for ev in ALL:
        p = LOG / f"ppo_loeo{suffix}_{ev}.json"
        if not p.exists():
            print(f"  [skip] missing fold {ev} ({p.name})"); continue
        d = json.loads(p.read_text())
        raw = d["raw"]["meta_test"][ev]
        per_fold[ev] = {k: float(np.mean(raw[k])) for k in METHODS}
        for k in METHODS:
            pooled[k].extend(raw[k])
        folds_loaded.append(ev)
    if not folds_loaded:
        print(f"No fold JSONs found for variant '{args.variant}' yet."); return
    n_pairs = len(pooled["ppo"])

    if args.variant == "v3":
        label = "PPO-v3 (v2 + richer 10-d chip features)"
    elif args.variant == "v2_20s":
        label = "PPO-v2 (20-seed LOEO, n=200)"
    elif args.variant == "v2":
        label = "PPO-v2 (GAE-λ + terminal + entropy)"
    else:
        label = "PPO (original)"
    print(f"\n=== {label}: LOEO aggregate over {len(folds_loaded)} folds = {n_pairs} paired pairs ===\n")
    print(f"{'event':<10}  " + "  ".join(f"{k:>8}" for k in METHODS))
    for ev in folds_loaded:
        vals = per_fold[ev]
        print(f"{ev:<10}  " + "  ".join(f"{vals[k]:>8.4f}" for k in METHODS))

    print(f"\n{'POOLED':<10}  " + "  ".join(f"{np.mean(pooled[k]):>8.4f}" for k in METHODS))

    print(f"\n=== Paired difference vs PPO (pooled n={n_pairs}) ===\n")
    ppo = np.asarray(pooled["ppo"])
    for k in ["random", "uncertainty", "coreset", "base", "full_pool"]:
        diff = ppo - np.asarray(pooled[k])
        p = paired(diff)
        sig = "***" if p["t_p"] < 0.001 else ("**" if p["t_p"] < 0.01 else ("*" if p["t_p"] < 0.05 else "n.s."))
        print(f"  PPO − {k:<12}  Δ={p['mean']:+.4f}  95%CI[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]  "
              f"t-p={p['t_p']:.4f}  W-p={p['wilcoxon_p']:.4f}  {sig}")

    out = {
        "variant": args.variant or "original",
        "n_folds": len(folds_loaded), "folds": folds_loaded, "n_pairs": n_pairs,
        "per_fold": per_fold,
        "pooled_mean": {k: float(np.mean(pooled[k])) for k in METHODS},
        "paired_vs_ppo": {k: paired(ppo - np.asarray(pooled[k]))
                          for k in ["random", "uncertainty", "coreset", "base", "full_pool"]},
    }
    out_path = Path(args.out) if args.out else (LOG / f"ppo_loeo{suffix}_aggregate.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
