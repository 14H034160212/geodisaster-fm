"""P0-A — score the ensemble-uncertainty baseline under the LOEO-v2 protocol.

We do NOT re-run PPO (which would be ~17h). Instead we replicate the exact
pool/test split logic used by ``scripts/eval_layer3_ppo_significance.py``
:func:`split_eval` for each of the 10 events × 10 seeds = 100 paired pairs,
select the top-``budget`` chips by **ensemble uncertainty** (per-pixel std
across the 3 multi-seed LOO models, averaged per chip — pre-computed by
``scripts/cache_ensemble_uncertainty.py`` and stored in the chip cache as
``ensemble_unc``), and compute the resulting test F1.

The output is a single JSON keyed by region with the per-seed F1 list, plus
paired-difference stats against PPO from the existing LOEO-v2 aggregate.
This gives us the missing "PPO vs ensemble uncertainty" row in the
headline R4 LOEO table without re-running the policy.
"""
from __future__ import annotations
import json, pickle, sys, statistics
from pathlib import Path
import numpy as np
import scipy.stats as sps
sys.path.insert(0, ".")
from geodisaster.dispatch.rl_policy import best_threshold_f1, f1_at_threshold
from geodisaster.utils.logging import get_logger, setup_logging

log = get_logger("eval-ensemble")
CACHE = "outputs/layer3_ppo/chip_cache_all10.pkl"
BUDGET = 4
SEEDS = list(range(20))                  # match the 20-seed LOEO protocol
ALL_REGIONS = ["Ghana", "India", "Mekong", "Nigeria", "Pakistan",
               "Paraguay", "Somalia", "Spain", "Sri-Lanka", "USA"]
OUT = Path("outputs/layer3_ppo/ensemble_baseline_loeo_20s.json")
V2_FOLD_PREFIX = "outputs/layer3_ppo/ppo_loeo_v2_20s_"


def split_pool_test(cache: dict, seed: int):
    """Replicate the pool/test split used by split_eval(seed=1000+s)."""
    rng = np.random.RandomState(seed)
    n = cache["n"]
    order = rng.permutation(n)
    n_test = max(8, n // 3)
    test_idx = order[:n_test]; pool_idx = order[n_test:]
    return pool_idx, test_idx, rng


def main():
    setup_logging()
    caches = pickle.loads(Path(CACHE).read_bytes())
    assert "ensemble_unc" in caches[ALL_REGIONS[0]], "run cache_ensemble_uncertainty.py first"

    per_region = {}
    for region in ALL_REGIONS:
        c = caches[region]
        ens = c["ensemble_unc"]
        probs, labels = c["probs"], c["labels"]
        per_seed_f1 = []
        for s in SEEDS:
            seed = 1000 + s
            pool_idx, test_idx, rng = split_pool_test(c, seed)
            # build test set with the same 2-million-pixel cap split_eval uses
            test_pr = np.concatenate([probs[i] for i in test_idx])
            test_lb = np.concatenate([labels[i] for i in test_idx])
            test_cap = 2_000_000
            if test_pr.size > test_cap:
                sel = rng.choice(test_pr.size, test_cap, replace=False)
                test_pr, test_lb = test_pr[sel], test_lb[sel]

            # Rank pool chips by ensemble uncertainty; pick top-BUDGET
            pool_ens = ens[pool_idx]
            order = np.argsort(pool_ens)[::-1]      # descending = most uncertain
            picked = pool_idx[order[: min(BUDGET, len(pool_idx))]]

            cal_pr = np.concatenate([probs[i] for i in picked])
            cal_lb = np.concatenate([labels[i] for i in picked])
            t, _ = best_threshold_f1(cal_pr, cal_lb)
            f1 = f1_at_threshold(test_pr, test_lb, t)
            per_seed_f1.append(float(f1))

        per_region[region] = per_seed_f1
        log.info("done_region", region=region,
                 mean=round(float(np.mean(per_seed_f1)), 4),
                 min=round(min(per_seed_f1), 4), max=round(max(per_seed_f1), 4))

    # Aggregate: paired-diff vs PPO from LOEO-v2
    v2 = json.loads(Path("outputs/layer3_ppo/ppo_loeo_v2_20s_aggregate.json").read_text())
    ppo_pooled = []
    ens_pooled = []
    for r in ALL_REGIONS:
        ppo_v2 = []
        # find this region's per-seed PPO values from per-fold JSON
        fold_path = Path(f"outputs/layer3_ppo/ppo_loeo_v2_20s_{r}.json")
        if fold_path.exists():
            fold = json.loads(fold_path.read_text())
            ppo_v2 = fold["raw"]["meta_test"][r]["ppo"]
        else:
            log.warning("missing_v2_fold", region=r); continue
        ppo_pooled.extend(ppo_v2)
        ens_pooled.extend(per_region[r])

    diffs = np.asarray(ens_pooled) - np.asarray(ppo_pooled)
    diffs_vs_random = []
    for r in ALL_REGIONS:
        fold = json.loads(Path(f"outputs/layer3_ppo/ppo_loeo_v2_20s_{r}.json").read_text())
        rnd = np.asarray(fold["raw"]["meta_test"][r]["random"])
        diffs_vs_random.extend(np.asarray(per_region[r]) - rnd)
    diffs_vs_random = np.asarray(diffs_vs_random)

    def paired(d):
        n = len(d); m = float(d.mean())
        if n < 2 or d.std(ddof=1) == 0:
            return {"n": n, "mean": m, "ci95": [m, m], "t_p": 1.0, "wilcoxon_p": 1.0}
        se = float(d.std(ddof=1) / np.sqrt(n))
        tc = float(sps.t.ppf(0.975, n - 1))
        ci = [m - tc * se, m + tc * se]
        t_p = float(sps.ttest_1samp(d, 0.0).pvalue)
        try:
            w_p = float(sps.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            w_p = 1.0
        return {"n": n, "mean": round(m, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)],
                "t_p": round(t_p, 4), "wilcoxon_p": round(w_p, 4)}

    out = {
        "budget": BUDGET,
        "regions": ALL_REGIONS,
        "per_region_per_seed": {r: per_region[r] for r in ALL_REGIONS},
        "pooled_mean_ensemble": round(float(np.mean(ens_pooled)), 4),
        "pooled_mean_ppo_v2":   round(float(np.mean(ppo_pooled)), 4),
        "paired_ppo_minus_ensemble": paired(-diffs),
        "paired_ensemble_minus_random": paired(diffs_vs_random),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT}")
    print(f"\n=== Ensemble uncertainty baseline (top-{BUDGET} by 3-seed pixel-std), 100 pairs ===")
    print(f"  ensemble pooled F1 = {out['pooled_mean_ensemble']}")
    print(f"  PPO-v2 pooled F1   = {out['pooled_mean_ppo_v2']}")
    print(f"  PPO-v2 − ensemble : Δ={out['paired_ppo_minus_ensemble']['mean']:+.4f}  "
          f"95%CI{out['paired_ppo_minus_ensemble']['ci95']}  t-p={out['paired_ppo_minus_ensemble']['t_p']}  "
          f"W-p={out['paired_ppo_minus_ensemble']['wilcoxon_p']}")
    print(f"  ensemble − random : Δ={out['paired_ensemble_minus_random']['mean']:+.4f}  "
          f"t-p={out['paired_ensemble_minus_random']['t_p']}")


if __name__ == "__main__":
    sys.exit(main() or 0)
