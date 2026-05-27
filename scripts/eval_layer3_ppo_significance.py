"""Multi-seed significance evaluation of the Layer 3 PPO chip-selection policy.

The single-split result (scripts/train_layer3_ppo.py) showed PPO beating random
chip selection by a *small* margin (+0.012 F1) on small per-region test sets
(8-22 chips). This script asks whether that margin is real or noise:

  1. Cache EVERY chip's prediction once per region (probs + labels + features),
     re-used across all seeds (the expensive forward pass happens once).
  2. For each of S seeds: re-shuffle the pool/test split, train a fresh
     cross-region PPO policy on that split, and evaluate
     zero-shot / random / uncertainty / PPO / full-pool-oracle test F1.
  3. Aggregate across seeds: per-method mean +/- 95% CI, and a PAIRED test
     (paired t-test + Wilcoxon signed-rank) of PPO - random and PPO - uncertainty.

Pure CPU (NumPy env + small MLP) — does not touch the GPU.
"""
from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import stats as sps

sys.path.insert(0, ".")
from geodisaster.data.tile import merge_manifests
from geodisaster.datasets import DisasterPatchDataModule, stats_with_fallbacks
from geodisaster.train import DisasterSegLightningModule
from geodisaster.dispatch.rl_policy import (
    ChipCalibEnv, train_ppo, rollout_greedy, best_threshold_f1, f1_at_threshold,
)
from geodisaster.utils.logging import get_logger, setup_logging

log = get_logger("layer3_ppo_sig")

HARD_REGIONS = {
    "Pakistan": "outputs/leave_one_region_out/test_Pakistan/checkpoints",
    "Somalia":  "outputs/leave_one_region_out/test_Somalia/checkpoints",
    "Paraguay": "outputs/leave_one_region_out/test_Paraguay/checkpoints",
    "India":    "outputs/leave_one_region_out/test_India/checkpoints",
}


def _latest_ckpt(d: str) -> Path | None:
    cks = sorted(Path(d).glob("*.ckpt"))
    return cks[-1] if cks else None


def load_module(ckpt: Path):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    hp = state["hyper_parameters"]
    mcfg = OmegaConf.create(hp.get("model_cfg", hp.get("model")))
    tcfg = OmegaConf.create(hp.get("train_cfg", hp.get("train")))
    sources = hp["sources"]
    m = DisasterSegLightningModule(mcfg, tcfg, sources)
    m.load_state_dict(state["state_dict"], strict=True)
    return m.eval(), sources


def cache_all_chips(region: str, ckpt: Path, patch_root: str, stats: str):
    """Forward EVERY chip once; return per-chip flat (prob, label) + raw features."""
    module, sources = load_module(ckpt)
    patches = merge_manifests([Path(patch_root) / f"sen1floods11_{region}"])
    norm = stats_with_fallbacks(stats, sources)
    dm = DisasterPatchDataModule(train_patches=[], val_patches=[], test_patches=patches,
                                 sources=sources, batch_size=1, num_workers=0, normalize=norm)
    probs, labels, feats = [], [], []
    with torch.no_grad():
        for b in dm.test_dataloader():
            logits = module(b)
            pr = torch.sigmoid(logits.squeeze(1)).numpy().ravel()
            lab = b["mask"].numpy().ravel()
            valid = lab != 255
            pr, lab = pr[valid], lab[valid].astype(np.uint8)
            ent = -(np.clip(pr, 1e-6, 1 - 1e-6) * np.log(np.clip(pr, 1e-6, 1 - 1e-6))
                    + (1 - pr) * np.log(np.clip(1 - pr, 1e-6, 1 - 1e-6)))
            probs.append(pr.astype(np.float32)); labels.append(lab)
            feats.append([pr.mean(), pr.std(), float((pr > 0.5).mean()),
                          float(ent.mean()), float(ent.std())])
    return {"probs": probs, "labels": labels,
            "feats_raw": np.array(feats, dtype=np.float32), "n": len(probs)}


def _subsample(arr_list, idx, cap, rng):
    """Concatenate chips at idx, subsample to <= cap pixels (fixed by rng)."""
    pr = np.concatenate([arr_list[i] for i in idx])
    return pr


def split_eval(cache, seed, budget, n_rand=5, test_cap=2_000_000):
    """One seed: split pool/test, train nothing here — just build env pieces +
    evaluate the non-PPO baselines. Returns (env_builder, baseline_dict)."""
    rng = np.random.RandomState(seed)
    n = cache["n"]
    order = rng.permutation(n)
    n_test = max(8, n // 3)
    test_idx = order[:n_test]; pool_idx = order[n_test:]

    probs, labels = cache["probs"], cache["labels"]
    test_pr = np.concatenate([probs[i] for i in test_idx])
    test_lb = np.concatenate([labels[i] for i in test_idx])
    if test_pr.size > test_cap:                       # cap for speed (stable F1)
        sel = rng.choice(test_pr.size, test_cap, replace=False)
        test_pr, test_lb = test_pr[sel], test_lb[sel]

    pool_pr = [probs[i] for i in pool_idx]
    pool_lb = [labels[i] for i in pool_idx]
    feats_raw = cache["feats_raw"][pool_idx]
    feats = (feats_raw - feats_raw.mean(0)) / (feats_raw.std(0) + 1e-6)

    def make_env():
        return ChipCalibEnv(pool_pr, pool_lb, test_pr, test_lb, feats, budget=budget)

    base = f1_at_threshold(test_pr, test_lb, 0.5)

    def _cal(idxs):
        pr = np.concatenate([pool_pr[i] for i in idxs])
        lb = np.concatenate([pool_lb[i] for i in idxs])
        t, _ = best_threshold_f1(pr, lb)
        return f1_at_threshold(test_pr, test_lb, t)

    np_n = len(pool_pr)
    rnd = statistics.mean(
        _cal(list(rng.choice(np_n, min(budget, np_n), replace=False)))
        for _ in range(n_rand))
    unc_order = np.argsort(feats_raw[:, 3])[::-1]      # highest entropy
    unc = _cal(list(unc_order[:budget]))
    full = _cal(list(range(np_n)))
    return make_env, {"base": base, "random": rnd, "uncertainty": unc, "full_pool": full}


def paired(diffs):
    """Paired-difference summary: mean, 95% CI, t-test p, Wilcoxon p."""
    d = np.asarray(diffs, float)
    n = len(d); m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    tcrit = float(sps.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    ci = (m - tcrit * se, m + tcrit * se)
    t_p = float(sps.ttest_1samp(d, 0.0).pvalue) if n > 1 and d.std() > 0 else 1.0
    try:
        w_p = float(sps.wilcoxon(d).pvalue) if n > 1 and np.any(d != 0) else 1.0
    except ValueError:
        w_p = 1.0
    return {"mean": m, "ci95": list(ci), "se": se, "t_p": t_p, "wilcoxon_p": w_p, "n": n}


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--patch-root", default="data/processed/patches")
    p.add_argument("--stats", default="data/processed/norm_stats_sen1floods11.yaml")
    p.add_argument("--budget", type=int, default=4)
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--cache", default="outputs/layer3_ppo/chip_cache.pkl")
    p.add_argument("--out", default="outputs/layer3_ppo/ppo_significance.json")
    args = p.parse_args()

    # ---- cache every chip once (re-used across seeds) ----
    cache_path = Path(args.cache)
    if cache_path.exists():
        log.info("loading_cache", path=str(cache_path))
        caches = pickle.loads(cache_path.read_bytes())
    else:
        caches = {}
        for region, ckdir in HARD_REGIONS.items():
            ck = _latest_ckpt(ckdir)
            if ck is None:
                log.warning("no_ckpt", region=region); continue
            log.info("caching_all_chips", region=region, ckpt=ck.name)
            caches[region] = cache_all_chips(region, ck, args.patch_root, args.stats)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps(caches))
    regions = list(caches)
    feat_dim = 5 + 2  # features + selected-mask + budget

    seeds = list(range(args.seeds))
    # per-region per-method F1 across seeds
    rec = {r: {k: [] for k in ["base", "random", "uncertainty", "ppo", "full_pool"]}
           for r in regions}
    agg = {k: [] for k in ["base", "random", "uncertainty", "ppo", "full_pool"]}

    for s in seeds:
        # build per-region split + baselines for this seed
        builders, bases = {}, {}
        for r in regions:
            mk, b = split_eval(caches[r], seed=1000 + s, budget=args.budget)
            builders[r], bases[r] = mk, b

        # train a cross-region PPO policy on THIS seed's splits
        import random as _random
        _rng = _random.Random(s)

        def make_env():
            return builders[_rng.choice(regions)]()

        policy, _hist = train_ppo(make_env, feat_dim, n_updates=args.updates,
                                  episodes_per_update=8, lr=3e-3, seed=s, log_every=10_000)

        per_seed = {k: [] for k in agg}
        for r in regions:
            ppo_roll = rollout_greedy(policy, builders[r]())
            b = bases[r]
            vals = {"base": b["base"], "random": b["random"],
                    "uncertainty": b["uncertainty"], "ppo": ppo_roll["f1"],
                    "full_pool": b["full_pool"]}
            for k, v in vals.items():
                rec[r][k].append(v); per_seed[k].append(v)
        for k in agg:
            agg[k].append(statistics.mean(per_seed[k]))   # region-averaged per seed
        log.info("seed_done", seed=s,
                 base=round(agg["base"][-1], 3), random=round(agg["random"][-1], 3),
                 unc=round(agg["uncertainty"][-1], 3), ppo=round(agg["ppo"][-1], 3))

    # ---- summaries ----
    def mean_ci(vals):
        d = np.asarray(vals, float); n = len(d)
        m = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        tc = float(sps.t.ppf(0.975, n - 1)) if n > 1 else 0.0
        return {"mean": m, "ci95": [m - tc * se, m + tc * se], "std": float(d.std(ddof=1)) if n > 1 else 0.0}

    out = {
        "budget": args.budget, "seeds": args.seeds, "updates": args.updates,
        "regions": {}, "aggregate": {}, "paired": {},
    }
    for r in regions:
        out["regions"][r] = {k: mean_ci(rec[r][k]) for k in rec[r]}
    for k in agg:
        out["aggregate"][k] = mean_ci(agg[k])

    ppo_arr = np.asarray(agg["ppo"]); rnd_arr = np.asarray(agg["random"])
    unc_arr = np.asarray(agg["uncertainty"]); base_arr = np.asarray(agg["base"])
    out["paired"] = {
        "ppo_vs_random": paired(ppo_arr - rnd_arr),
        "ppo_vs_uncertainty": paired(ppo_arr - unc_arr),
        "ppo_vs_zeroshot": paired(ppo_arr - base_arr),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    # ---- console report ----
    a = out["aggregate"]
    print("\n=== Layer 3 PPO — multi-seed significance "
          f"({args.seeds} seeds, {args.budget}-chip budget, {args.updates} updates) ===")
    for k in ["base", "random", "uncertainty", "ppo", "full_pool"]:
        m = a[k]; print(f"  {k:12s} {m['mean']:.4f}  (95% CI {m['ci95'][0]:.4f}–{m['ci95'][1]:.4f})")
    for name, pr in out["paired"].items():
        sig = "SIGNIFICANT" if pr["t_p"] < 0.05 else "not significant"
        print(f"  Δ {name:20s} {pr['mean']:+.4f}  95%CI [{pr['ci95'][0]:+.4f},{pr['ci95'][1]:+.4f}]  "
              f"t-p={pr['t_p']:.3f}  wilcoxon-p={pr['wilcoxon_p']:.3f}  → {sig}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
