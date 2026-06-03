"""Leakage-free PPO calibration-policy evaluation: strict meta-train / meta-test.

The original eval_layer3_ppo_significance.py trained a PPO policy on the same
4 hard regions it then evaluated on (only the within-region pool/test split
differed across seeds). A reviewer can argue this constitutes test-event
leakage: the policy was optimised against the very test signal it is reported
on.

This script implements the cleaner meta-learning protocol that the reviewer
asks for:

  meta-train events (policy training only)  : Ghana, Mekong, Nigeria,
                                              Sri-Lanka, USA, Spain
  meta-test  events (frozen evaluation only): Pakistan, Somalia, Paraguay,
                                              India

For each seed:
  1. split each event's chips into pool/test (within-event)
  2. train a SHARED cross-event PPO policy using ONLY the meta-train events'
     pool/test splits — the policy never sees any meta-test event during
     training
  3. freeze the policy; run rollout_greedy on each META-TEST event using its
     pool/test split; record F1
  4. compare PPO against random / uncertainty / coreset / full-pool baselines
     on the SAME meta-test events

The meta-train events are reported separately for completeness, but the
headline statistics use ONLY the meta-test events — these are the numbers
that survive the leakage critique.
"""
from __future__ import annotations
import argparse, json, pickle, statistics, sys
from pathlib import Path
import numpy as np
import scipy.stats as sps
sys.path.insert(0, ".")
from geodisaster.dispatch.rl_policy import (
    ChipCalibEnv, train_ppo, rollout_greedy, best_threshold_f1, f1_at_threshold,
)
from geodisaster.utils.logging import get_logger, setup_logging
from scripts.eval_layer3_ppo_significance import (
    cache_all_chips, _latest_ckpt, split_eval, paired,
)

log = get_logger("ppo-meta")

DEFAULT_META_TRAIN = ["Ghana", "Mekong", "Nigeria", "Sri-Lanka", "USA", "Spain"]
DEFAULT_META_TEST  = ["Pakistan", "Somalia", "Paraguay", "India"]


def mean_ci(vals):
    d = np.asarray(vals, float); n = len(d)
    m = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    tc = float(sps.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    return {"mean": m, "ci95": [m - tc * se, m + tc * se],
            "std": float(d.std(ddof=1)) if n > 1 else 0.0}


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="outputs/layer3_ppo/chip_cache_all10.pkl")
    p.add_argument("--meta-train", nargs="+", default=DEFAULT_META_TRAIN)
    p.add_argument("--meta-test",  nargs="+", default=DEFAULT_META_TEST)
    p.add_argument("--budget", type=int, default=4)
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--reward-mode", default="pixel",
                   help="'pixel' (per-step F1 gain, original) or 'terminal_pixel' "
                        "(episode-level F1 gain, recommended with GAE-λ).")
    p.add_argument("--ent-start", type=float, default=0.10)
    p.add_argument("--ent-end",   type=float, default=0.01)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--out", default="outputs/layer3_ppo/ppo_meta_b4.json")
    args = p.parse_args()
    all_hist = []

    cache_path = Path(args.cache)
    assert cache_path.exists(), f"chip cache missing: {cache_path} (run scripts/cache_all10_chips.py first)"
    caches = pickle.loads(cache_path.read_bytes())
    for r in args.meta_train + args.meta_test:
        assert r in caches, f"region {r} not in cache (have {list(caches)})"
    feat_dim = 5 + 2

    seeds = list(range(args.seeds))
    METHODS = ["base", "random", "uncertainty", "coreset", "ppo", "full_pool"]
    rec_test  = {r: {k: [] for k in METHODS} for r in args.meta_test}
    rec_train = {r: {k: [] for k in METHODS} for r in args.meta_train}
    agg_test  = {k: [] for k in METHODS}
    agg_train = {k: [] for k in METHODS}

    for s in seeds:
        # per-region split + baseline values
        builders, bases = {}, {}
        for r in args.meta_train + args.meta_test:
            mk, b = split_eval(caches[r], seed=1000 + s, budget=args.budget,
                               reward_mode=args.reward_mode)
            builders[r], bases[r] = mk, b

        # train policy ONLY on meta-train events — meta-test events are
        # invisible to PPO during training
        import random as _random
        _rng = _random.Random(s)

        def make_env_train():
            return builders[_rng.choice(args.meta_train)]()

        policy, hist = train_ppo(make_env_train, feat_dim, n_updates=args.updates,
                                 episodes_per_update=8, lr=args.lr, seed=s,
                                 log_every=max(1, args.updates // 6),
                                 gae_lambda=args.gae_lambda,
                                 ent_start=args.ent_start, ent_end=args.ent_end)
        all_hist.append(hist)

        # evaluate the FROZEN policy on every region (meta-train + meta-test)
        per_seed_test  = {k: [] for k in METHODS}
        per_seed_train = {k: [] for k in METHODS}
        for r in args.meta_train:
            ppo_roll = rollout_greedy(policy, builders[r]())
            b = bases[r]
            vals = {"base": b["base"], "random": b["random"],
                    "uncertainty": b["uncertainty"], "coreset": b["coreset"],
                    "ppo": ppo_roll["f1"], "full_pool": b["full_pool"]}
            for k, v in vals.items():
                rec_train[r][k].append(v); per_seed_train[k].append(v)
        for r in args.meta_test:
            ppo_roll = rollout_greedy(policy, builders[r]())
            b = bases[r]
            vals = {"base": b["base"], "random": b["random"],
                    "uncertainty": b["uncertainty"], "coreset": b["coreset"],
                    "ppo": ppo_roll["f1"], "full_pool": b["full_pool"]}
            for k, v in vals.items():
                rec_test[r][k].append(v); per_seed_test[k].append(v)
        for k in METHODS:
            agg_train[k].append(statistics.mean(per_seed_train[k]))
            agg_test[k].append(statistics.mean(per_seed_test[k]))
        log.info("seed_done", seed=s,
                 train_ppo=round(agg_train["ppo"][-1], 3),
                 test_ppo=round(agg_test["ppo"][-1], 3),
                 test_random=round(agg_test["random"][-1], 3))

    out = {
        "budget": args.budget, "seeds": args.seeds, "updates": args.updates,
        "meta_train": args.meta_train, "meta_test": args.meta_test,
        "headline_split": "meta_test_only",
        "regions": {"meta_train": {}, "meta_test": {}},
        "aggregate": {"meta_train": {}, "meta_test": {}},
        "paired": {"meta_train": {}, "meta_test": {}},
    }
    for r in args.meta_train:
        out["regions"]["meta_train"][r] = {k: mean_ci(rec_train[r][k]) for k in METHODS}
    for r in args.meta_test:
        out["regions"]["meta_test"][r] = {k: mean_ci(rec_test[r][k]) for k in METHODS}
    # raw per-seed per-region values — needed for LOEO pooled paired stats
    out["raw"] = {"meta_train": {r: {k: list(rec_train[r][k]) for k in METHODS} for r in args.meta_train},
                  "meta_test":  {r: {k: list(rec_test[r][k])  for k in METHODS} for r in args.meta_test}}
    for k in METHODS:
        out["aggregate"]["meta_train"][k] = mean_ci(agg_train[k])
        out["aggregate"]["meta_test"][k]  = mean_ci(agg_test[k])

    def paired_block(agg):
        ppo  = np.asarray(agg["ppo"])
        rnd  = np.asarray(agg["random"])
        unc  = np.asarray(agg["uncertainty"])
        core = np.asarray(agg["coreset"])
        base = np.asarray(agg["base"])
        full = np.asarray(agg["full_pool"])
        return {
            "ppo_vs_random":      paired(ppo - rnd),
            "ppo_vs_uncertainty": paired(ppo - unc),
            "ppo_vs_coreset":     paired(ppo - core),
            "ppo_vs_zeroshot":    paired(ppo - base),
            "ppo_vs_fullpool":    paired(ppo - full),
        }
    out["paired"]["meta_train"] = paired_block(agg_train)
    out["paired"]["meta_test"]  = paired_block(agg_test)
    out["training_returns"] = [list(h) for h in all_hist]   # convergence diagnostic

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    a_te = out["aggregate"]["meta_test"]; a_tr = out["aggregate"]["meta_train"]
    print(f"\n=== Leakage-free PPO ({args.seeds} seeds, budget={args.budget}, {args.updates} updates) ===")
    print(f"  meta-train events ({len(args.meta_train)}): {', '.join(args.meta_train)}")
    print(f"  meta-test  events ({len(args.meta_test)}): {', '.join(args.meta_test)}")
    print(f"\n  ----- meta-TEST (headline; policy never saw these) -----")
    for k in METHODS:
        v = a_te[k]
        print(f"    {k:14s}  {v['mean']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")
    p_te = out["paired"]["meta_test"]
    print(f"\n    PPO − random         = {p_te['ppo_vs_random']['mean']:+.4f}  "
          f"95%CI {p_te['ppo_vs_random']['ci95']}  t-p={p_te['ppo_vs_random']['t_p']:.4f}")
    print(f"    PPO − uncertainty    = {p_te['ppo_vs_uncertainty']['mean']:+.4f}  t-p={p_te['ppo_vs_uncertainty']['t_p']:.4f}")
    print(f"    PPO − coreset        = {p_te['ppo_vs_coreset']['mean']:+.4f}  t-p={p_te['ppo_vs_coreset']['t_p']:.4f}")
    print(f"    PPO − zero-shot(0.5) = {p_te['ppo_vs_zeroshot']['mean']:+.4f}  t-p={p_te['ppo_vs_zeroshot']['t_p']:.4f}")
    print(f"    PPO − full-pool      = {p_te['ppo_vs_fullpool']['mean']:+.4f}  t-p={p_te['ppo_vs_fullpool']['t_p']:.4f}")
    print(f"\n  ----- meta-TRAIN (in-distribution sanity check) -----")
    for k in METHODS:
        v = a_tr[k]
        print(f"    {k:14s}  {v['mean']:.4f}")
    print(f"\nSaved {args.out}")
    print("\n>> Headline interpretation: use meta-TEST numbers in MANUSCRIPT — "
          "the meta-train block is shown only as a sanity check.")


if __name__ == "__main__":
    sys.exit(main() or 0)
