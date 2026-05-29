"""A/B test of pixel-F1 reward vs decision-level (area-error) reward for the
PPO active-calibration policy.

Same 10-seed paired protocol as eval_layer3_ppo_significance.py, with two arms:
  - "pixel"          : reward = test pixel F1 gain (the standard signal).
  - "decision_area"  : reward = decrease in mean absolute relative AREA error
                       across test chips (a decision-level objective).
For each seed and region we train both policies on the SAME pool/test split,
do a greedy rollout, and evaluate BOTH metrics (pixel F1 and decision area
error). Paired t-test + Wilcoxon on per-seed differences.
"""
from __future__ import annotations
import argparse, json, pickle, statistics, sys
from pathlib import Path
import numpy as np
import torch
from scipy import stats as sps
sys.path.insert(0, ".")
from geodisaster.dispatch.rl_policy import (ChipCalibEnv, train_ppo, rollout_greedy,
                                            best_threshold_f1, f1_at_threshold)
from geodisaster.utils.logging import get_logger, setup_logging
log = get_logger("ppo_ab")
HARD = ["Pakistan", "Somalia", "Paraguay", "India"]


def split_with_per_chip(cache, seed, budget, n_rand=5, test_cap=2_000_000):
    """Like the existing split_eval but ALSO returns per-chip test arrays for
    decision-level metrics. Plus baseline error/F1."""
    rng = np.random.RandomState(seed)
    n = cache["n"]
    order = rng.permutation(n)
    n_test = max(8, n // 3)
    test_idx = order[:n_test]; pool_idx = order[n_test:]

    probs, labels = cache["probs"], cache["labels"]
    test_pr_chips = [probs[i] for i in test_idx]
    test_lb_chips = [labels[i] for i in test_idx]
    test_pr = np.concatenate(test_pr_chips); test_lb = np.concatenate(test_lb_chips)
    if test_pr.size > test_cap:
        sel = rng.choice(test_pr.size, test_cap, replace=False)
        test_pr_flat, test_lb_flat = test_pr[sel], test_lb[sel]
    else:
        test_pr_flat, test_lb_flat = test_pr, test_lb

    pool_pr = [probs[i] for i in pool_idx]; pool_lb = [labels[i] for i in pool_idx]
    feats_raw = cache["feats_raw"][pool_idx]
    feats = (feats_raw - feats_raw.mean(0)) / (feats_raw.std(0) + 1e-6)

    def make_env(reward_mode="pixel"):
        return ChipCalibEnv(pool_pr, pool_lb, test_pr_flat, test_lb_flat, feats,
                            budget=budget, reward_mode=reward_mode,
                            test_probs_per_chip=test_pr_chips,
                            test_labels_per_chip=test_lb_chips)

    base_f1 = f1_at_threshold(test_pr_flat, test_lb_flat, 0.5)
    env_for_baselines = make_env("decision_area")
    base_decision_err = env_for_baselines.base_decision_error
    return make_env, base_f1, base_decision_err


def eval_policy(policy, builders_region):
    """Greedy rollout per region; return per-region (pixel_F1, decision_err)."""
    out = []
    for env_b in builders_region.values():
        env = env_b("pixel")               # mode doesn't matter for rollout
        roll = rollout_greedy(policy, env)
        # rollout's env is fresh; recompute the two metrics on the same split
        thr_env = env_b("decision_area")   # gives access to per-chip arrays
        thr_env.selected = list(roll["selected"])
        f1 = thr_env._calibrated_test_f1()
        derr = thr_env._calibrated_decision_error()
        out.append((f1, derr))
    return out


def paired(diffs):
    d = np.asarray(diffs, float); n = len(d); m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    tcrit = float(sps.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    ci = (m - tcrit * se, m + tcrit * se)
    t_p = float(sps.ttest_1samp(d, 0.0).pvalue) if n > 1 and d.std() > 0 else 1.0
    try:
        w_p = float(sps.wilcoxon(d).pvalue) if n > 1 and np.any(d != 0) else 1.0
    except ValueError:
        w_p = 1.0
    return {"mean": m, "ci95": list(ci), "t_p": t_p, "wilcoxon_p": w_p, "n": n}


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="chip_cache.pkl (U-Net or AE)")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--budget", type=int, default=4)
    args = p.parse_args()

    log.info("loading_cache", path=args.cache)
    caches = pickle.loads(Path(args.cache).read_bytes())
    regions = list(caches)
    feat_dim = 5 + 2

    # per-seed per-arm region-mean of each metric
    rec = {arm: {"pixel_f1": [], "dec_err": []} for arm in ["pixel", "decision_area"]}
    base_f1s, base_errs = [], []

    import random as _random
    for s in range(args.seeds):
        builders = {}
        bf1_r, berr_r = [], []
        for r in regions:
            mk, bf1, berr = split_with_per_chip(caches[r], seed=1000 + s, budget=args.budget)
            builders[r] = mk; bf1_r.append(bf1); berr_r.append(berr)
        base_f1s.append(float(np.mean(bf1_r))); base_errs.append(float(np.mean(berr_r)))

        _rng = _random.Random(s)
        for arm in ["pixel", "decision_area"]:
            def make_env(arm=arm):
                return builders[_rng.choice(regions)](arm)
            policy, _ = train_ppo(make_env, feat_dim, n_updates=args.updates,
                                  episodes_per_update=8, lr=3e-3, seed=s, log_every=10_000)
            per_region = eval_policy(policy, builders)
            f1s = [x[0] for x in per_region]; errs = [x[1] for x in per_region]
            rec[arm]["pixel_f1"].append(float(np.mean(f1s)))
            rec[arm]["dec_err"].append(float(np.mean(errs)))
            log.info("seed_arm", seed=s, arm=arm,
                     pixel_f1=round(np.mean(f1s), 4), dec_err=round(np.mean(errs), 4))

    def stats(xs):
        a = np.asarray(xs, float); n = len(a)
        m = float(a.mean()); s = float(a.std(ddof=1)) if n > 1 else 0.0
        tc = float(sps.t.ppf(0.975, n - 1)) if n > 1 else 0.0
        return {"mean": m, "ci95": [m - tc * s / np.sqrt(n), m + tc * s / np.sqrt(n)],
                "std": s}

    out = {
        "n_seeds": args.seeds, "budget": args.budget,
        "base": {"pixel_f1": stats(base_f1s), "dec_err": stats(base_errs)},
        "pixel_arm": {"pixel_f1": stats(rec["pixel"]["pixel_f1"]),
                      "dec_err":  stats(rec["pixel"]["dec_err"])},
        "decision_arm": {"pixel_f1": stats(rec["decision_area"]["pixel_f1"]),
                         "dec_err":  stats(rec["decision_area"]["dec_err"])},
        "paired": {
            "dec_arm_vs_pix_arm_on_dec_err":  # negative = decision-reward better
                paired(np.asarray(rec["decision_area"]["dec_err"]) -
                       np.asarray(rec["pixel"]["dec_err"])),
            "dec_arm_vs_pix_arm_on_pixel_f1": # positive = decision-reward better
                paired(np.asarray(rec["decision_area"]["pixel_f1"]) -
                       np.asarray(rec["pixel"]["pixel_f1"])),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\n=== PPO A/B: pixel vs decision-area reward ===")
    print(f"  baseline                  pixel_F1={out['base']['pixel_f1']['mean']:.4f}  dec_err={out['base']['dec_err']['mean']:.4f}")
    print(f"  pixel-reward PPO          pixel_F1={out['pixel_arm']['pixel_f1']['mean']:.4f}  dec_err={out['pixel_arm']['dec_err']['mean']:.4f}")
    print(f"  decision-reward PPO       pixel_F1={out['decision_arm']['pixel_f1']['mean']:.4f}  dec_err={out['decision_arm']['dec_err']['mean']:.4f}")
    for k, v in out["paired"].items():
        sig = "SIG" if (v["ci95"][0] > 0 or v["ci95"][1] < 0) else "n.s."
        print(f"  {k:38s} d={v['mean']:+.4f}  CI={v['ci95']}  t-p={v['t_p']:.3f}  {sig}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
