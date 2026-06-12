"""Zero-label prior-correction baseline (Saerens-Latinne-Decaestecker EM).

The deepest referee question against the "four labels are enough" headline:
optimal-threshold drift under class-prior change is textbook label shift
(Elkan 2001; Saerens et al. 2002; Lipton et al. 2018 BBSE). Under pure label
shift there is a ZERO-label fix — estimate the new event's class prior from
the model's own unlabelled posterior outputs via EM, then adjust the decision
threshold analytically. If that recovers the same F1 as labelled
recalibration, the four labels are unnecessary; if it falls short, the gap
quantifies how much of the calibration drift is NOT pure label shift
(i.e. genuine score-distribution distortion that needs labels to see).

Method (per event, per seed, matching the LOEO-v2 20-seed splits exactly):
  1. Split pool/test with the same RandomState recipe as split_eval
     (seed = 1000 + s).
  2. Compute the model's training prior pi = mean positive rate over the
     OTHER nine events' labels (what the LOO model saw during training).
  3. Run Saerens EM on the POOL probabilities only — no labels touched:
        w1 = pi'_t / pi ;  w0 = (1 - pi'_t) / (1 - pi)
        p~(y=1|x) = p*w1 / (p*w1 + (1-p)*w0)
        pi'_{t+1} = mean(p~)            (iterate to convergence)
  4. The corrected posterior thresholded at 0.5 is equivalent to
     thresholding the original p at  tau_EM = w0 / (w0 + w1).
  5. Evaluate test F1 at tau_EM. Zero labels used.

Output: outputs/layer3_ppo/zero_label_prior_correction.json with per-event
per-seed F1 + pooled paired stats vs base / random / ppo / full_pool from
the 20-seed LOEO fold files.
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
import scipy.stats as sps
sys.path.insert(0, ".")
from geodisaster.dispatch.rl_policy import f1_at_threshold
from geodisaster.utils.logging import get_logger, setup_logging

log = get_logger("zero-label")

CACHE = "outputs/layer3_ppo/chip_cache_all10.pkl"
ALL = ["Ghana", "India", "Mekong", "Nigeria", "Pakistan",
       "Paraguay", "Somalia", "Spain", "Sri-Lanka", "USA"]
SEEDS = list(range(20))                      # match the 20-seed LOEO protocol
OUT = Path("outputs/layer3_ppo/zero_label_prior_correction.json")


def split_pool_test(cache: dict, seed: int):
    """Replicate split_eval's pool/test split + test subsample RNG stream."""
    rng = np.random.RandomState(seed)
    n = cache["n"]
    order = rng.permutation(n)
    n_test = max(8, n // 3)
    test_idx = order[:n_test]; pool_idx = order[n_test:]
    return pool_idx, test_idx, rng


def saerens_em(pool_probs: np.ndarray, train_prior: float,
               n_iter: int = 100, tol: float = 1e-6) -> float:
    """Return the EM estimate of the new-event positive prior."""
    p = np.clip(pool_probs, 1e-6, 1 - 1e-6)
    pi = float(np.clip(train_prior, 1e-6, 1 - 1e-6))
    pi_new = pi
    for _ in range(n_iter):
        w1 = pi_new / pi
        w0 = (1.0 - pi_new) / (1.0 - pi)
        post = p * w1 / (p * w1 + (1.0 - p) * w0)
        nxt = float(post.mean())
        if abs(nxt - pi_new) < tol:
            pi_new = nxt
            break
        pi_new = nxt
    return float(np.clip(pi_new, 1e-6, 1 - 1e-6))


def bbse_prior(pool_pred_pos_rate: float, tpr: float, fpr: float) -> float:
    """Black-Box Shift Estimation (Lipton et al. 2018), binary case.

    mu_hat = TPR * pi' + FPR * (1 - pi')  =>  pi' = (mu_hat - FPR) / (TPR - FPR)
    Uses SOURCE-domain confusion rates (labels from training events only)
    plus the TARGET pool's hard predicted-positive rate. Zero target labels.
    """
    denom = tpr - fpr
    if abs(denom) < 1e-6:
        return float("nan")
    return float(np.clip((pool_pred_pos_rate - fpr) / denom, 1e-6, 1 - 1e-6))


def main():
    setup_logging()
    caches = pickle.loads(Path(CACHE).read_bytes())

    # Training prior per LOO model = positive rate over the other 9 events
    train_prior = {}
    src_tpr, src_fpr = {}, {}
    for r in ALL:
        prs = [np.concatenate(caches[o]["probs"]) for o in ALL if o != r]
        labs = [np.concatenate(caches[o]["labels"]) for o in ALL if o != r]
        cat_pr = np.concatenate(prs); cat_lb = np.concatenate(labs)
        train_prior[r] = float((cat_lb == 1).mean())
        pred_pos = cat_pr >= 0.5
        pos = cat_lb == 1
        src_tpr[r] = float(pred_pos[pos].mean())
        src_fpr[r] = float(pred_pos[~pos].mean())
    log.info("train_priors", **{r: round(v, 4) for r, v in train_prior.items()})

    per_region = {}
    for region in ALL:
        c = caches[region]
        probs, labels = c["probs"], c["labels"]
        pi = train_prior[region]
        f1s, taus, priors = [], [], []
        f1s_bbse, taus_bbse, priors_bbse = [], [], []
        diag_mean_p, diag_true_prior = [], []
        for s in SEEDS:
            pool_idx, test_idx, rng = split_pool_test(c, 1000 + s)
            test_pr = np.concatenate([probs[i] for i in test_idx])
            test_lb = np.concatenate([labels[i] for i in test_idx])
            if test_pr.size > 2_000_000:
                sel = rng.choice(test_pr.size, 2_000_000, replace=False)
                test_pr, test_lb = test_pr[sel], test_lb[sel]
            pool_pr = np.concatenate([probs[i] for i in pool_idx])
            pool_lb = np.concatenate([labels[i] for i in pool_idx])  # DIAGNOSIS ONLY

            # ---- Saerens EM (soft posteriors) ----
            pi_new = saerens_em(pool_pr, pi)
            w1 = pi_new / pi
            w0 = (1.0 - pi_new) / (1.0 - pi)
            tau_em = w0 / (w0 + w1)
            f1s.append(float(f1_at_threshold(test_pr, test_lb, tau_em)))
            taus.append(float(tau_em)); priors.append(pi_new)

            # ---- BBSE (hard confusion rates) ----
            mu_hat = float((pool_pr >= 0.5).mean())
            pi_bb = bbse_prior(mu_hat, src_tpr[region], src_fpr[region])
            if np.isnan(pi_bb):
                pi_bb = pi
            w1b = pi_bb / pi
            w0b = (1.0 - pi_bb) / (1.0 - pi)
            tau_bb = w0b / (w0b + w1b)
            f1s_bbse.append(float(f1_at_threshold(test_pr, test_lb, tau_bb)))
            taus_bbse.append(float(tau_bb)); priors_bbse.append(pi_bb)

            # ---- diagnostics: how much TRUE prior shift is there? ----
            diag_mean_p.append(float(pool_pr.mean()))
            diag_true_prior.append(float((pool_lb == 1).mean()))

        per_region[region] = {
            "f1": f1s, "tau_em": taus, "pi_hat": priors,
            "f1_bbse": f1s_bbse, "tau_bbse": taus_bbse, "pi_hat_bbse": priors_bbse,
            "train_prior": pi, "src_tpr": src_tpr[region], "src_fpr": src_fpr[region],
            "diag_mean_pool_prob": float(np.mean(diag_mean_p)),
            "diag_true_pool_prior": float(np.mean(diag_true_prior)),
        }
        log.info("done_region", region=region,
                 em_f1=round(float(np.mean(f1s)), 4),
                 bbse_f1=round(float(np.mean(f1s_bbse)), 4),
                 true_prior=round(float(np.mean(diag_true_prior)), 4),
                 mean_p=round(float(np.mean(diag_mean_p)), 4))

    # Pull the matching 20-seed LOEO per-seed values for paired comparison
    pooled = {"em": [], "bbse": [], "base": [], "random": [], "uncertainty": [],
              "ppo": [], "full_pool": []}
    for r in ALL:
        fold = json.loads(Path(f"outputs/layer3_ppo/ppo_loeo_v2_20s_{r}.json").read_text())
        raw = fold["raw"]["meta_test"][r]
        n = min(len(per_region[r]["f1"]), len(raw["ppo"]))
        pooled["em"].extend(per_region[r]["f1"][:n])
        pooled["bbse"].extend(per_region[r]["f1_bbse"][:n])
        for k in ["base", "random", "uncertainty", "ppo", "full_pool"]:
            pooled[k].extend(raw[k][:n])

    def paired(d):
        d = np.asarray(d, float); n = len(d); m = float(d.mean())
        if n < 2 or d.std(ddof=1) == 0:
            return {"n": n, "mean": round(m, 4), "ci95": [m, m], "t_p": 1.0, "wilcoxon_p": 1.0}
        se = float(d.std(ddof=1) / np.sqrt(n))
        tc = float(sps.t.ppf(0.975, n - 1))
        t_p = float(sps.ttest_1samp(d, 0.0).pvalue)
        try:
            w_p = float(sps.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            w_p = 1.0
        return {"n": n, "mean": round(m, 4),
                "ci95": [round(m - tc * se, 4), round(m + tc * se, 4)],
                "t_p": round(t_p, 4), "wilcoxon_p": round(w_p, 4)}

    em = np.asarray(pooled["em"])
    bbse = np.asarray(pooled["bbse"])
    out = {
        "protocol": "Zero-label prior correction (Saerens EM + BBSE), 20-seed LOEO splits",
        "per_region": {r: {"mean_f1_em": round(float(np.mean(v["f1"])), 4),
                           "mean_tau_em": round(float(np.mean(v["tau_em"])), 3),
                           "mean_pi_hat_em": round(float(np.mean(v["pi_hat"])), 4),
                           "mean_f1_bbse": round(float(np.mean(v["f1_bbse"])), 4),
                           "mean_tau_bbse": round(float(np.mean(v["tau_bbse"])), 3),
                           "mean_pi_hat_bbse": round(float(np.mean(v["pi_hat_bbse"])), 4),
                           "train_prior": round(v["train_prior"], 4),
                           "src_tpr": round(v["src_tpr"], 4),
                           "src_fpr": round(v["src_fpr"], 4),
                           "diag_mean_pool_prob": round(v["diag_mean_pool_prob"], 4),
                           "diag_true_pool_prior": round(v["diag_true_pool_prior"], 4)}
                       for r, v in per_region.items()},
        "pooled_mean": {k: round(float(np.mean(vs)), 4) for k, vs in pooled.items()},
        "paired_em_minus": {k: paired(em - np.asarray(pooled[k]))
                            for k in ["base", "random", "uncertainty", "ppo", "full_pool"]},
        "paired_bbse_minus": {k: paired(bbse - np.asarray(pooled[k]))
                              for k in ["base", "random", "uncertainty", "ppo", "full_pool"]},
    }
    OUT.write_text(json.dumps(out, indent=2))

    print(f"\n=== Zero-label prior correction (n={len(em)} pairs) ===\n")
    print("Pooled F1:")
    for k in ["em", "bbse", "base", "random", "uncertainty", "ppo", "full_pool"]:
        print(f"  {k:12s} {out['pooled_mean'][k]:.4f}")
    print("\nPaired BBSE − comparator:")
    for k, v in out["paired_bbse_minus"].items():
        sig = "*" if v["t_p"] < 0.05 else " "
        print(f"  BBSE − {k:<12} Δ={v['mean']:+.4f}  95%CI{v['ci95']}  t-p={v['t_p']}{sig}")
    print(f"\nPer-region detail (diagnosis: true prior shift vs score distortion):")
    print(f"{'region':<10} {'train π':>8} {'true π_new':>10} {'mean p̂':>8} "
          f"{'π̂_EM':>7} {'π̂_BBSE':>8} {'F1_EM':>7} {'F1_BBSE':>8}")
    for r, v in out["per_region"].items():
        print(f"{r:<10} {v['train_prior']:>8.4f} {v['diag_true_pool_prior']:>10.4f} "
              f"{v['diag_mean_pool_prob']:>8.4f} {v['mean_pi_hat_em']:>7.4f} "
              f"{v['mean_pi_hat_bbse']:>8.4f} {v['mean_f1_em']:>7.4f} {v['mean_f1_bbse']:>8.4f}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    sys.exit(main() or 0)
