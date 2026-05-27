"""Train the Layer 3 PPO chip-selection policy + evaluate vs baselines.

Pipeline:
  1. For each hard region, run its leave-one-out base model over the pool +
     test chips, cache per-chip probability maps + labels (CPU; one-off).
  2. Build per-chip features (prob mean/std, predicted-water fraction, entropy).
  3. Train a PPO policy (geodisaster.dispatch.rl_policy) on the
     threshold-calibration MDP, sampling regions/episodes.
  4. Evaluate the trained policy (greedy) vs random / uncertainty / full-pool
     threshold calibration, on held-out test F1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, ".")
from geodisaster.data.tile import merge_manifests
from geodisaster.datasets import DisasterPatchDataModule, stats_with_fallbacks
from geodisaster.train import DisasterSegLightningModule
from geodisaster.dispatch.rl_policy import (
    ChipCalibEnv, train_ppo, rollout_greedy, best_threshold_f1, f1_at_threshold,
)
from geodisaster.utils.logging import get_logger, setup_logging

log = get_logger("layer3_ppo")

HARD_REGIONS = {
    "Pakistan":  "outputs/leave_one_region_out/test_Pakistan/checkpoints",
    "Somalia":   "outputs/leave_one_region_out/test_Somalia/checkpoints",
    "Paraguay":  "outputs/leave_one_region_out/test_Paraguay/checkpoints",
    "India":     "outputs/leave_one_region_out/test_India/checkpoints",
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


def cache_region(region: str, ckpt: Path, patch_root: str, stats: str, device):
    """Return dict with per-chip flat (prob, label) for pool+test + features."""
    import random
    module, sources = load_module(ckpt)
    module = module.to(device)
    patches = merge_manifests([Path(patch_root) / f"sen1floods11_{region}"])
    rng = random.Random(1234); rng.shuffle(patches)
    n_test = max(8, len(patches) // 3)
    test_p, pool_p = patches[:n_test], patches[n_test:]
    norm = stats_with_fallbacks(stats, sources)

    def _probs(plist):
        out_prob, out_lab = [], []
        dm = DisasterPatchDataModule(train_patches=[], val_patches=[], test_patches=plist,
                                     sources=sources, batch_size=1, num_workers=0, normalize=norm)
        with torch.no_grad():
            for b in dm.test_dataloader():
                for k, v in list(b.items()):
                    if isinstance(v, torch.Tensor):
                        b[k] = v.to(device)
                logits = module(b)
                pr = torch.sigmoid(logits.squeeze(1)).cpu().numpy().ravel()
                lab = b["mask"].cpu().numpy().ravel()
                valid = lab != 255
                out_prob.append(pr[valid]); out_lab.append(lab[valid])
        return out_prob, out_lab

    pool_prob, pool_lab = _probs(pool_p)
    test_prob, test_lab = _probs(test_p)

    # per-chip features
    feats = []
    for pr in pool_prob:
        ent = -(np.clip(pr, 1e-6, 1 - 1e-6) * np.log(np.clip(pr, 1e-6, 1 - 1e-6))
                + (1 - pr) * np.log(np.clip(1 - pr, 1e-6, 1 - 1e-6)))
        feats.append([pr.mean(), pr.std(), float((pr > 0.5).mean()),
                      float(ent.mean()), float(ent.std())])
    feats = np.array(feats, dtype=np.float32)
    # normalise features
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)

    return {
        "pool_probs": pool_prob, "pool_labels": pool_lab,
        "test_probs": np.concatenate(test_prob), "test_labels": np.concatenate(test_lab),
        "chip_features": feats, "n_pool": len(pool_p), "n_test": len(test_p),
    }


def eval_baselines(cache, budget, seeds=(0, 1, 2)):
    """random, uncertainty, full-pool threshold calibration → test F1."""
    pool_probs, pool_labels = cache["pool_probs"], cache["pool_labels"]
    test_probs, test_labels = cache["test_probs"], cache["test_labels"]
    base = f1_at_threshold(test_probs, test_labels, 0.5)

    def _cal(idxs):
        pr = np.concatenate([pool_probs[i] for i in idxs])
        lb = np.concatenate([pool_labels[i] for i in idxs])
        t, _ = best_threshold_f1(pr, lb)
        return f1_at_threshold(test_probs, test_labels, t)

    import random, statistics
    n = len(pool_probs)
    # random
    rnd = []
    for s in seeds:
        r = random.Random(s)
        rnd.append(_cal(r.sample(range(n), min(budget, n))))
    # uncertainty: highest mean entropy (feature index 3)
    unc_order = np.argsort(cache["chip_features"][:, 3])[::-1]
    unc = _cal(list(unc_order[:budget]))
    # full pool (oracle upper bound for calibration)
    full = _cal(list(range(n)))
    return {"base": base, "random": statistics.mean(rnd),
            "random_std": statistics.stdev(rnd) if len(rnd) > 1 else 0,
            "uncertainty": unc, "full_pool": full}


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--patch-root", default="data/processed/patches")
    p.add_argument("--stats", default="data/processed/norm_stats_sen1floods11.yaml")
    p.add_argument("--budget", type=int, default=4)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--out", default="outputs/layer3_ppo")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log.info("device", device=str(device))

    # ---- cache regions ----
    caches = {}
    for region, ckdir in HARD_REGIONS.items():
        ck = _latest_ckpt(ckdir)
        if ck is None:
            log.warning("no_ckpt", region=region); continue
        log.info("caching", region=region, ckpt=ck.name)
        caches[region] = cache_region(region, ck, args.patch_root, args.stats, device)
    if not caches:
        log.error("no_caches"); return 1
    feat_dim = next(iter(caches.values()))["chip_features"].shape[1] + 2  # +mask +budget

    # ---- PPO training: sample a region each episode ----
    import random
    regions = list(caches)

    def make_env():
        r = random.choice(regions)
        c = caches[r]
        return ChipCalibEnv(c["pool_probs"], c["pool_labels"],
                            c["test_probs"], c["test_labels"],
                            c["chip_features"], budget=args.budget)

    log.info("training_ppo", updates=args.updates, regions=regions, budget=args.budget)
    policy, history = train_ppo(make_env, feat_dim, n_updates=args.updates,
                                episodes_per_update=8, lr=3e-3, seed=0)

    # ---- evaluate trained policy vs baselines per region ----
    results = {"budget": args.budget, "regions": {}, "ppo_history": history}
    for region, c in caches.items():
        env = ChipCalibEnv(c["pool_probs"], c["pool_labels"], c["test_probs"],
                           c["test_labels"], c["chip_features"], budget=args.budget)
        ppo_roll = rollout_greedy(policy, env)
        base = eval_baselines(c, args.budget)
        results["regions"][region] = {
            "base_f1": base["base"],
            "random_f1": base["random"], "random_std": base["random_std"],
            "uncertainty_f1": base["uncertainty"],
            "ppo_f1": ppo_roll["f1"],
            "full_pool_f1": base["full_pool"],
            "ppo_selected": ppo_roll["selected"],
        }
        log.info("region_eval", region=region,
                 base=round(base["base"], 3), random=round(base["random"], 3),
                 uncertainty=round(base["uncertainty"], 3),
                 ppo=round(ppo_roll["f1"], 3), full=round(base["full_pool"], 3))

    # aggregate
    import statistics
    def _avg(key):
        return statistics.mean(r[key] for r in results["regions"].values())
    results["aggregate"] = {
        "base_f1": _avg("base_f1"), "random_f1": _avg("random_f1"),
        "uncertainty_f1": _avg("uncertainty_f1"), "ppo_f1": _avg("ppo_f1"),
        "full_pool_f1": _avg("full_pool_f1"),
    }
    (out / "ppo_results.json").write_text(json.dumps(results, indent=2))
    log.info("done", **{k: round(v, 4) for k, v in results["aggregate"].items()})
    print(json.dumps(results["aggregate"], indent=2))


if __name__ == "__main__":
    sys.exit(main() or 0)
