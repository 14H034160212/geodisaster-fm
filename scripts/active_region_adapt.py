"""Layer 3 prototype — active region adaptation environment.

The cross-region matrix showed Pakistan is the hard hold-out (F1 ~0.56).
This script asks the Layer-3 question directly: given a small budget of
in-region labels, how fast does the gap close, and does *smart* chip
selection beat random?

This is the environment a reinforcement-learning policy would act in:
  state  = base model predictions + uncertainty on the unlabelled pool
  action = pick the next chip to label
  reward = F1 gain on the held-out test set after fine-tuning

Here we implement the environment plus two non-RL baseline policies
(random, uncertainty/entropy sampling) and plot the learning curves.
The trained RL policy is the next step; the environment + baselines are
the foundation and already answer "do in-region labels close the gap?".
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, ".")
from geodisaster.data.tile import merge_manifests
from geodisaster.datasets import DisasterPatchDataModule, stats_with_fallbacks
from geodisaster.train import DisasterSegLightningModule, make_trainer
from geodisaster.metrics import BinaryConfusion, auprc
from geodisaster.utils.logging import get_logger, setup_logging

log = get_logger("active_adapt")


def load_base_module(ckpt_path: Path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = state["hyper_parameters"]
    model_cfg = OmegaConf.create(hp.get("model_cfg", hp.get("model")))
    train_cfg = OmegaConf.create(hp.get("train_cfg", hp.get("train")))
    sources = hp["sources"]
    module = DisasterSegLightningModule(model_cfg, train_cfg, sources)
    module.load_state_dict(state["state_dict"], strict=True)
    return module, model_cfg, train_cfg, sources


def evaluate(module, patches, sources, stats, device, batch_size=8):
    norm = stats_with_fallbacks(stats, sources)
    dm = DisasterPatchDataModule(train_patches=[], val_patches=[],
                                 test_patches=patches, sources=sources,
                                 batch_size=batch_size, num_workers=4, normalize=norm)
    module = module.to(device).eval()
    cm = BinaryConfusion(); ss = []; tt = []
    with torch.no_grad():
        for b in dm.test_dataloader():
            for k, v in list(b.items()):
                if isinstance(v, torch.Tensor):
                    b[k] = v.to(device)
            logits = module(b)
            sc = torch.sigmoid(logits.squeeze(1)); pr = (sc > 0.5).long()
            cm.update(pr, b["mask"]); ss.append(sc.cpu()); tt.append(b["mask"].cpu())
    m = cm.as_dict()
    if ss:
        m["auprc"] = auprc(torch.cat(ss), torch.cat(tt))
    return m


def chip_uncertainty(module, patches, sources, stats, device):
    """Mean prediction entropy per chip — higher = more uncertain."""
    norm = stats_with_fallbacks(stats, sources)
    scores = []
    module = module.to(device).eval()
    for p in patches:
        dm = DisasterPatchDataModule(train_patches=[], val_patches=[],
                                     test_patches=[p], sources=sources,
                                     batch_size=1, num_workers=0, normalize=norm)
        with torch.no_grad():
            for b in dm.test_dataloader():
                for k, v in list(b.items()):
                    if isinstance(v, torch.Tensor):
                        b[k] = v.to(device)
                logits = module(b)
                prob = torch.sigmoid(logits.squeeze(1)).clamp(1e-6, 1 - 1e-6)
                ent = -(prob * prob.log() + (1 - prob) * (1 - prob).log())
                scores.append(float(ent.mean().item()))
    return scores


def finetune(base_ckpt, train_patches, sources, stats, device, epochs=20):
    """Fresh-load base, fine-tune on train_patches, return the module."""
    module, model_cfg, train_cfg, _ = load_base_module(base_ckpt)
    tc = OmegaConf.create(OmegaConf.to_container(train_cfg, resolve=True))
    tc["epochs"] = epochs
    tc["lr"] = 5e-5            # low LR for fine-tuning
    tc["early_stopping_patience"] = epochs  # no early stop on tiny sets
    norm = stats_with_fallbacks(stats, sources)
    # use the few train patches also as val (tiny-data regime)
    dm = DisasterPatchDataModule(train_patches=train_patches,
                                 val_patches=train_patches,
                                 test_patches=train_patches,
                                 sources=sources, batch_size=min(8, len(train_patches)),
                                 num_workers=2, normalize=norm, augment_train=True)
    module.train_cfg = tc
    import pytorch_lightning as pl
    trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu" if device.type == "cuda" else "cpu",
                         devices=1, enable_checkpointing=False, logger=False,
                         enable_progress_bar=False, num_sanity_val_steps=0)
    trainer.fit(module, datamodule=dm)
    return module


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="Pakistan")
    p.add_argument("--base-ckpt",
                   default="outputs/leave_one_region_out/test_Pakistan/checkpoints/best-epoch018.ckpt")
    p.add_argument("--patch-root", default="data/processed/patches")
    p.add_argument("--stats", default="data/processed/norm_stats_sen1floods11.yaml")
    p.add_argument("--budgets", default="1,2,4,8,16")
    p.add_argument("--random-seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--out", default="outputs/active_adapt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in args.budgets.split(",")]
    rseeds = [int(x) for x in args.random_seeds.split(",")]

    # ---- split region chips into adapt-pool + test ----
    region_dir = Path(args.patch_root) / f"sen1floods11_{args.region}"
    all_patches = merge_manifests([region_dir])
    rng = random.Random(1234)
    rng.shuffle(all_patches)
    n_test = max(8, len(all_patches) // 3)
    test_patches = all_patches[:n_test]
    pool_patches = all_patches[n_test:]
    log.info("split", region=args.region, total=len(all_patches),
             pool=len(pool_patches), test=len(test_patches))

    module0, _, _, sources = load_base_module(Path(args.base_ckpt))
    results = {"region": args.region, "n_pool": len(pool_patches),
               "n_test": len(test_patches), "budgets": budgets, "curves": {}}

    # ---- 0-label baseline (zero-shot from other regions) ----
    m0 = evaluate(module0, test_patches, sources, args.stats, device)
    log.info("zero_shot_baseline", f1=round(m0["f1"], 4), iou=round(m0["iou"], 4))
    results["zero_shot_f1"] = m0["f1"]
    results["zero_shot_iou"] = m0["iou"]

    # ---- uncertainty ranking of pool (computed once on base model) ----
    unc = chip_uncertainty(module0, pool_patches, sources, args.stats, device)
    unc_order = [pool_patches[i] for i in np.argsort(unc)[::-1]]  # high → low

    # ---- strategy: uncertainty sampling ----
    results["curves"]["uncertainty"] = []
    for k in budgets:
        sel = unc_order[:k]
        ft = finetune(Path(args.base_ckpt), sel, sources, args.stats, device, args.epochs)
        m = evaluate(ft, test_patches, sources, args.stats, device)
        log.info("uncertainty", k=k, f1=round(m["f1"], 4))
        results["curves"]["uncertainty"].append({"k": k, "f1": m["f1"], "iou": m["iou"]})
        del ft; torch.cuda.empty_cache()

    # ---- strategy: random (multi-seed) ----
    results["curves"]["random"] = []
    for k in budgets:
        f1s, ious = [], []
        for s in rseeds:
            r = random.Random(s)
            sel = r.sample(pool_patches, min(k, len(pool_patches)))
            ft = finetune(Path(args.base_ckpt), sel, sources, args.stats, device, args.epochs)
            m = evaluate(ft, test_patches, sources, args.stats, device)
            f1s.append(m["f1"]); ious.append(m["iou"])
            del ft; torch.cuda.empty_cache()
        import statistics
        results["curves"]["random"].append({
            "k": k,
            "f1": statistics.mean(f1s),
            "f1_std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
            "iou": statistics.mean(ious),
        })
        log.info("random", k=k, f1=round(statistics.mean(f1s), 4),
                 std=round(statistics.stdev(f1s) if len(f1s) > 1 else 0, 4))

    (out / f"adapt_{args.region}.json").write_text(json.dumps(results, indent=2))
    log.info("done", out=str(out / f"adapt_{args.region}.json"))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
