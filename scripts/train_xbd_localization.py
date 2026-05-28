"""Multi-hazard building-localization on xBD: train on some hazards, test on a
held-out hazard — the cross-hazard analogue of our Sen1Floods11 leave-one-
region-out experiment.

Reuses the GeoDisaster-FM training stack (DisasterPatchDataModule + smp_unet)
on the patches written by convert_xbd_to_patches.py. Reports test F1/IoU/AUPRC
on the held-out hazard, and (optionally) loops over hazards for a full matrix.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from geodisaster.data.tile import merge_manifests
from geodisaster.datasets import DisasterPatchDataModule, stats_with_fallbacks
from geodisaster.train import DisasterSegLightningModule, make_trainer
from geodisaster.metrics import BinaryConfusion, auprc
from geodisaster.utils.io import load_config, ensure_dir
from geodisaster.utils.logging import get_logger, setup_logging

log = get_logger("xbd_loc")
PATCH_ROOT = Path("data/processed/patches")


def _patches(disasters):
    return merge_manifests([PATCH_ROOT / f"xbd_{d}" for d in disasters])


@torch.no_grad()
def evaluate(module, dm, device):
    module.eval().to(device)
    cm = BinaryConfusion(); ss, ts = [], []
    for batch in dm.test_dataloader():
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        logits = module(batch)
        scores = torch.sigmoid(logits.squeeze(1))
        cm.update((scores > 0.5).long(), batch["mask"])
        ss.append(scores.cpu()); ts.append(batch["mask"].cpu())
    m = cm.as_dict(); m["auprc"] = auprc(torch.cat(ss), torch.cat(ts))
    return m


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="+", required=True, help="train disasters")
    p.add_argument("--val", required=True, help="val disaster")
    p.add_argument("--test", required=True, help="held-out test disaster (hazard)")
    p.add_argument("--model-config", default="configs/model/unet_xbd.yaml")
    p.add_argument("--stats", default="data/processed/norm_stats_xbd.yaml")
    p.add_argument("--default-config", default="configs/default.yaml")
    p.add_argument("--workdir", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulate", type=int, default=2)
    p.add_argument("--val-cap", type=int, default=150)
    p.add_argument("--out", default="outputs/xbd_localization/results.json")
    args = p.parse_args()

    defaults = load_config(args.default_config)
    defaults.train.batch_size = args.batch_size
    defaults.train.accumulate_grad_batches = args.accumulate
    model_cfg = load_config(args.model_config)
    workdir = ensure_dir(args.workdir or f"outputs/xbd_localization/test_{args.test}")

    import pytorch_lightning as pl
    pl.seed_everything(int(defaults.project.seed))

    sources = ["optical"]
    norm = stats_with_fallbacks(args.stats, sources)
    train_p = _patches(args.train); val_p = _patches([args.val]); test_p = _patches([args.test])
    # cap val for speed/memory (a held-out disaster can have >1000 patches)
    import random as _rnd
    if len(val_p) > args.val_cap:
        val_p = _rnd.Random(0).sample(val_p, args.val_cap)
    log.info("split", n_train=len(train_p), n_val=len(val_p), n_test=len(test_p),
             train_disasters=args.train, test_hazard=args.test)

    dm = DisasterPatchDataModule(train_patches=train_p, val_patches=val_p, test_patches=test_p,
                                 sources=sources, batch_size=args.batch_size,
                                 num_workers=int(defaults.train.num_workers), normalize=norm)
    module = DisasterSegLightningModule(model_cfg=model_cfg, train_cfg=defaults.train, sources=sources)
    trainer = make_trainer(defaults.train, workdir=workdir)
    trainer.fit(module, datamodule=dm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = evaluate(module, dm, device)

    rec = {"test_hazard": args.test, "train_hazards": args.train, "val": args.val,
           "n_train": len(train_p), "n_test": len(test_p),
           **{k: float(v) for k, v in m.items()}}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    # append to a list keyed by test hazard
    existing = json.loads(out.read_text()) if out.exists() else []
    existing = [r for r in existing if r.get("test_hazard") != args.test]
    existing.append(rec)
    out.write_text(json.dumps(existing, indent=2))
    print(f"\n=== xBD localization | test hazard = {args.test} ===")
    for k in ["f1", "iou", "precision", "recall", "auprc"]:
        print(f"  {k:10s} {m[k]:.4f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
