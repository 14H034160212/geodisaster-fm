"""`geodisaster evaluate` — load a checkpoint and report test-set metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..data.catalog import EventCatalog
from ..data.tile import merge_manifests
from ..datasets import DisasterPatchDataModule
from ..metrics import BinaryConfusion, auprc, expected_calibration_error
from ..train import DisasterSegLightningModule
from ..utils.io import load_config
from ..utils.logging import get_logger

log = get_logger("cli.eval")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser("geodisaster evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--default-config", default="configs/default.yaml")
    p.add_argument("--patch-root", default="data/processed/patches")
    p.add_argument("--catalog", default="data/catalog/japan_events.yaml")
    p.add_argument("--test-events", action="append", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    defaults = load_config(args.default_config)
    catalog = EventCatalog.load(args.catalog)
    event_index = {e.event_id: e for e in catalog}
    test_events = [event_index[i] for i in args.test_events if i in event_index]
    test_patches = merge_manifests([Path(args.patch_root) / e.event_id for e in test_events])

    module = DisasterSegLightningModule.load_from_checkpoint(args.checkpoint)
    module.eval()
    sources = module.sources
    dm = DisasterPatchDataModule(
        train_patches=[], val_patches=[], test_patches=test_patches,
        sources=sources,
        batch_size=int(defaults.train.batch_size),
        num_workers=int(defaults.train.num_workers),
    )

    cm = BinaryConfusion()
    all_scores, all_targets = [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = module.to(device)
    with torch.no_grad():
        for batch in dm.test_dataloader():
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            logits = module(batch)
            scores = torch.sigmoid(logits.squeeze(1))
            preds = (scores > 0.5).long()
            cm.update(preds, batch["mask"])
            all_scores.append(scores.cpu())
            all_targets.append(batch["mask"].cpu())

    import torch as _t
    scores = _t.cat(all_scores) if all_scores else _t.zeros(0)
    targets = _t.cat(all_targets) if all_targets else _t.zeros(0)
    summary = cm.as_dict()
    summary["auprc"] = auprc(scores, targets) if scores.numel() else 0.0
    summary["ece"] = expected_calibration_error(scores, targets) if scores.numel() else 0.0
    summary["test_events"] = list(args.test_events)
    summary["checkpoint"] = args.checkpoint

    out = Path(args.out) if args.out else Path(args.checkpoint).with_suffix(".eval.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("eval_complete", **{k: v for k, v in summary.items() if isinstance(v, (int, float))})
    print(json.dumps(summary, indent=2))
    return 0
