"""`geodisaster run-few-shot` — sweep label fractions."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..data.catalog import EventCatalog
from ..experiments import run_few_shot
from ..utils.io import load_config, ensure_dir


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser("geodisaster run-few-shot")
    p.add_argument("--default-config", default="configs/default.yaml")
    p.add_argument("--model-config", required=True)
    p.add_argument("--experiment-config", default="configs/experiment/few_shot.yaml")
    p.add_argument("--catalog", default="data/catalog/japan_events.yaml")
    p.add_argument("--patch-root", default="data/processed/patches")
    p.add_argument("--workdir", default=None)
    p.add_argument("--stats", default="data/processed/norm_stats.yaml")
    p.add_argument("--seed", type=int, default=None,
                   help="Override config base seed (so multi-seed sweeps differ).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override train batch size (lower to fit a busy GPU).")
    p.add_argument("--accumulate", type=int, default=None,
                   help="Override gradient accumulation (keep effective batch).")
    args = p.parse_args(argv)

    defaults = load_config(args.default_config)
    model_cfg = load_config(args.model_config)
    experiment_cfg = load_config(args.experiment_config)
    if args.batch_size is not None:
        defaults.train.batch_size = int(args.batch_size)
    if args.accumulate is not None:
        defaults.train.accumulate_grad_batches = int(args.accumulate)
    catalog = EventCatalog.load(args.catalog)
    workdir = ensure_dir(args.workdir or f"{defaults.project.workdir}/few_shot_{model_cfg.name}")
    run_few_shot(
        defaults=defaults,
        model_cfg=model_cfg,
        experiment_cfg=experiment_cfg,
        catalog=catalog,
        patch_root=Path(args.patch_root),
        workdir=Path(workdir),
        stats_path=args.stats,
        seed=args.seed,
    )
    return 0
