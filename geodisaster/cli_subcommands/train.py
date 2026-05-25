"""`geodisaster train` — train one model on one split."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..data.catalog import EventCatalog
from ..data import splits as split_mod
from ..data.tile import merge_manifests
from ..datasets import DisasterPatchDataModule, stats_with_fallbacks
from ..train import DisasterSegLightningModule, make_trainer
from ..utils.io import load_config, ensure_dir
from ..utils.logging import get_logger, setup_logging

log = get_logger("cli.train")


def _gather_patches(patch_root: Path, events) -> list[dict]:
    dirs = [patch_root / e.event_id for e in events if (patch_root / e.event_id).is_dir()]
    return merge_manifests(dirs)


def _model_sources(model_cfg) -> list[str]:
    """Default modality list per model family."""
    fam = model_cfg.get("family") or model_cfg.get("name")
    if fam == "alphaearth_head":
        srcs = ["alphaearth"]
        for k, c in (model_cfg.get("aux_channels", {}) or {}).items():
            if int(c) > 0:
                srcs.append(k)
        return srcs
    if fam == "multi_modal_fusion":
        srcs = ["alphaearth"] + list((model_cfg.get("aux_channels", {}) or {}).keys())
        return [s for s in srcs if int((model_cfg.get("aux_channels", {}) or {}).get(s, 1)) > 0 or s == "alphaearth"]
    # Image-only nets just take the channels listed in cfg.backbone.sources, or default to sentinel2
    return list(model_cfg.backbone.get("sources", ["sentinel2"]))


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser("geodisaster train")
    p.add_argument("--default-config", default="configs/default.yaml")
    p.add_argument("--model-config", required=True)
    p.add_argument("--data-config", default="configs/data/japan_flood.yaml")
    p.add_argument("--catalog", default="data/catalog/japan_events.yaml")
    p.add_argument("--patch-root", default="data/processed/patches")
    p.add_argument("--workdir", default=None)
    p.add_argument("--train-events", action="append", default=None)
    p.add_argument("--val-events", action="append", default=None)
    p.add_argument("--test-events", action="append", default=None)
    p.add_argument("--label-fraction", type=float, default=1.0,
                   help="Few-shot sub-sampling of training patches.")
    p.add_argument("--stats", default="data/processed/norm_stats.yaml",
                   help="Normalize-stats YAML (from `geodisaster compute-stats`). "
                        "Falls back to EMPIRICAL_FALLBACKS if file is absent.")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    defaults = load_config(args.default_config)
    model_cfg = load_config(args.model_config)
    data_cfg = load_config(args.data_config)
    workdir = ensure_dir(Path(args.workdir or f"{defaults.project.workdir}/{model_cfg.name}"))
    setup_logging(level=defaults.logging.level, log_file=workdir / "run.log")

    import pytorch_lightning as pl
    pl.seed_everything(args.seed if args.seed is not None else int(defaults.project.seed))

    catalog = EventCatalog.load(args.catalog)
    event_index = {e.event_id: e for e in catalog}
    train_events = args.train_events or list((data_cfg.get("scope", {}) or {}).get("events", []))
    val_events = args.val_events
    test_events = args.test_events or (val_events or train_events[-1:])
    if val_events is None:
        if len(train_events) > 1:
            val_events = train_events[-1:]
            train_events = train_events[:-1]
            log.warning("auto_split_val", note="last train event used as val; "
                        "pass --val-events explicitly for a clean split")
        else:
            val_events = train_events  # tiny-data smoke run
            log.warning("val_overlaps_train",
                        note="only one train event available; val == train. "
                        "Tile more events or pass --val-events to fix.")

    def _evts(ids):
        return [event_index[i] for i in ids if i in event_index]

    train_patches = _gather_patches(Path(args.patch_root), _evts(train_events))
    val_patches = _gather_patches(Path(args.patch_root), _evts(val_events))
    test_patches = _gather_patches(Path(args.patch_root), _evts(test_events))
    if args.label_fraction < 1.0:
        train_patches = split_mod.few_shot_subsample(train_patches, args.label_fraction,
                                                    seed=int(defaults.project.seed))
    log.info("split_sizes", train=len(train_patches), val=len(val_patches), test=len(test_patches))

    sources = _model_sources(model_cfg)
    norm = stats_with_fallbacks(args.stats, sources)
    log.info("normalize_stats", path=str(args.stats),
             stats={k: (round(v[0], 4), round(v[1], 4)) for k, v in norm.items()})
    dm = DisasterPatchDataModule(
        train_patches=train_patches, val_patches=val_patches, test_patches=test_patches,
        sources=sources,
        batch_size=int(defaults.train.batch_size),
        num_workers=int(defaults.train.num_workers),
        normalize=norm,
    )
    module = DisasterSegLightningModule(model_cfg=model_cfg, train_cfg=defaults.train, sources=sources)
    trainer = make_trainer(defaults.train, workdir=workdir)
    trainer.fit(module, datamodule=dm)
    trainer.test(module, datamodule=dm, ckpt_path="best")
    return 0
