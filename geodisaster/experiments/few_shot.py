"""Few-shot label-fraction sweep (proposal §8).

For each label fraction in cfg.label_fractions, train the same model
``cfg.repeats`` times with different sub-samples and record val/test metrics.
The output is one row per (fraction, repeat) — pandas can pivot it into the
proposal §9 Figure 3 curve.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from ..data.catalog import EventCatalog
from ..data import splits as split_mod
from ..data.tile import merge_manifests
from ..datasets import DisasterPatchDataModule, stats_with_fallbacks
from ..train import DisasterSegLightningModule, make_trainer
from ..utils.io import ensure_dir
from ..utils.logging import get_logger

log = get_logger("exp.few_shot")


def _gather(patch_root: Path, event_ids) -> list[dict]:
    return merge_manifests([patch_root / e for e in event_ids if (patch_root / e).is_dir()])


def run_few_shot(
    defaults: DictConfig,
    model_cfg: DictConfig,
    experiment_cfg: DictConfig,
    catalog: EventCatalog,
    patch_root: Path,
    workdir: Path,
    stats_path: str | Path | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    import pytorch_lightning as pl

    # Base seed: explicit --seed overrides the config default so that
    # multi-seed sweeps actually differ (config seed alone is constant).
    base_seed = int(seed) if seed is not None else int(defaults.project.seed)

    fractions = list(experiment_cfg.get("label_fractions", [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]))
    repeats = int(experiment_cfg.get("repeats", 3))
    splits_cfg = experiment_cfg.get("splits", {})
    train_ids = list(splits_cfg.get("train_events", []))
    val_ids = list(splits_cfg.get("val_events", []))
    test_ids = list(splits_cfg.get("test_events", val_ids))

    workdir = ensure_dir(workdir)
    train_patches_full = _gather(patch_root, train_ids)
    val_patches = _gather(patch_root, val_ids)
    test_patches = _gather(patch_root, test_ids)

    sources = ["alphaearth"]
    if (model_cfg.get("family") or model_cfg.get("name")) in {"alphaearth_head", "multi_modal_fusion"}:
        sources += list((model_cfg.get("aux_channels", {}) or {}).keys())
    else:
        sources = list(model_cfg.backbone.get("sources", sources))

    rows: list[dict[str, Any]] = []
    for frac in fractions:
        for rep in range(repeats):
            pl.seed_everything(base_seed + rep)
            sub = split_mod.few_shot_subsample(
                train_patches_full, fraction=frac,
                seed=base_seed + rep,
                stratify_by_pos=True,
            )
            log.info("few_shot_run", fraction=frac, repeat=rep,
                     n_train=len(sub), n_val=len(val_patches), n_test=len(test_patches))
            norm = stats_with_fallbacks(stats_path, sources)
            dm = DisasterPatchDataModule(
                train_patches=sub, val_patches=val_patches, test_patches=test_patches,
                sources=sources,
                batch_size=int(defaults.train.batch_size),
                num_workers=int(defaults.train.num_workers),
                normalize=norm,
            )
            module = DisasterSegLightningModule(
                model_cfg=model_cfg, train_cfg=defaults.train, sources=sources,
            )
            run_workdir = ensure_dir(workdir / f"frac{frac}_rep{rep}")
            trainer = make_trainer(defaults.train, workdir=run_workdir)
            trainer.fit(module, datamodule=dm)
            test_out = trainer.test(module, datamodule=dm, ckpt_path="best")
            metrics = test_out[0] if test_out else {}
            row = {"label_fraction": frac, "repeat": rep, "n_train": len(sub),
                   "seed": base_seed + rep}
            row.update({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(workdir / "few_shot_results.csv", index=False)
    (workdir / "few_shot_summary.json").write_text(
        json.dumps(df.groupby("label_fraction").mean(numeric_only=True).to_dict(), indent=2),
        encoding="utf-8",
    )
    log.info("few_shot_done", rows=len(rows), out=str(workdir / "few_shot_results.csv"))
    return df
