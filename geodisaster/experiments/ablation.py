"""Ablation: AE only / SAR only / DEM only / AE+SAR / AE+SAR+DEM / +aux.

Each configuration trains the same backbone family with a different subset of
``aux_channels`` (and possibly a different primary modality). Results feed
proposal §8 ablation table + §9 Extended Data.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..data.catalog import EventCatalog
from ..data.tile import merge_manifests
from ..datasets import DisasterPatchDataModule
from ..train import DisasterSegLightningModule, make_trainer
from ..utils.io import ensure_dir
from ..utils.logging import get_logger

log = get_logger("exp.ablation")


# Variant keys MUST match patch source names emitted by tile-dataset
# (sentinel1, sentinel2, dem). 0 means modality is excluded.
DEFAULT_VARIANTS: dict[str, dict[str, Any]] = {
    "ae_only":         {"sentinel1": 0, "dem": 0, "sentinel2": 0},
    "sar_only":        {"primary": "sentinel1", "sentinel1": 2, "dem": 0, "sentinel2": 0},
    "dem_only":        {"primary": "dem",       "sentinel1": 0, "dem": 4, "sentinel2": 0},
    "ae_sar":          {"sentinel1": 2, "dem": 0, "sentinel2": 0},
    "ae_sar_dem":      {"sentinel1": 2, "dem": 4, "sentinel2": 0},
    "ae_sar_dem_aux":  {"sentinel1": 2, "dem": 4, "sentinel2": 6},
}


def _apply_variant(base: DictConfig, variant: dict[str, Any]) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    aux = cfg.get("aux_channels", {})
    for k in ("sentinel1", "dem", "sentinel2"):
        if k in variant:
            aux[k] = int(variant[k])
    cfg["aux_channels"] = aux
    return cfg


def run_ablation(
    defaults: DictConfig,
    model_cfg: DictConfig,
    catalog: EventCatalog,
    patch_root: Path,
    workdir: Path,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    variants: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    variants = variants or DEFAULT_VARIANTS
    workdir = ensure_dir(workdir)
    rows: list[dict[str, Any]] = []
    train_patches = merge_manifests([patch_root / e for e in train_ids if (patch_root / e).is_dir()])
    val_patches   = merge_manifests([patch_root / e for e in val_ids if (patch_root / e).is_dir()])
    test_patches  = merge_manifests([patch_root / e for e in test_ids if (patch_root / e).is_dir()])

    for name, variant in variants.items():
        cfg = _apply_variant(model_cfg, variant)
        sources = ["alphaearth"] + [k for k, v in (cfg.get("aux_channels", {}) or {}).items() if int(v) > 0]
        log.info("ablation_run", variant=name, sources=sources)
        dm = DisasterPatchDataModule(
            train_patches=train_patches, val_patches=val_patches, test_patches=test_patches,
            sources=sources,
            batch_size=int(defaults.train.batch_size),
            num_workers=int(defaults.train.num_workers),
        )
        module = DisasterSegLightningModule(model_cfg=cfg, train_cfg=defaults.train, sources=sources)
        run_dir = ensure_dir(workdir / name)
        trainer = make_trainer(defaults.train, workdir=run_dir)
        trainer.fit(module, datamodule=dm)
        res = trainer.test(module, datamodule=dm, ckpt_path="best")
        metrics = {k: float(v) for k, v in (res[0] if res else {}).items() if isinstance(v, (int, float))}
        rows.append({"variant": name, **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(workdir / "ablation_results.csv", index=False)
    log.info("ablation_done", rows=len(rows))
    return df
