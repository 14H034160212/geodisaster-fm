"""Cross-domain generalization matrix runner.

Implements proposal §8 four-way evaluation:
    cross_region   train_filter -> test_filter (by region substring)
    cross_event    train_events -> test_events (held-out events)
    cross_hazard   train hazard -> test hazard (e.g. flood->landslide)
    global_to_japan train on external manifest -> test on Japan events
    temporal       train year -> test year

The output is a long-format CSV: one row per (mode, train_id, test_id) plus
metrics. ``viz/fig4_xdomain.py`` reshapes it into the heatmap from proposal §9.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from ..data.catalog import EventCatalog, HazardType
from ..data.tile import merge_manifests
from ..datasets import DisasterPatchDataModule, stats_with_fallbacks
from ..train import DisasterSegLightningModule, make_trainer
from ..utils.io import ensure_dir
from ..utils.logging import get_logger

__all__ = ["run_cross_domain"]

log = get_logger("exp.cross_domain")


def _patches_for(patch_root: Path, event_ids) -> list[dict]:
    return merge_manifests([patch_root / e for e in event_ids if (patch_root / e).is_dir()])


def _filter_events(catalog: EventCatalog, flt: dict) -> list[str]:
    if not flt:
        return [e.event_id for e in catalog]
    out = []
    for e in catalog:
        if "hazard" in flt and e.hazard != HazardType(flt["hazard"]):
            continue
        if "region_contains" in flt and flt["region_contains"].lower() not in e.region.lower():
            continue
        if "country" in flt and e.country != flt["country"]:
            continue
        out.append(e.event_id)
    return out


def _train_one(defaults, model_cfg, train_patches, val_patches, sources, workdir,
               stats_path=None):
    norm = stats_with_fallbacks(stats_path, sources)
    dm = DisasterPatchDataModule(
        train_patches=train_patches, val_patches=val_patches, test_patches=val_patches,
        sources=sources,
        batch_size=int(defaults.train.batch_size),
        num_workers=int(defaults.train.num_workers),
        normalize=norm,
    )
    module = DisasterSegLightningModule(model_cfg=model_cfg, train_cfg=defaults.train, sources=sources)
    trainer = make_trainer(defaults.train, workdir=workdir)
    trainer.fit(module, datamodule=dm)
    return module, trainer


def _test_on(trainer, module, test_patches, sources, defaults,
             stats_path=None) -> dict[str, float]:
    norm = stats_with_fallbacks(stats_path, sources)
    dm = DisasterPatchDataModule(
        train_patches=[], val_patches=[], test_patches=test_patches,
        sources=sources,
        batch_size=int(defaults.train.batch_size),
        num_workers=int(defaults.train.num_workers),
        normalize=norm,
    )
    res = trainer.test(module, datamodule=dm)
    return {k: float(v) for k, v in (res[0] if res else {}).items() if isinstance(v, (int, float))}


def run_cross_domain(
    defaults: DictConfig,
    model_cfg: DictConfig,
    experiment_cfg: DictConfig,
    catalog: EventCatalog,
    patch_root: Path,
    workdir: Path,
    stats_path: str | Path | None = None,
) -> pd.DataFrame:
    workdir = ensure_dir(workdir)
    rows: list[dict[str, Any]] = []

    sources = ["alphaearth"]
    fam = model_cfg.get("family") or model_cfg.get("name")
    if fam in {"alphaearth_head", "multi_modal_fusion"}:
        sources += list((model_cfg.get("aux_channels", {}) or {}).keys())

    modes = experiment_cfg.get("modes", {})
    for mode_name, mode_cfg in modes.items():
        log.info("cross_domain_mode", mode=mode_name)
        if mode_name == "cross_event":
            train_ids = list(mode_cfg.get("train_events", []))
            test_ids = list(mode_cfg.get("test_events", []))
        elif mode_name in {"cross_region", "cross_hazard"}:
            train_ids = _filter_events(catalog, mode_cfg.get("train_filter", {}))
            test_ids = _filter_events(catalog, mode_cfg.get("test_filter", {}))
        elif mode_name == "temporal":
            ty = mode_cfg.get("train_year")
            sy = mode_cfg.get("test_year")
            train_ids = [e.event_id for e in catalog if e.event_date and e.event_date.year == ty]
            test_ids = [e.event_id for e in catalog if e.event_date and e.event_date.year == sy]
        elif mode_name == "global_to_japan":
            log.warning("global_to_japan_uses_external_manifest_skipping_default")
            continue
        else:
            log.warning("unknown_cross_mode", mode=mode_name)
            continue

        if not train_ids or not test_ids:
            log.warning("empty_split", mode=mode_name, train=train_ids, test=test_ids)
            continue

        train_patches = _patches_for(patch_root, train_ids)
        # Use last 10% of train patches as val
        n_val = max(1, len(train_patches) // 10)
        val_patches = train_patches[-n_val:]
        train_patches = train_patches[:-n_val]
        test_patches = _patches_for(patch_root, test_ids)
        if not train_patches or not test_patches:
            continue

        mode_workdir = ensure_dir(workdir / mode_name)
        module, trainer = _train_one(
            defaults, model_cfg, train_patches, val_patches, sources, mode_workdir,
            stats_path=stats_path,
        )
        metrics = _test_on(trainer, module, test_patches, sources, defaults,
                           stats_path=stats_path)
        rows.append({
            "mode": mode_name,
            "train": ",".join(train_ids),
            "test":  ",".join(test_ids),
            **metrics,
        })

    df = pd.DataFrame(rows)
    df.to_csv(workdir / "cross_domain_results.csv", index=False)
    (workdir / "cross_domain_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    log.info("cross_domain_done", rows=len(rows))
    return df
