"""Model registry.

Every model module exposes ``build(cfg) -> nn.Module`` and is registered here.
The factory ``build_model(cfg)`` is the single entry point used by training
code so new baselines can be added without touching the trainer.
"""
from __future__ import annotations

from typing import Callable

import torch.nn as nn
from omegaconf import DictConfig

_REGISTRY: dict[str, Callable[[DictConfig], nn.Module]] = {}


def register(name: str):
    def deco(fn: Callable[[DictConfig], nn.Module]):
        _REGISTRY[name] = fn
        return fn
    return deco


def build_model(cfg: DictConfig) -> nn.Module:
    family = cfg.get("family") or cfg.get("name")
    if family not in _REGISTRY:
        raise KeyError(
            f"unknown model family '{family}'. registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[family](cfg)


# Import-side registration. Each module wraps a different external dep
# (smp / transformers / timm); missing deps for one model should NOT take down
# the entire registry. We log a warning and skip.
import importlib as _il
from .alphaearth_head import _build  # noqa: F401  always available (pure torch)

for _mod in (
    "geodisaster.models.unet",            # smp_unet, smp_deeplabv3plus
    "geodisaster.models.segformer",       # segformer
    "geodisaster.models.dinov2_adapter",  # dinov2_adapter
    "geodisaster.models.sam_adapter",     # sam_adapter
    "geodisaster.models.rsfm",            # satmae, prithvi, remoteclip, crossearth
    "geodisaster.models.fusion",          # multi_modal_fusion
):
    try:
        _il.import_module(_mod)
    except Exception as _e:  # pragma: no cover - depends on optional installs
        import warnings as _w
        _w.warn(f"{_mod} registration skipped: {_e}", RuntimeWarning, stacklevel=2)
