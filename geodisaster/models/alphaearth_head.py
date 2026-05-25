"""Main method: AlphaEarth embedding + lightweight head.

AlphaEarth annual embeddings live at 10 m, 64 dims. Because the embedding is
already a strong geospatial representation, we deliberately keep the head
small — the proposal §3 calls out that the value of the foundation model is
*not having to train a giant decoder*.

Heads:
    - mlp        : pointwise MLP applied per-pixel (no spatial context)
    - convhead   : 1x1 + 3x3 + 1x1 small conv stack (mild spatial context)
    - xgboost    : sklearn-compatible per-pixel classifier (CPU; for ablations)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from . import register


class _MLPHead(nn.Module):
    def __init__(self, in_channels: int, hidden: list[int], out_channels: int,
                 dropout: float = 0.2, activation: str = "gelu"):
        super().__init__()
        act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
        layers: list[nn.Module] = []
        prev = in_channels
        for h in hidden:
            layers += [nn.Conv2d(prev, h, 1), act(), nn.Dropout2d(dropout)]
            prev = h
        layers += [nn.Conv2d(prev, out_channels, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ConvHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AlphaEarthSegmentor(nn.Module):
    """Wraps an AlphaEarth-feature head with optional auxiliary inputs.

    Forward inputs are a dict ``{"alphaearth": Tensor[B, C_ae, H, W], ...}``.
    Auxiliary channels (Sentinel-1, DEM, Sentinel-2) are concatenated along the
    channel axis when their config blocks declare a non-zero size.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        bb = cfg.backbone
        head_cfg = cfg.head
        aux = cfg.get("aux_channels", {}) or {}

        self.input_keys = ["alphaearth"]
        in_c = int(bb.get("in_channels", 64))
        for k, c in aux.items():
            if int(c) > 0:
                self.input_keys.append(k)
                in_c += int(c)
        self.frozen_backbone = bool(bb.get("frozen", True))  # AE features are not a learnable module

        out_c = int(head_cfg.get("out_channels", 1))
        ht = head_cfg.get("type", "mlp")
        if ht == "mlp":
            self.head: nn.Module = _MLPHead(
                in_channels=in_c,
                hidden=list(head_cfg.get("hidden", [256, 128])),
                out_channels=out_c,
                dropout=float(head_cfg.get("dropout", 0.2)),
                activation=head_cfg.get("activation", "gelu"),
            )
        elif ht == "conv":
            self.head = _ConvHead(
                in_channels=in_c,
                hidden=int(head_cfg.get("hidden_dim", 128)),
                out_channels=out_c,
                dropout=float(head_cfg.get("dropout", 0.1)),
            )
        else:
            raise ValueError(f"unknown alphaearth head type: {ht}")

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        feats = [batch["alphaearth"]]
        ref_shape = batch["alphaearth"].shape[-2:]
        for k in self.input_keys[1:]:
            t = batch[k]
            if t.shape[-2:] != ref_shape:
                t = F.interpolate(t, size=ref_shape, mode="bilinear", align_corners=False)
            feats.append(t)
        x = torch.cat(feats, dim=1)
        return self.head(x)


@register("alphaearth_head")
def _build(cfg: DictConfig) -> nn.Module:
    return AlphaEarthSegmentor(cfg)


# ---------------------------------------------------------------------------
# XGBoost / sklearn classical heads (fit per-pixel; used in ablations §8)
# ---------------------------------------------------------------------------
class _SklearnPixelHead:
    """Thin adapter to run XGBoost / RF / LR on flattened pixel embeddings."""

    def __init__(self, kind: str = "xgboost", **kwargs):
        self.kind = kind
        self.model = None
        self.kwargs = kwargs

    def _make(self):
        if self.kind == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(**{"tree_method": "hist", **self.kwargs})
        if self.kind == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**self.kwargs)
        if self.kind == "lr":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**{"max_iter": 1000, **self.kwargs})
        raise ValueError(self.kind)

    def fit(self, X, y):
        self.model = self._make()
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


def build_pixel_classifier(cfg: DictConfig) -> _SklearnPixelHead:
    """Factory for AE + classical head; used only by ablation_xgboost runner."""
    kind = cfg.head.get("classical", "xgboost")
    return _SklearnPixelHead(kind=kind, **cfg.head.get("classical_kwargs", {}))
