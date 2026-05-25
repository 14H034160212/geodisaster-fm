"""Multi-modal fusion: AlphaEarth + SAR + DEM + auxiliary priors.

This is the headline model that the proposal §3 calls out — the ``GeoDisaster-FM``
fusion. Each modality enters through its own short stem, features are concatenated,
and a shared U-Net-like decoder predicts the impact mask.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from . import register


def _stem(in_c: int, out_c: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.GroupNorm(8, out_c), nn.GELU(),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.GroupNorm(8, out_c), nn.GELU(),
    )


class MultiModalFusion(nn.Module):
    def __init__(self, modalities: dict[str, int], hidden: int = 128, num_classes: int = 1):
        super().__init__()
        # modalities: name -> in_channels (0 means absent)
        self.modalities = {k: int(v) for k, v in modalities.items() if int(v) > 0}
        self.stems = nn.ModuleDict({
            name: _stem(c, hidden) for name, c in self.modalities.items()
        })
        n = len(self.modalities)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * n, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Conv2d(hidden, num_classes, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ref = batch[next(iter(self.modalities))]
        H, W = ref.shape[-2:]
        feats = []
        for name in self.modalities:
            x = batch[name]
            if x.shape[-2:] != (H, W):
                x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
            feats.append(self.stems[name](x))
        z = self.fuse(torch.cat(feats, dim=1))
        return self.head(z)


@register("multi_modal_fusion")
def _build(cfg: DictConfig) -> nn.Module:
    aux = cfg.get("aux_channels", {}) or {}
    modalities = {"alphaearth": int(cfg.backbone.get("in_channels", 64))}
    modalities.update({k: int(v) for k, v in aux.items()})
    return MultiModalFusion(
        modalities=modalities,
        hidden=int(cfg.head.get("hidden_dim", 128)),
        num_classes=int(cfg.head.get("out_channels", 1)),
    )
