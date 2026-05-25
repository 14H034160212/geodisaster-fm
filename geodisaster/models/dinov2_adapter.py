"""DINOv2 backbone + lightweight segmentation decoder.

Frozen DINOv2 ViT-B/14 is widely used as a vision-FM baseline; we add a small
linear+upsample decoder so it produces dense per-pixel outputs at the patch
input resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from . import register


class DINOv2Segmentor(nn.Module):
    def __init__(self, model_name: str, in_channels: int, num_classes: int,
                 patch_size: int = 14, frozen: bool = True):
        super().__init__()
        from transformers import AutoModel
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.stem = nn.Conv2d(in_channels, 3, 1) if in_channels != 3 else nn.Identity()
        self.backbone = AutoModel.from_pretrained(model_name)
        embed_dim = self.backbone.config.hidden_size
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 1), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        B, _, H, W = x.shape
        # DINOv2 wants H/W divisible by patch_size
        ph = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size
        pw = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size
        if (ph, pw) != (H, W):
            x = F.interpolate(x, size=(ph, pw), mode="bilinear", align_corners=False)
        out = self.backbone(pixel_values=x)
        feats = out.last_hidden_state[:, 1:, :]  # drop CLS
        h_p, w_p = ph // self.patch_size, pw // self.patch_size
        feats = feats.transpose(1, 2).reshape(B, -1, h_p, w_p)
        logits = self.decoder(feats)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)


@register("dinov2_adapter")
def _build(cfg: DictConfig) -> nn.Module:
    bb = cfg.backbone
    return DINOv2Segmentor(
        model_name=bb.get("type", "facebook/dinov2-base"),
        in_channels=bb.get("in_channels", 3),
        num_classes=cfg.head.get("out_channels", 1),
        patch_size=bb.get("patch_size", 14),
        frozen=bb.get("frozen", True),
    )
