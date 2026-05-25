"""SAM / SAM2 image encoder + light decoder (no prompts).

We treat SAM purely as a frozen image encoder and learn a small per-pixel head.
Prompt-based finetuning is also possible but adds engineering overhead that
isn't on the GeoDisaster-FM critical path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from . import register


class SAMSegmentor(nn.Module):
    def __init__(self, model_name: str, in_channels: int, num_classes: int, frozen: bool = True):
        super().__init__()
        try:
            from transformers import SamModel
        except ImportError as e:
            raise ImportError("Install transformers>=4.30 for SamModel") from e
        self.stem = nn.Conv2d(in_channels, 3, 1) if in_channels != 3 else nn.Identity()
        self.encoder = SamModel.from_pretrained(model_name).vision_encoder
        embed_dim = self.encoder.config.output_channels  # 256 for SAM ViT-B
        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 1), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        H, W = x.shape[-2:]
        # SAM image encoder canonical input is 1024
        x_in = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        feats = self.encoder(x_in).last_hidden_state  # B, 256, 64, 64
        logits = self.decoder(feats)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)


@register("sam_adapter")
def _build(cfg: DictConfig) -> nn.Module:
    bb = cfg.backbone
    return SAMSegmentor(
        model_name=bb.get("type", "facebook/sam-vit-base"),
        in_channels=bb.get("in_channels", 3),
        num_classes=cfg.head.get("out_channels", 1),
        frozen=bb.get("frozen", True),
    )
