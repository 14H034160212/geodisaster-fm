"""SegFormer baseline via HuggingFace transformers."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from . import register


class SegFormerWrapper(nn.Module):
    def __init__(self, model_name: str, in_channels: int, num_classes: int):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        cfg = SegformerConfig.from_pretrained(model_name)
        cfg.num_labels = num_classes
        # MiT backbones expect 3 channels; adapt with a 1x1 stem if needed
        self.in_channels = in_channels
        self.stem = (
            nn.Conv2d(in_channels, 3, kernel_size=1) if in_channels != 3 else nn.Identity()
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, config=cfg, ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        out = self.model(pixel_values=x).logits
        # SegFormer outputs at H/4 — upsample to input size
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


@register("segformer")
def _build_segformer(cfg: DictConfig) -> nn.Module:
    bb = cfg.backbone
    return SegFormerWrapper(
        model_name=bb.get("type", "nvidia/segformer-b0-finetuned-ade-512-512"),
        in_channels=bb.get("in_channels", 3),
        num_classes=cfg.head.get("out_channels", 1),
    )
