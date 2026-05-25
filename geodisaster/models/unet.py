"""U-Net and DeepLabV3+ baselines via segmentation_models_pytorch."""
from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig

from . import register


@register("smp_unet")
def _build_unet(cfg: DictConfig) -> nn.Module:
    import segmentation_models_pytorch as smp
    bb = cfg.backbone
    return smp.Unet(
        encoder_name=bb.get("type", "resnet34"),
        encoder_weights=bb.get("encoder_weights", "imagenet"),
        in_channels=bb.get("in_channels", 3),
        classes=cfg.head.get("out_channels", 1),
    )


@register("smp_deeplabv3plus")
def _build_deeplab(cfg: DictConfig) -> nn.Module:
    import segmentation_models_pytorch as smp
    bb = cfg.backbone
    return smp.DeepLabV3Plus(
        encoder_name=bb.get("type", "resnet50"),
        encoder_weights=bb.get("encoder_weights", "imagenet"),
        in_channels=bb.get("in_channels", 3),
        classes=cfg.head.get("out_channels", 1),
    )
