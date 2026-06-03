"""Lightning training module + training entrypoint.

Used by ``geodisaster train`` and by the experiment drivers in P4. Models
that take a dict input (alphaearth_head, multi_modal_fusion) get the whole
batch; image-only models get the channel-concat of selected sources.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

try:
    import pytorch_lightning as pl
except ImportError:
    pl = None  # type: ignore

from .metrics import BinaryConfusion, auprc, expected_calibration_error, focal_bce_loss
from .models import build_model
from .utils.logging import get_logger

log = get_logger("train")


DICT_INPUT_FAMILIES = {"alphaearth_head", "multi_modal_fusion"}


def _model_takes_dict(cfg: DictConfig) -> bool:
    return cfg.get("family") in DICT_INPUT_FAMILIES


def _pack_image(batch: dict[str, torch.Tensor], sources: list[str]) -> torch.Tensor:
    ref_shape = batch[sources[0]].shape[-2:]
    chans = []
    for s in sources:
        t = batch[s]
        if t.shape[-2:] != ref_shape:
            t = F.interpolate(t, size=ref_shape, mode="bilinear", align_corners=False)
        chans.append(t)
    return torch.cat(chans, dim=1)


class DisasterSegLightningModule(pl.LightningModule if pl else object):
    def __init__(self, model_cfg, train_cfg, sources: list[str]):
        super().__init__()
        # Keys match __init__ arg names so load_from_checkpoint can re-instantiate.
        self.save_hyperparameters({
            "model_cfg": OmegaConf.to_container(model_cfg, resolve=True)
                         if isinstance(model_cfg, DictConfig) else model_cfg,
            "train_cfg": OmegaConf.to_container(train_cfg, resolve=True)
                         if isinstance(train_cfg, DictConfig) else train_cfg,
            "sources":   list(sources),
        })
        # When loading from checkpoint we get plain dicts back; re-wrap them.
        if isinstance(model_cfg, dict):
            model_cfg = OmegaConf.create(model_cfg)
        if isinstance(train_cfg, dict):
            train_cfg = OmegaConf.create(train_cfg)
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.sources = list(sources)
        self.model: nn.Module = build_model(model_cfg)
        self.takes_dict = _model_takes_dict(model_cfg)
        loss_cfg = model_cfg.get("loss", {})
        self.loss_alpha = float(loss_cfg.get("alpha", 0.75))
        self.loss_gamma = float(loss_cfg.get("gamma", 2.0))
        self._val_cm: BinaryConfusion = BinaryConfusion()
        self._test_cm: BinaryConfusion = BinaryConfusion()
        self._val_scores: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        if self.takes_dict:
            return self.model(batch)
        x = _pack_image(batch, self.sources)
        return self.model(x)

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        logits = self(batch)
        target = batch["mask"]
        loss = focal_bce_loss(logits, target, alpha=self.loss_alpha, gamma=self.loss_gamma)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=stage == "train", on_epoch=True)
        if stage == "val":
            scores = torch.sigmoid(logits.squeeze(1))
            preds = (scores > 0.5).long()
            self._val_cm.update(preds, target)
            # keep on CPU: concatenating a large val set on-GPU for AUPRC/ECE
            # can OOM (e.g. xBD with >1000 val patches on a shared GPU).
            self._val_scores.append(scores.detach().cpu())
            self._val_targets.append(target.detach().cpu())
        if stage == "test":
            preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).long()
            self._test_cm.update(preds, target)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def on_validation_epoch_end(self):
        m = self._val_cm.as_dict()
        scores = torch.cat(self._val_scores) if self._val_scores else torch.zeros(0)
        targets = torch.cat(self._val_targets) if self._val_targets else torch.zeros(0)
        m["auprc"] = auprc(scores, targets) if scores.numel() else 0.0
        m["ece"] = expected_calibration_error(scores, targets) if scores.numel() else 0.0
        for k, v in m.items():
            self.log(f"val/{k}", v, prog_bar=k in {"f1", "iou"})
        self._val_cm = BinaryConfusion()
        self._val_scores = []
        self._val_targets = []

    def on_test_epoch_end(self):
        for k, v in self._test_cm.as_dict().items():
            self.log(f"test/{k}", v)
        self._test_cm = BinaryConfusion()

    def configure_optimizers(self):
        lr = float(self.train_cfg.get("lr", 1e-4))
        wd = float(self.train_cfg.get("weight_decay", 1e-2))
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        epochs = int(self.train_cfg.get("epochs", 50))
        sched_kind = self.train_cfg.get("scheduler", "cosine")
        if sched_kind == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        elif sched_kind == "step":
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.5)
        else:
            return opt
        return {"optimizer": opt, "lr_scheduler": sched}


def make_trainer(train_cfg: DictConfig, workdir: str | Path) -> "pl.Trainer":
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    workdir = Path(workdir)
    # Note: ``val/f1`` in filenames would create per-epoch subdirs because of the
    # forward slash. Use a slash-free format key (``val_f1``) and Lightning's
    # ``auto_insert_metric_name=False`` to avoid that.
    callbacks = [
        ModelCheckpoint(
            dirpath=workdir / "checkpoints", monitor="val/f1", mode="max",
            filename="best-epoch{epoch:03d}", save_top_k=3,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(monitor="val/f1", mode="max",
                      patience=int(train_cfg.get("early_stopping_patience", 10))),
    ]
    return pl.Trainer(
        max_epochs=int(train_cfg.get("epochs", 50)),
        accumulate_grad_batches=int(train_cfg.get("accumulate_grad_batches", 1)),
        precision=train_cfg.get("precision", "bf16-mixed"),
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 25)),
        callbacks=callbacks,
        default_root_dir=str(workdir),
    )
