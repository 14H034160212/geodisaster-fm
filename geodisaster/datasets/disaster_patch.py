"""Disaster patch dataset + Lightning DataModule.

Consumes the patch manifests written by ``geodisaster.data.tile.tile_event``.
Each item is a dict of stacked tensors per source plus a label mask:
    {
        "alphaearth": Tensor[C_ae, H, W],
        "sentinel1":  Tensor[2, H, W],
        "sentinel2":  Tensor[C_s2, H, W],
        "dem":        Tensor[C_dem, H, W],
        "mask":       Tensor[H, W] (long),
        "event_id":   str,
        "patch_id":   str,
    }
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore


def _load_npy(path: str) -> np.ndarray:
    return np.load(path)


class DisasterPatchDataset(Dataset):
    def __init__(
        self,
        patches: list[dict[str, Any]],
        sources: Sequence[str] = ("alphaearth", "sentinel1", "dem"),
        augment: bool = False,
        normalize: dict[str, tuple[float, float]] | None = None,
    ):
        self.patches = list(patches)
        self.sources = tuple(sources)
        self.augment = augment
        # per-source (mean, std) for input z-scoring. Compute with
        # ``geodisaster compute-stats`` for your training subset; otherwise
        # we fall back to EMPIRICAL_FALLBACKS in geodisaster.datasets.stats.
        from .stats import stats_with_fallbacks
        if normalize is None:
            self.normalize = stats_with_fallbacks(None, sources)
        else:
            self.normalize = dict(normalize)

    def __len__(self) -> int:
        return len(self.patches)

    def _norm(self, src: str, arr: np.ndarray) -> np.ndarray:
        m, s = self.normalize.get(src, (0.0, 1.0))
        # m/s may be scalars (broadcast across all bands) or per-band arrays
        m = np.asarray(m, dtype=np.float32)
        s = np.asarray(s, dtype=np.float32)
        if m.ndim == 1 and arr.ndim >= 3 and m.size == arr.shape[0]:
            m = m.reshape(-1, *([1] * (arr.ndim - 1)))
            s = s.reshape(-1, *([1] * (arr.ndim - 1)))
        s = np.where(s > 1e-6, s, np.float32(1.0))
        return (arr - m) / s

    def _augment(self, tensors: dict[str, torch.Tensor], mask: torch.Tensor):
        # Horizontal flip
        if torch.rand(1).item() < 0.5:
            for k in tensors:
                tensors[k] = torch.flip(tensors[k], dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        # Vertical flip
        if torch.rand(1).item() < 0.5:
            for k in tensors:
                tensors[k] = torch.flip(tensors[k], dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        # 90-deg rotation
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            for kk in tensors:
                tensors[kk] = torch.rot90(tensors[kk], k=k, dims=(-2, -1))
            mask = torch.rot90(mask, k=k, dims=(-2, -1))
        return tensors, mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.patches[idx]
        out: dict[str, Any] = {}
        for src in self.sources:
            path = rec["sources"].get(src)
            if path is None:
                continue
            arr = _load_npy(path)
            arr = self._norm(src, arr)
            out[src] = torch.from_numpy(np.nan_to_num(arr).astype(np.float32))
        mask = torch.from_numpy(_load_npy(rec["label_path"]).astype(np.int64))
        if self.augment:
            out, mask = self._augment(out, mask)
        out["mask"] = mask
        out["event_id"] = rec["event_id"]
        out["patch_id"] = rec["patch_id"]
        return out


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


class DisasterPatchDataModule(pl.LightningDataModule if pl else object):
    def __init__(
        self,
        train_patches: list[dict],
        val_patches: list[dict],
        test_patches: list[dict],
        sources: Sequence[str] = ("alphaearth", "sentinel1", "dem"),
        batch_size: int = 16,
        num_workers: int = 8,
        normalize: dict[str, tuple[float, float]] | None = None,
        augment_train: bool = True,
    ):
        super().__init__()
        self.train_patches = train_patches
        self.val_patches = val_patches
        self.test_patches = test_patches
        self.sources = tuple(sources)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.augment_train = augment_train

    def _make(self, patches, augment):
        return DisasterPatchDataset(
            patches, sources=self.sources, augment=augment, normalize=self.normalize,
        )

    def train_dataloader(self):
        return DataLoader(
            self._make(self.train_patches, augment=self.augment_train),
            batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=True, pin_memory=True, collate_fn=collate, drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._make(self.val_patches, augment=False),
            batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, pin_memory=True, collate_fn=collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self._make(self.test_patches, augment=False),
            batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, pin_memory=True, collate_fn=collate,
        )
