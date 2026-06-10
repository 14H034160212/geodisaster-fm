"""Phase 2 — Train U-Net leave-one-event-out on HLS-Burn-Scars wildfires.

Events = fire seasons (year): 2018 / 2019 / 2020 / 2021. For each year,
train on the other three years, hold the focal year out for evaluation.

The HLS imagery has 6 bands (B, G, R, NIR, SW1, SW2 — HLS S30) at 30 m;
labels are binary 0/1 with -1 for missing. Different channel count from
Sen1Floods11, so we use a fresh U-Net (random init, in_channels=6).

This is a one-off script (the existing patch pipeline is bound to
Sen1Floods11 chip-array format); it reads TIFFs directly and trains
U-Net via segmentation-models-pytorch (already used by the main paper).

Output per fold:
  outputs/leave_one_event_out_burnscars/test_<YEAR>/checkpoints/best.ckpt
  outputs/leave_one_event_out_burnscars/test_<YEAR>/metrics.json
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import rasterio

ALL_YEARS = [2018, 2019, 2020, 2021]
RAW = Path("data/raw/hls_burn_scars")
OUT_ROOT = Path("outputs/leave_one_event_out_burnscars")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_pairs() -> list[tuple[Path, int]]:
    """Return list of (merged_tif_path, year) across train+val dirs."""
    pat = re.compile(r"\.(\d{4})\d{3}\.v1\.4_merged\.tif$")
    out = []
    for split in ("training", "validation"):
        for p in sorted((RAW / split).glob("*_merged.tif")):
            m = pat.search(p.name)
            if not m:
                continue
            out.append((p, int(m.group(1))))
    return out


def _mask_path(merged: Path) -> Path:
    return merged.with_name(merged.name.replace("_merged.tif", ".mask.tif"))


class HLSDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, int]], augment: bool = False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, _year = self.pairs[idx]
        with rasterio.open(img_p) as src:
            img = src.read().astype(np.float32)            # (6, 512, 512)
        with rasterio.open(_mask_path(img_p)) as src:
            mask = src.read(1)                              # (512, 512), 0/1/-1
        # Robust per-channel z-score (HLS reflectance is roughly 0–10000 but
        # has outliers; we standardise per chip).
        img = (img - img.mean(axis=(1, 2), keepdims=True)) / \
              (img.std(axis=(1, 2), keepdims=True) + 1e-3)
        mask = mask.astype(np.int64)
        mask[mask < 0] = 255                                # missing → ignore
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, :, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
                mask = np.ascontiguousarray(mask[::-1, :])
        return torch.from_numpy(img), torch.from_numpy(mask)


def f1_at(probs: np.ndarray, labels: np.ndarray, t: float) -> float:
    pred = probs >= t
    pos = labels == 1
    tp = float(np.sum(pred & pos))
    fp = float(np.sum(pred & ~pos))
    fn = float(np.sum(~pred & pos))
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def train_one_fold(held_out_year: int, *, epochs: int, batch_size: int, lr: float):
    pairs = list_pairs()
    train_pairs = [p for p in pairs if p[1] != held_out_year]
    test_pairs  = [p for p in pairs if p[1] == held_out_year]
    print(f"[fold={held_out_year}] train={len(train_pairs)} test={len(test_pairs)}")

    train_ds = HLSDataset(train_pairs, augment=True)
    test_ds  = HLSDataset(test_pairs, augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=6, classes=1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    out_dir = OUT_ROOT / f"test_{held_out_year}" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for ep in range(epochs):
        model.train()
        running = 0.0; nbatch = 0
        for x, y in train_dl:
            x = x.to(DEV, non_blocking=True); y = y.to(DEV, non_blocking=True)
            logits = model(x).squeeze(1)
            mask = (y != 255).float()
            target = (y == 1).float()
            loss = (bce(logits, target) * mask).sum() / (mask.sum() + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss.item()); nbatch += 1
        sched.step()

        # eval on held-out
        model.eval()
        P, L = [], []
        with torch.no_grad():
            for x, y in test_dl:
                x = x.to(DEV); pr = torch.sigmoid(model(x).squeeze(1)).cpu().numpy()
                lb = y.numpy()
                valid = lb != 255
                P.append(pr[valid]); L.append((lb[valid] == 1).astype(np.uint8))
        probs = np.concatenate(P); labels = np.concatenate(L)
        f1_05 = f1_at(probs, labels, 0.5)
        # best threshold sweep
        grid = np.linspace(0.05, 0.95, 19)
        f1s = [f1_at(probs, labels, t) for t in grid]
        bi = int(np.argmax(f1s)); best_t = float(grid[bi]); f1_best = float(f1s[bi])
        print(f"  ep {ep+1:2d}/{epochs}  train loss={running/max(nbatch,1):.4f}  "
              f"test F1@0.5={f1_05:.4f}  F1@τ*={f1_best:.4f} (τ*={best_t:.2f})")
        if f1_best > best_f1:
            best_f1 = f1_best
            torch.save({"epoch": ep, "model": model.state_dict(),
                        "f1_at_0.5": f1_05, "f1_at_best": f1_best,
                        "best_threshold": best_t,
                        "held_out_year": held_out_year,
                        "in_channels": 6, "arch": "smp_unet_resnet34"},
                       out_dir / "best.ckpt")

    metrics = {"held_out_year": held_out_year,
               "n_train_chips": len(train_pairs), "n_test_chips": len(test_pairs),
               "best_f1": best_f1, "final_f1_at_0.5": f1_05,
               "final_f1_at_best": f1_best, "final_best_threshold": best_t}
    (out_dir.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[fold={held_out_year}] DONE best F1={best_f1:.4f}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=ALL_YEARS,
                    help="which fire-season years to hold out (default all 4)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for y in args.years:
        summary[str(y)] = train_one_fold(y, epochs=args.epochs,
                                         batch_size=args.batch_size, lr=args.lr)
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {OUT_ROOT/'summary.json'}")


if __name__ == "__main__":
    sys.exit(main() or 0)
