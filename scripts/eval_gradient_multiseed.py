"""Multi-seed error bars for the three-backbone calibration-drift gradient.

The R5 'three-backbone gradient' claim (task-match weakens => calibration
drift grows: U-Net +0.001 -> Prithvi +0.004 -> DOFA +0.013) was computed
from single-seed head training. A referee will (rightly) ask for error bars.
This script retrains the segmentation heads of the two cached-feature
backbones (Prithvi, DOFA) with N seeds per LOEO fold and reports
mean +/- std for F1@0.5, F1@tau*, and the calibration gain.

The U-Net entry cannot be re-seeded this cheaply (full training per fold);
its single-seed numbers stay, flagged as such in the manuscript.

Output: outputs/decision/gradient_multiseed.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALL_YEARS = [2018, 2019, 2020, 2021]

CACHES = {
    "prithvi": Path("outputs/leave_one_event_out_prithvi_burnscars/feat_cache.npz"),
    "dofa":    Path("outputs/leave_one_event_out_dofa_burnscars/feat_cache.npz"),
}
OUT = Path("outputs/decision/gradient_multiseed.json")


class SegHead(nn.Module):
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, 256)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.GELU(),
        )
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, tokens):
        x = self.in_proj(tokens)
        x = x.transpose(1, 2).reshape(-1, 256, 14, 14)
        return self.out(self.up(x)).squeeze(1)


class CachedDataset(Dataset):
    def __init__(self, feats, masks):
        self.feats = feats; self.masks = masks

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.feats[idx].astype(np.float32)),
                torch.from_numpy(self.masks[idx].astype(np.int64)))


def f1_at(probs, labels, t):
    pred = probs >= t; pos = labels == 1
    tp = float(np.sum(pred & pos)); fp = float(np.sum(pred & ~pos))
    fn = float(np.sum(~pred & pos))
    d = 2*tp + fp + fn
    return float(2*tp / d) if d > 0 else 0.0


def train_head(cache, held_out: int, seed: int, epochs=30, lr=3e-4, bs=16):
    torch.manual_seed(seed); np.random.seed(seed)
    train_idx = np.where(cache["years"] != held_out)[0]
    test_idx = np.where(cache["years"] == held_out)[0]
    train_dl = DataLoader(CachedDataset(cache["feats"][train_idx], cache["masks"][train_idx]),
                          batch_size=bs, shuffle=True, num_workers=0,
                          generator=torch.Generator().manual_seed(seed))
    test_dl = DataLoader(CachedDataset(cache["feats"][test_idx], cache["masks"][test_idx]),
                         batch_size=bs, shuffle=False, num_workers=0)
    head = SegHead().to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(epochs):
        head.train()
        for f, y in train_dl:
            f = f.to(DEV); y = y.to(DEV)
            logits = head(f)
            valid = (y != -1).float()
            target = (y == 1).float()
            loss = (bce(logits, target) * valid).sum() / (valid.sum() + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    head.eval()
    P, L = [], []
    with torch.no_grad():
        for f, y in test_dl:
            f = f.to(DEV)
            pr = torch.sigmoid(head(f)).cpu().numpy().reshape(-1)
            lb = y.numpy().reshape(-1)
            valid = lb >= 0
            P.append(pr[valid]); L.append((lb[valid] == 1).astype(np.uint8))
    probs = np.concatenate(P); labels = np.concatenate(L)
    f1_05 = f1_at(probs, labels, 0.5)
    grid = np.linspace(0.05, 0.95, 19)
    f1s = [f1_at(probs, labels, t) for t in grid]
    bi = int(np.argmax(f1s))
    return {"f1_at_0.5": f1_05, "f1_at_best": float(f1s[bi]),
            "best_threshold": float(grid[bi]),
            "calib_gain": float(f1s[bi]) - f1_05}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    out = {}
    for backbone, cache_p in CACHES.items():
        cache = np.load(cache_p, allow_pickle=True)
        per_fold = {}
        for year in ALL_YEARS:
            runs = []
            for s in range(args.seeds):
                r = train_head(cache, year, seed=s, epochs=args.epochs)
                runs.append(r)
                print(f"[{backbone} fold={year} seed={s}] "
                      f"F1@0.5={r['f1_at_0.5']:.4f} F1@τ*={r['f1_at_best']:.4f} "
                      f"τ*={r['best_threshold']:.2f} gain={r['calib_gain']:+.4f}")
            per_fold[str(year)] = {
                "runs": runs,
                "f1_best_mean": round(float(np.mean([r["f1_at_best"] for r in runs])), 4),
                "f1_best_std": round(float(np.std([r["f1_at_best"] for r in runs])), 4),
                "gain_mean": round(float(np.mean([r["calib_gain"] for r in runs])), 4),
                "gain_std": round(float(np.std([r["calib_gain"] for r in runs])), 4),
            }
        all_gains = [r["calib_gain"] for y in per_fold.values() for r in y["runs"]]
        all_f1 = [r["f1_at_best"] for y in per_fold.values() for r in y["runs"]]
        out[backbone] = {
            "per_fold": per_fold,
            "overall_f1_best_mean": round(float(np.mean(all_f1)), 4),
            "overall_f1_best_std": round(float(np.std(all_f1)), 4),
            "overall_gain_mean": round(float(np.mean(all_gains)), 4),
            "overall_gain_std": round(float(np.std(all_gains)), 4),
            "n_seeds": args.seeds,
        }
        print(f"\n[{backbone}] overall F1@τ* = {out[backbone]['overall_f1_best_mean']:.4f} "
              f"± {out[backbone]['overall_f1_best_std']:.4f}   "
              f"gain = {out[backbone]['overall_gain_mean']:+.4f} "
              f"± {out[backbone]['overall_gain_std']:.4f}\n")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    sys.exit(main() or 0)
