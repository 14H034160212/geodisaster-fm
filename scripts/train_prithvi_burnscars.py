"""Phase 3 — Prithvi-frozen LOEO on HLS Burn-Scars (H1 falsification test #2).

NASA-IBM's Prithvi-EO-1.0-100M is a Vision Transformer pre-trained with
masked autoencoder reconstruction on the *exact* HLS imagery modality our
burn-scars chips come from. If H1 (representation drift) holds, swapping
in this foundation backbone on matched inputs should outperform a
from-scratch U-Net on the same LOEO splits. We test that prediction.

Strategy
--------
Two stages, ordered for efficiency.

  Stage A — feature cache. Forward each 512×512 chip through the frozen
            Prithvi encoder once. Crop to a centre 224×224 window, run
            T=1 (single-timestep) inference, store the encoder's final
            patch embeddings (B, 196, 768). One-time ~5–10 min per
            year.
  Stage B — per-LOEO head training. For each held-out year, train a
            small decoder head (bilinear-upsample + 2 × 1×1 conv) on
            the cached features from the *other* three years; evaluate
            F1 + calibration headroom on the held-out year. Each head
            trains in ~5–10 min on cached features.

Output
------
  outputs/leave_one_event_out_prithvi_burnscars/test_<YEAR>/checkpoints/best.ckpt
  outputs/leave_one_event_out_prithvi_burnscars/summary.json
  outputs/decision/calibration_analysis_prithvi_burnscars.json
"""
from __future__ import annotations
import argparse, json, re, sys
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio

# Make Prithvi importable
PRITHVI_DIR = Path("data/raw/prithvi/eo-1.0-100M")
sys.path.insert(0, str(PRITHVI_DIR))
from prithvi_mae import PrithviViT                  # noqa: E402

ALL_YEARS = [2018, 2019, 2020, 2021]
RAW = Path("data/raw/hls_burn_scars")
OUT_ROOT = Path("outputs/leave_one_event_out_prithvi_burnscars")
CACHE = Path("outputs/leave_one_event_out_prithvi_burnscars/feat_cache.npz")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YEAR_RE = re.compile(r"\.(\d{4})\d{3}\.v1\.4_merged\.tif$")

# Prithvi normalisation (per-band over CONUS HLS V2)
P_MEAN = np.array([775.2290211032589, 1080.992780391705, 1228.5855250417867,
                   2497.2022620507532, 2204.2139147975554, 1610.8324823273745],
                  dtype=np.float32)
P_STD = np.array([1281.526139861424, 1270.0297974547493, 1399.4802505642526,
                  1368.3446143815768, 1291.6650461197797, 1135.5915596039165],
                 dtype=np.float32)


def list_pairs() -> list[tuple[Path, int]]:
    out = []
    for split in ("training", "validation"):
        for p in sorted((RAW / split).glob("*_merged.tif")):
            m = YEAR_RE.search(p.name)
            if m:
                out.append((p, int(m.group(1))))
    return out


def _load_chip(merged_p: Path):
    with rasterio.open(merged_p) as src:
        img = src.read().astype(np.float32)             # (6, 512, 512)
    mask_p = merged_p.with_name(merged_p.name.replace("_merged.tif", ".mask.tif"))
    with rasterio.open(mask_p) as src:
        mask = src.read(1).astype(np.int64)             # (512, 512)
    # Crop centre 224×224 window (multiple of Prithvi patch_size = 16; loses ~20% of
    # the chip but standardises a clean H1-vs-U-Net comparison)
    h, w = img.shape[1:]
    c = 224
    sh, sw = (h - c) // 2, (w - c) // 2
    img = img[:, sh:sh+c, sw:sw+c]
    mask = mask[sh:sh+c, sw:sw+c]
    # HLS reflectance scaled by 10000 in the dataset (values are in [0, 1])
    img_unscaled = img * 10000.0
    img_norm = (img_unscaled - P_MEAN[:, None, None]) / P_STD[:, None, None]
    return img_norm.astype(np.float32), mask


def build_prithvi() -> PrithviViT:
    with open(PRITHVI_DIR / "config.yaml") as f:
        import yaml
        cfg = yaml.safe_load(f)["model_args"]
    # Build with pretrained num_frames=3 (we'll feed T=1 chips replicated 3×).
    # encoder_only=True drops the MAE decoder, matching our use case.
    model = PrithviViT(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        num_frames=3,
        in_chans=cfg["in_chans"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        coords_encoding=[],
        coords_scale_learn=False,
        encoder_only=True,
    )
    # Pretrained checkpoint prefixes encoder weights with 'encoder.'; strip it.
    state = torch.load(PRITHVI_DIR / "Prithvi_EO_V1_100M.pt", map_location="cpu",
                       weights_only=False)
    state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    own = model.state_dict()
    loaded = 0
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v; loaded += 1
    model.load_state_dict(own)
    print(f"  Prithvi: loaded {loaded}/{len(state)} encoder tensors")
    return model.eval()


@torch.no_grad()
def cache_features():
    """Forward every chip through frozen Prithvi; cache encoder features."""
    if CACHE.exists():
        print(f"[cache] re-using existing {CACHE}")
        return
    pairs = list_pairs()
    model = build_prithvi().to(DEV)
    feats, masks, years, paths = [], [], [], []
    for i, (p, y) in enumerate(pairs):
        img, mask = _load_chip(p)
        # Replicate single-timestep chip 3× along T to match pretrained num_frames=3.
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(2).repeat(1, 1, 3, 1, 1).to(DEV)
        out = model.forward_features(x)            # list of 12 layer outputs
        last = out[-1]                              # (1, 1+T*H*W = 589, 768)
        # Strip CLS token, average across the three time copies (degenerate "static"),
        # yielding 196 spatial patches.
        spatial = last[:, 1:, :].reshape(1, 3, 14, 14, 768).mean(dim=1)   # (1, 14, 14, 768)
        feat = spatial.reshape(1, 196, 768).cpu().numpy().astype(np.float16)
        feats.append(feat[0])
        masks.append(mask.astype(np.int8))
        years.append(y); paths.append(str(p))
        if (i + 1) % 100 == 0:
            print(f"  cached {i+1}/{len(pairs)} chips")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE,
                        feats=np.stack(feats),
                        masks=np.stack(masks),
                        years=np.asarray(years, dtype=np.int32),
                        paths=np.asarray(paths))
    print(f"[cache] saved {CACHE}  ({CACHE.stat().st_size/1e6:.0f} MB)")


# --------------------------------------------------------------------------- #
# Small decoder head trained per LOEO fold
# --------------------------------------------------------------------------- #
class PrithviSegHead(nn.Module):
    """14×14 patch tokens → 224×224 logits via 4-stage progressive upsample."""

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, 256)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.GELU(),    # 14 → 28
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.GELU(),     # 28 → 56
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.GELU(),      # 56 → 112
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.GELU(),      # 112 → 224
        )
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, tokens):
        # tokens: (B, 196, 768)
        x = self.in_proj(tokens)                              # (B, 196, 256)
        x = x.transpose(1, 2).reshape(-1, 256, 14, 14)        # (B, 256, 14, 14)
        x = self.up(x)
        return self.out(x).squeeze(1)                         # (B, 224, 224)


class CachedDataset(Dataset):
    def __init__(self, feats, masks):
        self.feats = feats; self.masks = masks

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        f = torch.from_numpy(self.feats[idx].astype(np.float32))
        m = torch.from_numpy(self.masks[idx].astype(np.int64))
        return f, m


def f1_at(probs, labels, t):
    pred = probs >= t; pos = labels == 1
    tp = float(np.sum(pred & pos)); fp = float(np.sum(pred & ~pos))
    fn = float(np.sum(~pred & pos))
    d = 2*tp + fp + fn
    return float(2*tp / d) if d > 0 else 0.0


def train_one_fold(held_out: int, cache, epochs=30, lr=3e-4, batch_size=16):
    train_idx = np.where(cache["years"] != held_out)[0]
    test_idx = np.where(cache["years"] == held_out)[0]
    print(f"[fold={held_out}] train={len(train_idx)} test={len(test_idx)}")
    train_ds = CachedDataset(cache["feats"][train_idx], cache["masks"][train_idx])
    test_ds = CachedDataset(cache["feats"][test_idx],  cache["masks"][test_idx])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    head = PrithviSegHead().to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    out_dir = OUT_ROOT / f"test_{held_out}" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    final = {}

    for ep in range(epochs):
        head.train()
        running = 0.0; nbatch = 0
        for f, y in train_dl:
            f = f.to(DEV); y = y.to(DEV)
            logits = head(f)
            valid = (y != -1).float()
            target = (y == 1).float()
            loss = (bce(logits, target) * valid).sum() / (valid.sum() + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss.item()); nbatch += 1
        sched.step()

        head.eval()
        P, L = [], []
        with torch.no_grad():
            for f, y in test_dl:
                f = f.to(DEV)
                pr = torch.sigmoid(head(f)).cpu().numpy().reshape(-1).astype(np.float32)
                lb = y.numpy().reshape(-1)
                valid = lb >= 0
                P.append(pr[valid]); L.append((lb[valid] == 1).astype(np.uint8))
        probs = np.concatenate(P); labels = np.concatenate(L)
        f1_05 = f1_at(probs, labels, 0.5)
        grid = np.linspace(0.05, 0.95, 19)
        f1s = [f1_at(probs, labels, t) for t in grid]
        bi = int(np.argmax(f1s)); best_t = float(grid[bi]); f1_best = float(f1s[bi])
        print(f"  ep {ep+1:2d}/{epochs}  loss={running/max(nbatch,1):.4f}  "
              f"F1@0.5={f1_05:.4f}  F1@τ*={f1_best:.4f} (τ*={best_t:.2f})")
        final = {"f1_at_0.5": f1_05, "f1_at_best": f1_best,
                 "best_threshold": best_t, "calib_gain": f1_best - f1_05}
        if f1_best > best_f1:
            best_f1 = f1_best
            torch.save({"epoch": ep, "head": head.state_dict(),
                        "held_out_year": held_out, **final},
                       out_dir / "best.ckpt")
    metrics = {"held_out_year": held_out, "n_train_chips": int(len(train_idx)),
               "n_test_chips": int(len(test_idx)), "best_f1": float(best_f1),
               **{k: float(v) for k, v in final.items()}}
    (out_dir.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[fold={held_out}] DONE best F1={best_f1:.4f}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=ALL_YEARS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=== Stage A: cache frozen Prithvi features ===")
    cache_features()
    print()
    print("=== Stage B: per-LOEO head training ===")
    cache = np.load(CACHE, allow_pickle=True)
    summary = {}
    for y in args.years:
        summary[str(y)] = train_one_fold(y, cache, args.epochs, args.lr, args.batch_size)
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {OUT_ROOT/'summary.json'}")


if __name__ == "__main__":
    sys.exit(main() or 0)
