"""Phase 3.3 — DOFA-frozen LOEO on HLS Burn-Scars (H1 falsification test #3).

DOFA (Dynamic One-For-All, Zhu-XLab) is a wavelength-conditioned ViT-Base
foundation model pre-trained across five EO modalities (Sentinel-1/2,
Landsat, NAIP, EnMAP). Its dynamic patch embedding accepts arbitrary band
sets via their physical wavelengths — we feed the six HLS S30 bands with
their centre wavelengths. Third backbone in the H1-falsification panel
(after AlphaEarth-on-floods and Prithvi-on-wildfires).

Same two-stage cached pipeline as scripts/train_prithvi_burnscars.py:
  Stage A: frozen DOFA encoder forward, cache patch tokens (196 × 768).
  Stage B: per-LOEO-fold segmentation head on cached features.

Output:
  outputs/leave_one_event_out_dofa_burnscars/test_<YEAR>/checkpoints/best.ckpt
  outputs/leave_one_event_out_dofa_burnscars/summary.json
  outputs/decision/calibration_analysis_dofa_burnscars.json
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
from torchgeo.models import dofa_base_patch16_224

ALL_YEARS = [2018, 2019, 2020, 2021]
RAW = Path("data/raw/hls_burn_scars")
OUT_ROOT = Path("outputs/leave_one_event_out_dofa_burnscars")
CACHE = OUT_ROOT / "feat_cache.npz"
WEIGHTS = Path("data/raw/dofa/DOFA_ViT_base_e100.pth")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YEAR_RE = re.compile(r"\.(\d{4})\d{3}\.v1\.4_merged\.tif$")

# HLS S30 band centre wavelengths in micrometres:
# Blue B02, Green B03, Red B04, NIR-narrow B8A, SWIR1 B11, SWIR2 B12
WAVELENGTHS = [0.490, 0.560, 0.665, 0.865, 1.610, 2.190]


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
        img = src.read().astype(np.float32)              # (6, 512, 512), [0, 1]
    mask_p = merged_p.with_name(merged_p.name.replace("_merged.tif", ".mask.tif"))
    with rasterio.open(mask_p) as src:
        mask = src.read(1).astype(np.int64)
    h, w = img.shape[1:]
    c = 224
    sh, sw = (h - c) // 2, (w - c) // 2
    img = img[:, sh:sh+c, sw:sw+c]
    mask = mask[sh:sh+c, sw:sw+c]
    # DOFA pre-training uses imagery standardised per-chip; we z-score per band.
    img = (img - img.mean(axis=(1, 2), keepdims=True)) / \
          (img.std(axis=(1, 2), keepdims=True) + 1e-6)
    return img.astype(np.float32), mask


def build_dofa():
    model = dofa_base_patch16_224()
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    own = model.state_dict()
    loaded = 0
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v; loaded += 1
    model.load_state_dict(own)
    print(f"  DOFA: loaded {loaded}/{len(state)} pretrained tensors")
    return model.eval()


@torch.no_grad()
def dofa_patch_tokens(model, x: torch.Tensor) -> torch.Tensor:
    """Replicate DOFA.forward_features but return patch tokens (B, 196, 768)."""
    wavelist = torch.tensor(WAVELENGTHS, device=x.device).float()
    model.waves = wavelist
    tok, _ = model.patch_embed(x, model.waves)
    tok = tok + model.pos_embed[:, 1:, :]
    cls_token = model.cls_token + model.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(tok.shape[0], -1, -1)
    tok = torch.cat((cls_tokens, tok), dim=1)
    for block in model.blocks:
        tok = block(tok)
    return tok[:, 1:, :]                                  # strip CLS


@torch.no_grad()
def cache_features():
    if CACHE.exists():
        print(f"[cache] re-using {CACHE}")
        return
    pairs = list_pairs()
    model = build_dofa().to(DEV)
    feats, masks, years = [], [], []
    for i, (p, y) in enumerate(pairs):
        img, mask = _load_chip(p)
        x = torch.from_numpy(img).unsqueeze(0).to(DEV)     # (1, 6, 224, 224)
        tokens = dofa_patch_tokens(model, x)               # (1, 196, 768)
        feats.append(tokens[0].cpu().numpy().astype(np.float16))
        masks.append(mask.astype(np.int8))
        years.append(y)
        if (i + 1) % 100 == 0:
            print(f"  cached {i+1}/{len(pairs)}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, feats=np.stack(feats), masks=np.stack(masks),
                        years=np.asarray(years, dtype=np.int32))
    print(f"[cache] saved {CACHE} ({CACHE.stat().st_size/1e6:.0f} MB)")


class SegHead(nn.Module):
    """Same head as the Prithvi run: 14×14 tokens → 224×224 logits."""

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
        x = self.up(x)
        return self.out(x).squeeze(1)


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


def train_one_fold(held_out, cache, epochs=30, lr=3e-4, batch_size=16):
    train_idx = np.where(cache["years"] != held_out)[0]
    test_idx = np.where(cache["years"] == held_out)[0]
    print(f"[fold={held_out}] train={len(train_idx)} test={len(test_idx)}")
    train_dl = DataLoader(CachedDataset(cache["feats"][train_idx], cache["masks"][train_idx]),
                          batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(CachedDataset(cache["feats"][test_idx], cache["masks"][test_idx]),
                         batch_size=batch_size, shuffle=False, num_workers=0)
    head = SegHead().to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    out_dir = OUT_ROOT / f"test_{held_out}" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    final = {}
    for ep in range(epochs):
        head.train()
        running = 0.0; nb = 0
        for f, y in train_dl:
            f = f.to(DEV); y = y.to(DEV)
            logits = head(f)
            valid = (y != -1).float()
            target = (y == 1).float()
            loss = (bce(logits, target) * valid).sum() / (valid.sum() + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss.item()); nb += 1
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
        bi = int(np.argmax(f1s)); best_t = float(grid[bi]); f1_best = float(f1s[bi])
        print(f"  ep {ep+1:2d}/{epochs}  loss={running/max(nb,1):.4f}  "
              f"F1@0.5={f1_05:.4f}  F1@τ*={f1_best:.4f} (τ*={best_t:.2f})")
        final = {"f1_at_0.5": f1_05, "f1_at_best": f1_best,
                 "best_threshold": best_t, "calib_gain": f1_best - f1_05}
        if f1_best > best_f1:
            best_f1 = f1_best
            torch.save({"epoch": ep, "head": head.state_dict(),
                        "held_out_year": held_out, **final}, out_dir / "best.ckpt")
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
    print("=== Stage A: cache frozen DOFA features ===")
    cache_features()
    print("\n=== Stage B: per-LOEO head training ===")
    cache = np.load(CACHE, allow_pickle=True)
    summary = {}
    for y in args.years:
        summary[str(y)] = train_one_fold(y, cache, args.epochs, args.lr, args.batch_size)
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {OUT_ROOT/'summary.json'}")

    # Also emit the calibration-analysis JSON in the standard format
    calib = {"n_events": len(summary), "events": list(summary.keys()), "per_event": {}}
    gains, ts = [], []
    for y, m in summary.items():
        calib["per_event"][y] = {
            "n_chips": m["n_test_chips"],
            "f1_at_0.5": round(m["f1_at_0.5"], 4),
            "f1_at_best": round(m["f1_at_best"], 4),
            "best_threshold": m["best_threshold"],
            "calib_gain": round(m["calib_gain"], 4),
        }
        gains.append(m["calib_gain"]); ts.append(m["best_threshold"])
    calib["mean_calib_gain"] = round(sum(gains)/len(gains), 4)
    calib["best_threshold_range"] = [min(ts), max(ts)]
    calib["frac_events_optimal_not_0.5"] = round(
        sum(abs(t-0.5) > 0.001 for t in ts)/len(ts), 3)
    out_calib = Path("outputs/decision/calibration_analysis_dofa_burnscars.json")
    out_calib.write_text(json.dumps(calib, indent=2))
    print(f"Saved {out_calib}")


if __name__ == "__main__":
    sys.exit(main() or 0)
