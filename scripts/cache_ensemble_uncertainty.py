"""P0-A — compute per-chip ensemble uncertainty from the 3-seed LOO multi-seed
checkpoints, and add it as a new feature column to the chip cache.

For each of the 10 Sen1Floods11 events we have three independently-trained
U-Net LOO checkpoints (outputs/leave_one_region_out_multiseed/seed{42,1337,2024}
/test_<region>/checkpoints/*.ckpt). For each chip we forward all three models,
compute per-pixel ensemble std across the three predictions, then take the
chip-level mean of that std — a strong epistemic-uncertainty signal that
chatgpt's review explicitly asked us to add as a baseline beyond single-model
entropy.

Output (in-place update): outputs/layer3_ppo/chip_cache_all10.pkl gains a new
`ensemble_unc` numpy array (shape (n_chips,), one float per chip).

Downstream: scripts/eval_layer3_ppo_significance.py will get a new
`ensemble` baseline that selects the top-budget chips by this uncertainty.
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
sys.path.insert(0, ".")
from geodisaster.data.tile import merge_manifests
from geodisaster.datasets import DisasterPatchDataModule, stats_with_fallbacks
from geodisaster.train import DisasterSegLightningModule
from geodisaster.utils.logging import get_logger, setup_logging

log = get_logger("ensemble-cache")

CACHE = Path("outputs/layer3_ppo/chip_cache_all10.pkl")
SEEDS = [42, 1337, 2024]
PATCH_ROOT = "data/processed/patches"
STATS = "data/processed/norm_stats_sen1floods11.yaml"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _latest_ckpt(d: Path) -> Path:
    cks = sorted(d.glob("*.ckpt"))
    if not cks:
        raise FileNotFoundError(f"no ckpt in {d}")
    return cks[-1]


def _load_module(ckpt: Path):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    hp = state["hyper_parameters"]
    mcfg = OmegaConf.create(hp.get("model_cfg", hp.get("model")))
    tcfg = OmegaConf.create(hp.get("train_cfg", hp.get("train")))
    sources = hp["sources"]
    m = DisasterSegLightningModule(mcfg, tcfg, sources)
    m.load_state_dict(state["state_dict"], strict=True)
    return m.eval().to(DEV), sources


@torch.no_grad()
def _per_chip_probs(module, sources, region: str) -> list[np.ndarray]:
    patches = merge_manifests([Path(PATCH_ROOT) / f"sen1floods11_{region}"])
    norm = stats_with_fallbacks(STATS, sources)
    dm = DisasterPatchDataModule(train_patches=[], val_patches=[], test_patches=patches,
                                 sources=sources, batch_size=1, num_workers=0, normalize=norm)
    out = []
    for b in dm.test_dataloader():
        for k, v in list(b.items()):
            if isinstance(v, torch.Tensor):
                b[k] = v.to(DEV)
        pr = torch.sigmoid(module(b).squeeze(1)).cpu().numpy().reshape(-1).astype(np.float32)
        lb = b["mask"].cpu().numpy().reshape(-1)
        valid = lb != 255
        pr = pr[valid]
        if pr.size == 0:
            continue
        out.append(pr)
    return out


def main():
    setup_logging()
    caches = pickle.loads(CACHE.read_bytes())
    new_ensemble_unc = {}
    for region in caches:
        log.info("processing", region=region)
        seed_probs: list[list[np.ndarray]] = []  # outer = seed, inner = per-chip 1-D probs
        for seed in SEEDS:
            ckdir = Path(f"outputs/leave_one_region_out_multiseed/seed{seed}/test_{region}/checkpoints")
            if not ckdir.exists():
                log.warning("missing_seed", region=region, seed=seed)
                seed_probs.append(None); continue
            ck = _latest_ckpt(ckdir)
            mod, sources = _load_module(ck)
            seed_probs.append(_per_chip_probs(mod, sources, region))
            del mod
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        # Keep only seeds that returned chip-aligned predictions
        seed_probs = [s for s in seed_probs if s is not None]
        if len(seed_probs) < 2:
            log.warning("not_enough_seeds", region=region, n=len(seed_probs))
            new_ensemble_unc[region] = np.zeros(caches[region]["n"], dtype=np.float32)
            continue
        # Validate per-chip count consistency
        n_chip = caches[region]["n"]
        # Some seeds may include 'empty' chips that the cache dropped (Ghana 5 empty).
        # Use the minimum chip count across seeds and trim accordingly.
        min_chip = min(len(s) for s in seed_probs)
        log.info("chip_counts", region=region, cache_n=n_chip,
                 seed_chip_counts=[len(s) for s in seed_probs], using=min_chip)
        ens_unc = np.zeros(min_chip, dtype=np.float32)
        for i in range(min_chip):
            # Stack per-pixel preds from the seeds; need matching pixel counts (same chip).
            pixels = [s[i] for s in seed_probs]
            min_pix = min(p.size for p in pixels)
            stacked = np.stack([p[:min_pix] for p in pixels], axis=0)  # (n_seeds, min_pix)
            std = stacked.std(axis=0)
            ens_unc[i] = float(std.mean())
        # pad/trim to match cache_n if mismatched (defensive)
        if min_chip != n_chip:
            if min_chip > n_chip:
                ens_unc = ens_unc[:n_chip]
            else:
                ens_unc = np.concatenate([ens_unc, np.zeros(n_chip - min_chip, dtype=np.float32)])
        new_ensemble_unc[region] = ens_unc
        log.info("done_region", region=region, mean_unc=float(ens_unc.mean()),
                 std_unc=float(ens_unc.std()))

    # Merge into cache: add `ensemble_unc` key alongside existing `feats_raw`
    for r, v in caches.items():
        v["ensemble_unc"] = new_ensemble_unc.get(r, np.zeros(v["n"], dtype=np.float32))
    CACHE.write_bytes(pickle.dumps(caches))
    log.info("saved", path=str(CACHE), regions=list(caches))


if __name__ == "__main__":
    sys.exit(main() or 0)
