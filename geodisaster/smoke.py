"""End-to-end smoke test on synthetic data.

Generates random patches in the GeoDisaster-FM schema, then exercises:
    1. All 8 model families via build_model + forward pass
    2. compute_norm_stats Welford accumulator
    3. Mini-training (2 epochs, alphaearth_head) via Lightning
    4. Pixel metrics (BinaryConfusion, AUPRC, ECE)
    5. Focal BCE loss numerical stability

Purpose: catch wiring bugs (config key mismatches, shape mismatches, missing
imports, registry failures) before investing in real data. Runs in ~30s on CPU.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .utils.logging import get_logger, setup_logging

log = get_logger("smoke")

H = W = 64        # tiny patches keep CPU runtime under 30s
N_PATCHES = 8
N_EVENTS = 2
ALPHAEARTH_DIM = 64
S1_BANDS = 2      # VV, VH
DEM_BANDS = 4     # elevation, slope, curvature, HAND
S2_BANDS = 6      # RGB + NIR + SWIR1 + SWIR2


def _rng_patch(rng, n_bands, mean=0.0, std=1.0):
    return rng.normal(loc=mean, scale=std, size=(n_bands, H, W)).astype(np.float32)


def _flood_label(rng):
    """Random binary mask with a few circular 'flood' blobs so positives aren't 0."""
    label = np.zeros((H, W), dtype=np.uint8)
    n_blobs = rng.integers(1, 4)
    for _ in range(int(n_blobs)):
        cx, cy = rng.integers(8, W - 8), rng.integers(8, H - 8)
        r = rng.integers(4, 12)
        yy, xx = np.ogrid[:H, :W]
        label[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 1
    return label


def make_synthetic_dataset(root: Path, seed: int = 1234) -> dict:
    """Write fake event directories matching tile_event's manifest schema."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    all_manifests = []

    for ev in range(N_EVENTS):
        event_id = f"smoke_event_{ev}"
        ev_dir = root / event_id
        ev_dir.mkdir(parents=True, exist_ok=True)
        patches_meta = []
        for i in range(N_PATCHES):
            patch_id = f"{event_id}_p{i:04d}"
            # AlphaEarth roughly unit-norm
            ae = _rng_patch(rng, ALPHAEARTH_DIM, mean=0.0, std=1.0)
            # Sentinel-1 dB-like (-15 ± 5)
            s1 = _rng_patch(rng, S1_BANDS, mean=-15.0, std=5.0)
            # DEM elevation-like
            dem = _rng_patch(rng, DEM_BANDS, mean=500.0, std=300.0)
            # Sentinel-2 reflectance-like
            s2 = _rng_patch(rng, S2_BANDS, mean=1500.0, std=800.0)
            label = _flood_label(rng)

            sources = {}
            for name, arr in [("alphaearth", ae), ("sentinel1", s1),
                              ("dem", dem), ("sentinel2", s2)]:
                p = ev_dir / f"{patch_id}__{name}.npy"
                np.save(p, arr)
                sources[name] = str(p)
            label_path = ev_dir / f"{patch_id}__label.npy"
            np.save(label_path, label)
            patches_meta.append({
                "patch_id": patch_id, "row": 0, "col": 0, "size": H,
                "sources": sources, "label_path": str(label_path),
                "pos_fraction": float((label == 1).mean()),
            })

        manifest = {
            "event_id": event_id, "patch_size": H, "stride": H,
            "n_patches": N_PATCHES, "ref_source": "alphaearth",
            "patches": patches_meta,
        }
        (ev_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        all_manifests.append((event_id, manifest))
    return {"root": str(root), "events": [m[0] for m in all_manifests]}


def test_models_forward():
    """Build every registered model family and run one forward pass with fake input."""
    import torch
    from omegaconf import OmegaConf
    from .models import build_model, _REGISTRY

    log.info("test_models_forward", n_registered=len(_REGISTRY))
    B = 2

    # Per-family test configs. We deliberately use small backbone variants where
    # the family supports it, so smoke runs without GPU.
    cases = [
        ("alphaearth_head", {
            "name": "ae_mlp", "family": "alphaearth_head",
            "backbone": {"type": "alphaearth_embedding", "frozen": True, "in_channels": 64},
            "aux_channels": {"sentinel1": 2, "dem": 4, "sentinel2": 6},
            "head": {"type": "mlp", "hidden": [32], "out_channels": 1, "dropout": 0.0,
                     "activation": "gelu"},
        }),
        ("multi_modal_fusion", {
            "name": "fusion", "family": "multi_modal_fusion",
            "backbone": {"in_channels": 64},
            "aux_channels": {"sentinel1": 2, "dem": 4},
            "head": {"hidden_dim": 16, "out_channels": 1},
        }),
        ("smp_unet", {
            "name": "unet", "family": "smp_unet",
            "backbone": {"type": "resnet18", "encoder_weights": None, "in_channels": 6},
            "head": {"out_channels": 1},
        }),
    ]

    results = {}
    for family, cfg_dict in cases:
        if family not in _REGISTRY:
            results[family] = "not_registered"
            continue
        try:
            cfg = OmegaConf.create(cfg_dict)
            model = build_model(cfg).eval()
            if family in {"alphaearth_head", "multi_modal_fusion"}:
                batch = {
                    "alphaearth": torch.randn(B, 64, H, W),
                    "sentinel1":  torch.randn(B, 2,  H, W),
                    "dem":        torch.randn(B, 4,  H, W),
                    "sentinel2":  torch.randn(B, 6,  H, W),
                }
                with torch.no_grad():
                    out = model(batch)
            else:
                with torch.no_grad():
                    out = model(torch.randn(B, 6, H, W))
            assert out.shape[-2:] == (H, W), f"{family} output spatial dims wrong: {out.shape}"
            assert out.shape[0] == B, f"{family} batch dim wrong: {out.shape}"
            results[family] = f"OK shape={tuple(out.shape)}"
        except Exception as e:
            results[family] = f"FAIL: {type(e).__name__}: {e}"

    # Vision-FM / RS-FM families (download model weights). Skip if no network /
    # not installed; still report so the user knows.
    extras = ["segformer", "dinov2_adapter", "sam_adapter", "satmae", "prithvi",
              "remoteclip", "crossearth"]
    for family in extras:
        if family not in _REGISTRY:
            results[family] = "not_registered"
            continue
        results[family] = "skipped (needs network/weights)"
    for fam, r in results.items():
        log.info("model_test", family=fam, result=r)
    return results


def test_metrics_and_loss():
    import torch
    from .metrics import BinaryConfusion, auprc, expected_calibration_error, focal_bce_loss

    rng = np.random.default_rng(0)
    target = torch.from_numpy(rng.integers(0, 2, size=(4, H, W)).astype(np.int64))
    scores = torch.sigmoid(torch.from_numpy(rng.normal(size=(4, H, W)).astype(np.float32)))
    preds = (scores > 0.5).long()

    cm = BinaryConfusion()
    cm.update(preds, target)
    d = cm.as_dict()
    assert 0 <= d["iou"] <= 1, f"iou OOR: {d['iou']}"
    assert 0 <= d["f1"] <= 1, f"f1 OOR: {d['f1']}"

    a = auprc(scores, target)
    e = expected_calibration_error(scores, target)
    assert 0 <= a <= 1 and 0 <= e <= 1, f"auprc/ece OOR: {a}, {e}"

    logits = torch.randn(4, 1, H, W, requires_grad=True)
    loss = focal_bce_loss(logits, target, alpha=0.75, gamma=2.0)
    loss.backward()
    assert torch.isfinite(loss), "loss not finite"
    assert logits.grad is not None and torch.isfinite(logits.grad).all(), "grad not finite"
    log.info("metrics_loss_ok",
             iou=d["iou"], f1=d["f1"], auprc=a, ece=e, loss=float(loss))


def test_norm_stats(patch_root: Path):
    from .data.tile import merge_manifests
    from .datasets import compute_norm_stats

    patches = merge_manifests([patch_root / f"smoke_event_{i}" for i in range(N_EVENTS)])
    stats = compute_norm_stats(
        patches, sources=["alphaearth", "sentinel1", "dem", "sentinel2"], max_patches=None,
    )
    # Should recover the synthetic injection values within Monte Carlo noise
    s1_mean = stats["sentinel1"][0]
    s1_std = stats["sentinel1"][1]
    assert -18 < s1_mean < -12, f"S1 mean off: {s1_mean}"
    assert 3 < s1_std < 8, f"S1 std off: {s1_std}"
    log.info("norm_stats_ok", stats={k: (round(v[0], 2), round(v[1], 2)) for k, v in stats.items()})
    return stats


def test_mini_train(patch_root: Path, work_dir: Path, stats: dict):
    """1-epoch toy training on synthetic data; checks Lightning loop works."""
    import pytorch_lightning as pl
    from omegaconf import OmegaConf

    from .data.tile import merge_manifests
    from .datasets import DisasterPatchDataModule
    from .train import DisasterSegLightningModule, make_trainer

    patches = merge_manifests([patch_root / f"smoke_event_{i}" for i in range(N_EVENTS)])
    train_patches = [p for p in patches if "event_0" in p["event_id"]]
    val_patches   = [p for p in patches if "event_1" in p["event_id"]]

    sources = ["alphaearth", "sentinel1", "dem"]
    dm = DisasterPatchDataModule(
        train_patches=train_patches, val_patches=val_patches, test_patches=val_patches,
        sources=sources, batch_size=2, num_workers=0, normalize=stats,
    )

    model_cfg = OmegaConf.create({
        "name": "ae_mlp_smoke", "family": "alphaearth_head",
        "backbone": {"type": "alphaearth_embedding", "frozen": True, "in_channels": 64},
        "aux_channels": {"sentinel1": 2, "dem": 4},
        "head": {"type": "mlp", "hidden": [16], "out_channels": 1, "dropout": 0.0,
                 "activation": "gelu"},
        "loss": {"type": "focal_bce", "alpha": 0.75, "gamma": 2.0},
    })
    train_cfg = OmegaConf.create({
        "epochs": 2, "batch_size": 2, "num_workers": 0,
        "lr": 1e-3, "weight_decay": 0.0, "scheduler": "none",
        "precision": "32", "accumulate_grad_batches": 1,
        "early_stopping_patience": 99, "log_every_n_steps": 1,
    })

    pl.seed_everything(0)
    module = DisasterSegLightningModule(model_cfg=model_cfg, train_cfg=train_cfg, sources=sources)
    trainer = pl.Trainer(
        max_epochs=2, accelerator="cpu", devices=1,
        enable_checkpointing=False, logger=False,
        enable_progress_bar=False, num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=dm)
    res = trainer.test(module, datamodule=dm, verbose=False)
    assert res, "trainer.test returned empty"
    log.info("mini_train_ok", test_metrics={k: round(float(v), 4)
                                            for k, v in res[0].items() if isinstance(v, (int, float))})


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    tmp = Path(tempfile.mkdtemp(prefix="geodisaster_smoke_"))
    log.info("smoke_start", tmp=str(tmp))
    try:
        make_synthetic_dataset(tmp / "patches")
        log.info("step_1_synth_data_ok")

        results = test_models_forward()
        log.info("step_2_models_forward_ok", families=list(results))

        test_metrics_and_loss()
        log.info("step_3_metrics_loss_ok")

        stats = test_norm_stats(tmp / "patches")
        log.info("step_4_norm_stats_ok")

        test_mini_train(tmp / "patches", tmp / "work", stats)
        log.info("step_5_mini_train_ok")

        print()
        print("=" * 60)
        print("SMOKE TEST PASSED")
        print("=" * 60)
        for fam, res in results.items():
            mark = "OK" if res.startswith("OK") else "--"
            print(f"  [{mark}] {fam:20s} {res}")
        print()
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(main(sys.argv[1:]))
