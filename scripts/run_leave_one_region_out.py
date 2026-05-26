"""Leave-one-region-out cross-domain matrix for Sen1Floods11.

For each of 10 regions, train on 8 of the other 9 and validate on the 9th, then
test on the held-out region. Produces a 10x1 result table that, combined with
in-domain numbers, becomes CrossEarth-style Fig 4 heatmap data.

Why this experiment matters:
  - Replaces our single (train=8, val=Spain, test=USA) split with full coverage.
  - Tests "does Spain just happen to be hard? Is USA just hard?" → no, every
    region tests against generalization.
  - Direct analog to CrossEarth's 28 cross-domain settings table.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ALL_REGIONS = [
    "Ghana", "India", "Mekong", "Nigeria", "Pakistan",
    "Paraguay", "Somalia", "Spain", "Sri-Lanka", "USA",
]


def evaluate_checkpoint(ckpt_path: Path, test_region: str, stats_path: str) -> dict:
    """Reload best checkpoint and evaluate on test region."""
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, ".")
    from geodisaster.data.tile import merge_manifests
    from geodisaster.datasets import DisasterPatchDataModule, stats_with_fallbacks
    from geodisaster.train import DisasterSegLightningModule
    from geodisaster.metrics import BinaryConfusion, auprc, expected_calibration_error

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = state["hyper_parameters"]
    model_cfg = OmegaConf.create(hp.get("model_cfg", hp.get("model")))
    train_cfg = OmegaConf.create(hp.get("train_cfg", hp.get("train")))
    sources = hp["sources"]
    module = DisasterSegLightningModule(model_cfg, train_cfg, sources)
    module.load_state_dict(state["state_dict"], strict=True)
    module.eval().cuda()

    test = merge_manifests([Path("data/processed/patches") / f"sen1floods11_{test_region}"])
    norm = stats_with_fallbacks(stats_path, sources)
    dm = DisasterPatchDataModule(
        train_patches=[], val_patches=[], test_patches=test,
        sources=sources, batch_size=16, num_workers=4, normalize=norm,
    )
    cm = BinaryConfusion()
    all_scores, all_targets = [], []
    with torch.no_grad():
        for batch in dm.test_dataloader():
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            logits = module(batch)
            scores = torch.sigmoid(logits.squeeze(1))
            preds = (scores > 0.5).long()
            cm.update(preds, batch["mask"])
            all_scores.append(scores.cpu())
            all_targets.append(batch["mask"].cpu())
    s = torch.cat(all_scores)
    t = torch.cat(all_targets)
    m = cm.as_dict()
    m["auprc"] = auprc(s, t)
    m["ece"] = expected_calibration_error(s, t)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="configs/model/unet_s1s2.yaml")
    p.add_argument("--catalog", default="data/catalog/sen1floods11_events.yaml")
    p.add_argument("--patch-root", default="data/processed/patches")
    p.add_argument("--stats", default="data/processed/norm_stats_sen1floods11.yaml")
    p.add_argument("--workdir-root", default="outputs/leave_one_region_out")
    p.add_argument("--cuda-visible", default="2")
    args = p.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible
    workdir_root = Path(args.workdir_root)
    workdir_root.mkdir(parents=True, exist_ok=True)
    results_path = workdir_root / "results.json"
    all_results: list[dict] = []
    if results_path.exists():
        all_results = json.loads(results_path.read_text())

    for i, test_region in enumerate(ALL_REGIONS):
        done = {r["test_region"] for r in all_results}
        if test_region in done:
            print(f"[{i+1}/{len(ALL_REGIONS)}] SKIP {test_region} (already done)")
            continue
        val_region = ALL_REGIONS[(i + 1) % len(ALL_REGIONS)]
        train_regions = [r for r in ALL_REGIONS if r not in {test_region, val_region}]
        workdir = workdir_root / f"test_{test_region}"
        workdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "-m", "geodisaster.cli", "train",
            "--catalog", args.catalog,
            "--model-config", args.model_config,
            "--patch-root", args.patch_root,
            "--workdir", str(workdir),
            "--val-events", f"sen1floods11_{val_region}",
            "--test-events", f"sen1floods11_{test_region}",
            "--stats", args.stats,
        ]
        for r in train_regions:
            cmd += ["--train-events", f"sen1floods11_{r}"]

        print(f"\n[{i+1}/{len(ALL_REGIONS)}] === test={test_region} val={val_region} "
              f"train=8 regions ===")
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        dt = time.time() - t0
        print(f"  train+test finished in {dt:.0f}s exit={proc.returncode}")
        if proc.returncode != 0:
            print("  stderr tail:", proc.stderr[-500:])

        ckpts = sorted((workdir / "checkpoints").glob("*.ckpt"))
        if not ckpts:
            print(f"  no checkpoint produced for {test_region}, skipping eval")
            continue
        best_ckpt = ckpts[-1]
        try:
            metrics = evaluate_checkpoint(best_ckpt, test_region, args.stats)
        except Exception as e:
            print(f"  eval failed: {e}")
            continue

        row = {
            "test_region": test_region,
            "val_region":  val_region,
            "train_regions": train_regions,
            "ckpt": str(best_ckpt),
            "train_time_s": int(dt),
            "f1": float(metrics["f1"]),
            "iou": float(metrics["iou"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "auprc": float(metrics["auprc"]),
            "ece": float(metrics["ece"]),
            "tp": int(metrics["tp"]),
            "fp": int(metrics["fp"]),
            "fn": int(metrics["fn"]),
            "tn": int(metrics["tn"]),
        }
        all_results.append(row)
        results_path.write_text(json.dumps(all_results, indent=2))
        print(f"  test {test_region}: F1={row['f1']:.4f} IoU={row['iou']:.4f} "
              f"AUPRC={row['auprc']:.4f}")

    print(f"\n=== Leave-one-region-out summary ({len(all_results)} runs) ===")
    print(f"{'Test region':<14} {'F1':>7} {'IoU':>7} {'P':>7} {'R':>7} {'AUPRC':>7}")
    for r in all_results:
        print(f"  {r['test_region']:<12} {r['f1']:>7.4f} {r['iou']:>7.4f} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['auprc']:>7.4f}")
    if all_results:
        avg_f1 = sum(r["f1"] for r in all_results) / len(all_results)
        avg_iou = sum(r["iou"] for r in all_results) / len(all_results)
        print(f"  {'AVERAGE':<12} {avg_f1:>7.4f} {avg_iou:>7.4f}")


if __name__ == "__main__":
    main()
