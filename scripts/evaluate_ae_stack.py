"""Evaluate the AlphaEarth pre+post + S1 stacked model on USA test set,
update the multi-model comparison JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, ".")
from geodisaster.data.tile import merge_manifests
from geodisaster.datasets import DisasterPatchDataModule, stats_with_fallbacks
from geodisaster.train import DisasterSegLightningModule
from geodisaster.metrics import BinaryConfusion, auprc, expected_calibration_error


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default="outputs/sen1floods11_ae_stack")
    p.add_argument("--stats", default="data/processed/norm_stats_sen1floods11_ae.yaml")
    p.add_argument("--out-comp", default="outputs/sen1floods11_comparison.json")
    args = p.parse_args()

    ckpt_dir = Path(args.workdir) / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        print(f"No checkpoint in {ckpt_dir}")
        return 1
    ckpt = ckpts[-1]
    print(f"Using: {ckpt}")

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    hp = state["hyper_parameters"]
    model_cfg = OmegaConf.create(hp.get("model_cfg", hp.get("model")))
    train_cfg = OmegaConf.create(hp.get("train_cfg", hp.get("train")))
    sources = hp["sources"]
    module = DisasterSegLightningModule(model_cfg, train_cfg, sources)
    module.load_state_dict(state["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.eval().to(device)

    test = merge_manifests([Path("data/processed/patches") / "sen1floods11_USA"])
    norm = stats_with_fallbacks(args.stats, sources)
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
                    batch[k] = v.to(device)
            logits = module(batch)
            scores = torch.sigmoid(logits.squeeze(1))
            preds = (scores > 0.5).long()
            cm.update(preds, batch["mask"])
            all_scores.append(scores.cpu())
            all_targets.append(batch["mask"].cpu())
    s = torch.cat(all_scores); t = torch.cat(all_targets)
    m = cm.as_dict()
    m["auprc"] = auprc(s, t)
    m["ece"] = expected_calibration_error(s, t)

    print("\n=== AlphaEarth pre+post + S1 stack (test USA) ===")
    print(f"  sources: {sources}")
    for k in ["f1", "iou", "precision", "recall", "auprc", "ece"]:
        print(f"  {k:10s} {m[k]:.4f}")
    print(f"  tp={int(m['tp']):>10,d}  fp={int(m['fp']):>10,d}  fn={int(m['fn']):>10,d}")

    # Update comparison JSON
    cp = Path(args.out_comp)
    existing = json.loads(cp.read_text()) if cp.exists() else {}
    existing["AE_pre_post_S1_stack"] = {k: float(v) for k, v in m.items()}
    cp.write_text(json.dumps(existing, indent=2))
    print(f"\nUpdated {cp}")


if __name__ == "__main__":
    sys.exit(main() or 0)
