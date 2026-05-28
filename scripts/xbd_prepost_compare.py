"""Does pre+post change detection beat post-only for xBD building localization,
and is it stable across seeds? In-domain image-level split; for each seed we
train post-only (3ch) and pre+post (6ch) on the SAME split (paired), eval test F1.
Reports mean +/- std over seeds + a paired comparison.
"""
from __future__ import annotations
import argparse, json, sys, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, ".")
from geodisaster.data.tile import merge_manifests
from geodisaster.datasets import DisasterPatchDataModule, stats_with_fallbacks
from geodisaster.train import DisasterSegLightningModule, make_trainer
from geodisaster.metrics import BinaryConfusion, auprc
from geodisaster.utils.io import load_config, ensure_dir
from geodisaster.utils.logging import get_logger, setup_logging
log = get_logger("xbd_prepost")
ROOT = Path("data/processed/patches")


def img_of(pid): return pid.split("_r")[0]


def run_one(prefix, model_cfg_path, disasters, seed, test_frac, stats, defaults):
    import pytorch_lightning as pl
    pl.seed_everything(seed)
    patches = merge_manifests([ROOT / f"{prefix}_{d}" for d in disasters])
    by = defaultdict(list)
    for r in patches: by[img_of(r["patch_id"])].append(r)
    imgs = sorted(by); rng = np.random.RandomState(seed); rng.shuffle(imgs)
    nt = max(1, int(len(imgs) * test_frac))
    test_i, train_i = set(imgs[:nt]), imgs[nt:]
    train_p = [r for im in train_i for r in by[im]]
    test_p = [r for im in by if im in test_i for r in by[im]]
    val_p = train_p[::8]
    sources = ["optical"]; norm = stats_with_fallbacks(stats, sources)
    model_cfg = load_config(model_cfg_path)
    dm = DisasterPatchDataModule(train_patches=train_p, val_patches=val_p, test_patches=test_p,
                                 sources=sources, batch_size=defaults.train.batch_size,
                                 num_workers=int(defaults.train.num_workers), normalize=norm)
    module = DisasterSegLightningModule(model_cfg=model_cfg, train_cfg=defaults.train, sources=sources)
    wd = ensure_dir(f"outputs/xbd_prepost/{prefix}_seed{seed}")
    trainer = make_trainer(defaults.train, workdir=wd)
    trainer.fit(module, datamodule=dm)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.eval().to(dev); cm = BinaryConfusion(); ss, ts = [], []
    with torch.no_grad():
        for b in dm.test_dataloader():
            for k, v in list(b.items()):
                if isinstance(v, torch.Tensor): b[k] = v.to(dev)
            sc = torch.sigmoid(module(b).squeeze(1))
            cm.update((sc > 0.5).long(), b["mask"]); ss.append(sc.cpu()); ts.append(b["mask"].cpu())
    m = cm.as_dict(); m["auprc"] = auprc(torch.cat(ss), torch.cat(ts))
    return {"f1": float(m["f1"]), "iou": float(m["iou"]), "auprc": float(m["auprc"]),
            "n_test_tiles": len(test_p)}


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--disasters", nargs="+", default=["hurricane-harvey", "palu-tsunami",
                    "hurricane-florence", "mexico-earthquake"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accumulate", type=int, default=2)
    ap.add_argument("--out", default="outputs/xbd_prepost/results.json")
    args = ap.parse_args()
    defaults = load_config("configs/default.yaml")
    defaults.train.batch_size = args.batch_size; defaults.train.accumulate_grad_batches = args.accumulate
    arms = {"post_only": ("xbd", "configs/model/unet_xbd.yaml"),
            "pre_post":  ("xbdpp", "configs/model/unet_xbd_pp.yaml")}
    res = {a: [] for a in arms}
    for seed in args.seeds:
        for arm, (prefix, mc) in arms.items():
            r = run_one(prefix, mc, args.disasters, seed, args.test_frac,
                        "data/processed/norm_stats_xbd.yaml", defaults)
            res[arm].append(r["f1"]); log.info("done", arm=arm, seed=seed, f1=round(r["f1"], 4))
    summary = {"disasters": args.disasters, "seeds": args.seeds, "arms": {}}
    for a in arms:
        fs = res[a]; summary["arms"][a] = {"f1_mean": round(statistics.mean(fs), 4),
            "f1_std": round(statistics.stdev(fs), 4) if len(fs) > 1 else 0.0, "f1_seeds": fs}
    gain = summary["arms"]["pre_post"]["f1_mean"] - summary["arms"]["post_only"]["f1_mean"]
    summary["prepost_gain"] = round(gain, 4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\n=== xBD localization: post-only vs pre+post (in-domain, multi-seed) ===")
    for a in arms:
        s = summary["arms"][a]; print(f"  {a:10s} F1 {s['f1_mean']:.3f} ± {s['f1_std']:.3f}  seeds={[round(x,3) for x in s['f1_seeds']]}")
    print(f"  pre+post gain: {gain:+.3f} F1")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
