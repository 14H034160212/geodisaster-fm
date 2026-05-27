"""Aggregate the (now genuinely multi-seed) AlphaEarth pre+post+S1 few-shot runs.

After the seed bug fix, outputs/few_shot_ae_stack_seed{1234,42,1337}/ hold three
INDEPENDENT seeds. This computes mean +/- std per label fraction for F1/IoU and
writes a summary JSON the blog can show with real confidence intervals.
"""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

import pandas as pd

SEED_DIRS = sorted(glob.glob("outputs/few_shot_ae_stack_seed*"))
OUT = "outputs/few_shot_ae_stack/few_shot_multiseed_summary.json"


def main():
    frames = []
    for d in SEED_DIRS:
        csv = Path(d) / "few_shot_results.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["seed_dir"] = Path(d).name
            frames.append(df)
    if len(frames) < 2:
        print(f"Only {len(frames)} seed CSV(s) present — need >=2 to aggregate.")
        return 1
    alldf = pd.concat(frames, ignore_index=True)

    summary = {"n_seeds": len(frames), "seed_dirs": [Path(d).name for d in SEED_DIRS],
               "by_fraction": {}}
    for frac, g in alldf.groupby("label_fraction"):
        f1s = list(g["test/f1"]); ious = list(g["test/iou"])
        def ms(xs):
            m = statistics.mean(xs)
            s = statistics.stdev(xs) if len(xs) > 1 else 0.0
            return {"mean": round(m, 4), "std": round(s, 4), "n": len(xs)}
        summary["by_fraction"][f"{frac:g}"] = {"f1": ms(f1s), "iou": ms(ious),
                                               "n_train": int(g["n_train"].iloc[0])}

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(summary, indent=2))
    print(f"Aggregated {len(frames)} seeds -> {OUT}")
    for frac, v in summary["by_fraction"].items():
        print(f"  frac {frac:>5}: F1 {v['f1']['mean']:.3f} ± {v['f1']['std']:.3f} "
              f"(n_train={v['n_train']})")


if __name__ == "__main__":
    raise SystemExit(main())
