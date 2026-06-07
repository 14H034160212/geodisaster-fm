"""Aggregate xBD pre/post LOHO across the available seeds.

Reads outputs/xbd_prepost_loho/results_seed*.json (each is a list of per-hazard
dicts with an `f1` field) and rewrites outputs/xbd_prepost_loho/aggregate.json
with the up-to-date seed count. Run after any new seed is added.

This replaces the hand-aggregated 2-seed file with whatever seeds are present on
disk — so dropping a `results_seed2.json` and rerunning here lifts the
paper-quoted statistic from 2-seed to 3-seed automatically.
"""
from __future__ import annotations
import json, re
from pathlib import Path
from statistics import mean, pstdev

DIR = Path("outputs/xbd_prepost_loho")
OUT = DIR / "aggregate.json"
SEED_RE = re.compile(r"results_seed(\d+)\.json$")


def main():
    seed_files = sorted(DIR.glob("results_seed*.json"),
                        key=lambda p: int(SEED_RE.search(p.name).group(1)))
    assert seed_files, f"no seed files in {DIR}"
    seed_ids = [int(SEED_RE.search(p.name).group(1)) for p in seed_files]

    # Load per-hazard F1 per seed
    per_hazard = {}                     # hazard -> [f1_seed0, f1_seed1, ...]
    hazard_order = None
    for f in seed_files:
        rows = json.loads(f.read_text())
        if hazard_order is None:
            hazard_order = [r["test_hazard"] for r in rows]
        for r in rows:
            per_hazard.setdefault(r["test_hazard"], []).append(float(r["f1"]))

    # Preserve the original single-seed post-only numbers (these are not re-run
    # here — they come from a separate single-seed post-only LOHO experiment).
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    post_only = prev.get("post_only_single_seed", {})

    pre_post = {}
    for h in hazard_order:
        vs = per_hazard[h]
        pre_post[h] = {
            "seeds": vs,
            "mean": round(mean(vs), 4),
            "std": round(pstdev(vs), 4) if len(vs) > 1 else 0.0,
            "n": len(vs),
        }

    pre_post_mean = round(mean(v["mean"] for v in pre_post.values()), 4)
    post_only_mean = round(mean(post_only.get(h, 0.0) for h in hazard_order), 4) if post_only else None
    gain = round(pre_post_mean - post_only_mean, 4) if post_only else None

    out = {
        "hazards": hazard_order,
        "seed_ids": seed_ids,
        "n_seeds": len(seed_ids),
        "post_only_single_seed": post_only,
        f"pre_post_{len(seed_ids)}seed": pre_post,
        "post_only_mean": post_only_mean,
        "pre_post_mean": pre_post_mean,
        "gain": gain,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Saved {OUT}")
    print(f"  seeds = {seed_ids} (n={len(seed_ids)})")
    print(f"  hazards          post_only  pre_post (mean ± std, n)")
    for h in hazard_order:
        po = post_only.get(h, float("nan"))
        v = pre_post[h]
        diff = v["mean"] - po if po == po else float("nan")
        print(f"  {h:22s} {po:.4f}    {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n']})   Δ={diff:+.4f}")
    if post_only_mean is not None:
        print(f"\n  mean F1 across hazards:")
        print(f"    post-only   = {post_only_mean:.4f}")
        print(f"    pre/post    = {pre_post_mean:.4f}")
        print(f"    gain        = {gain:+.4f}")


if __name__ == "__main__":
    main()
