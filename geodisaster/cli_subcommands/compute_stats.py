"""`geodisaster compute-stats` — per-source normalization statistics over a patch split."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..data.catalog import EventCatalog
from ..data.tile import merge_manifests
from ..datasets import compute_norm_stats, save_stats
from ..utils.logging import get_logger

log = get_logger("cli.compute_stats")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser("geodisaster compute-stats")
    p.add_argument("--catalog", default="data/catalog/japan_events.yaml")
    p.add_argument("--patch-root", default="data/processed/patches")
    p.add_argument("--event", action="append", required=True,
                   help="event_id(s) to use for statistics. Use ONLY your training events.")
    p.add_argument("--source", action="append", required=True,
                   help="source name(s), e.g. alphaearth sentinel1 dem")
    p.add_argument("--max-patches", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="data/processed/norm_stats.yaml")
    args = p.parse_args(argv)

    catalog = EventCatalog.load(args.catalog)
    event_ids = set(args.event)
    valid_events = [e for e in catalog if e.event_id in event_ids]
    missing = event_ids - {e.event_id for e in valid_events}
    if missing:
        log.warning("unknown_event_ids", missing=list(missing))

    patch_dirs = [Path(args.patch_root) / e.event_id for e in valid_events
                  if (Path(args.patch_root) / e.event_id).is_dir()]
    if not patch_dirs:
        log.error("no_patch_dirs", note="run `geodisaster tile-dataset` first")
        return 2
    patches = merge_manifests(patch_dirs)
    log.info("computing_stats",
             events=[e.event_id for e in valid_events],
             sources=args.source, n_patches=len(patches),
             max_patches=args.max_patches)

    stats = compute_norm_stats(
        patches=patches,
        sources=args.source,
        max_patches=args.max_patches,
        rng_seed=args.seed,
    )
    out_path = save_stats(stats, args.out)
    log.info("stats_saved", path=str(out_path),
             summary={k: (round(v[0], 4), round(v[1], 4)) for k, v in stats.items()})
    for k, (m, s) in stats.items():
        print(f"  {k:14s} mean={m:.4f}  std={s:.4f}")
    return 0
