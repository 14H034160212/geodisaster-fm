"""`geodisaster tile-dataset` — slice per-event rasters into aligned patches."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..data.catalog import EventCatalog
from ..data.tile import tile_event
from ..utils.io import load_config
from ..utils.logging import get_logger

log = get_logger("cli.tile")


def _discover_event_inputs(raw_dir: Path) -> tuple[dict[str, Path], Path | None]:
    """Map source name -> tif path from a per-event raw directory.

    Conventions match the GEE downloaders + label ingestion outputs:
        {event_id}_alphaearth_YYYY.tif    -> alphaearth
        {event_id}_s1_post.tif            -> sentinel1 (prefer post for impact)
        {event_id}_s1_pre.tif             -> sentinel1_pre (optional)
        {event_id}_s2_post.tif            -> sentinel2
        {event_id}_dem.tif                -> dem
        {event_id}_worldpop_*.tif         -> worldpop
        {event_id}_label_*.tif            -> label
    """
    sources: dict[str, Path] = {}
    label: Path | None = None
    for f in sorted(raw_dir.glob("*.tif")):
        n = f.name
        if "_alphaearth_" in n:
            sources["alphaearth"] = f
        elif n.endswith("_s1_post.tif"):
            sources["sentinel1"] = f
        elif n.endswith("_s1_pre.tif"):
            sources["sentinel1_pre"] = f
        elif n.endswith("_s2_post.tif"):
            sources["sentinel2"] = f
        elif n.endswith("_dem.tif"):
            sources["dem"] = f
        elif "_worldpop_" in n:
            sources["worldpop"] = f
        elif "_label_" in n:
            label = f
    return sources, label


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser("geodisaster tile-dataset")
    p.add_argument("--catalog", default="data/catalog/japan_events.yaml")
    p.add_argument("--default-config", default="configs/default.yaml")
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--label-root", default="data/processed/labels")
    p.add_argument("--out-root", default="data/processed/patches")
    p.add_argument("--event", action="append", default=None)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--min-pos-fraction", type=float, default=0.0)
    args = p.parse_args(argv)

    defaults = load_config(args.default_config)
    size = args.size or int(defaults.data.patch.size_px)
    stride = args.stride or int(defaults.data.patch.stride_px)
    catalog = EventCatalog.load(args.catalog)
    if args.event:
        events = [e for e in catalog if e.event_id in set(args.event)]
    else:
        events = list(catalog)

    for event in events:
        raw_dir = Path(args.raw_root) / event.event_id
        if not raw_dir.is_dir():
            log.warning("no_raw_dir", event=event.event_id, path=str(raw_dir))
            continue
        sources, raw_label = _discover_event_inputs(raw_dir)
        label_dir = Path(args.label_root) / event.event_id
        label_path: Path | None = raw_label
        if label_dir.is_dir():
            for f in label_dir.glob("*_label_*.tif"):
                label_path = f
                break
        if "alphaearth" not in sources:
            log.warning("no_alphaearth", event=event.event_id,
                        note="alphaearth is the reference grid; skipping")
            continue
        tile_event(
            event=event,
            sources={k: str(v) for k, v in sources.items()},
            label_path=str(label_path) if label_path else None,
            out_dir=args.out_root,
            size=size, stride=stride,
            min_pos_fraction=args.min_pos_fraction,
        )
    return 0
