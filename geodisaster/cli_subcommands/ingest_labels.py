"""`geodisaster ingest-labels` — rasterize official Japan labels + index global datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..data.catalog import EventCatalog, HazardType
from ..data.labels import (
    ingest_gsi_flood, ingest_gsi_landslide,
    ingest_xbd, ingest_openearthmap, ingest_sen1floods11,
)
from ..utils.logging import get_logger

log = get_logger("cli.ingest_labels")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser("geodisaster ingest-labels")
    p.add_argument("--catalog", default="data/catalog/japan_events.yaml")
    p.add_argument("--raw-root", default="data/external/gsi",
                   help="root containing per-event polygon files (GSI/JAXA/MLIT downloads)")
    p.add_argument("--out-root", default="data/processed/labels")
    p.add_argument("--event", action="append", default=None)
    p.add_argument("--global-dataset", choices=["xbd", "openearthmap", "sen1floods11"], default=None)
    p.add_argument("--global-root", default=None)
    p.add_argument("--split", default="train")
    args = p.parse_args(argv)

    if args.global_dataset:
        if not args.global_root:
            print("--global-root required with --global-dataset", flush=True)
            return 2
        if args.global_dataset == "xbd":
            manifest = ingest_xbd(args.global_root, split=args.split)
        elif args.global_dataset == "openearthmap":
            manifest = ingest_openearthmap(args.global_root, split=args.split)
        else:
            manifest = ingest_sen1floods11(args.global_root, split=args.split)
        out = Path(args.out_root) / "global" / f"{manifest.name}_{args.split}.json"
        manifest.save(out)
        log.info("global_manifest_saved", path=str(out), pairs=len(manifest.image_paths))
        return 0

    catalog = EventCatalog.load(args.catalog)
    if args.event:
        wanted = set(args.event)
        events = [e for e in catalog if e.event_id in wanted]
    else:
        events = list(catalog)

    for event in events:
        try:
            if event.hazard == HazardType.FLOOD or event.hazard == HazardType.TYPHOON:
                ingest_gsi_flood(event, args.raw_root, args.out_root)
            elif event.hazard == HazardType.LANDSLIDE:
                ingest_gsi_landslide(event, args.raw_root, args.out_root)
            else:
                log.info("skip_no_default_ingestor", event=event.event_id, hazard=event.hazard.value)
        except FileNotFoundError as e:
            log.warning("missing_input", event=event.event_id, err=str(e))
        except Exception as e:
            log.error("ingest_failed", event=event.event_id, err=str(e))
    return 0
