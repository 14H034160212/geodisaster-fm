"""`geodisaster download-gee` — pull AlphaEarth / Sentinel / DEM / WorldPop for one or more events."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..data.catalog import EventCatalog, HazardType
from ..data.gee import alphaearth, dem as gee_dem, sentinel1, sentinel2, worldpop
from ..utils.io import load_config, ensure_dir
from ..utils.logging import get_logger

log = get_logger("cli.download_gee")

SOURCES = {
    "alphaearth": alphaearth.download,
    "sentinel1":  sentinel1.download,
    "sentinel2":  sentinel2.download,
    "dem":        gee_dem.download,
    "worldpop":   worldpop.download,
}


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser("geodisaster download-gee")
    p.add_argument("--catalog", default="data/catalog/japan_events.yaml")
    p.add_argument("--data-config", default="configs/data/japan_flood.yaml")
    p.add_argument("--default-config", default="configs/default.yaml")
    p.add_argument("--event", action="append", default=None,
                   help="specific event_id(s); repeatable. default = all matching data-config scope")
    p.add_argument("--source", action="append", default=None,
                   help=f"subset of {sorted(SOURCES)}; default = all listed in data-config.sources")
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    defaults = load_config(args.default_config)
    data_cfg = load_config(args.data_config)
    catalog = EventCatalog.load(args.catalog)

    if args.event:
        wanted = set(args.event)
        events = [e for e in catalog if e.event_id in wanted]
        missing = wanted - {e.event_id for e in events}
        if missing:
            log.warning("unknown_event_ids", missing=list(missing))
    else:
        scope = data_cfg.get("scope", {})
        if scope.get("hazard"):
            catalog = catalog.filter(hazard=HazardType(scope["hazard"]))
        if scope.get("region"):
            catalog = catalog.filter(region=scope["region"])
        if scope.get("country"):
            catalog = catalog.filter(country=scope["country"])
        if scope.get("events"):
            ids = set(scope["events"])
            catalog = EventCatalog([e for e in catalog if e.event_id in ids])
        events = list(catalog)

    sources = args.source or list(data_cfg.get("sources", list(SOURCES)))
    sources = [s for s in sources if s in SOURCES]

    for event in events:
        event_out = ensure_dir(Path(args.out_dir) / event.event_id)
        for src in sources:
            cfg_block = data_cfg.get(src, {}) or {}
            # carry GEE project from defaults
            if defaults.get("gee", {}).get("project") and "project" not in cfg_block:
                cfg_block["project"] = defaults.gee.project
            log.info("downloading", event=event.event_id, source=src)
            if args.dry_run:
                continue
            try:
                SOURCES[src](event, cfg_block, event_out)
            except Exception as e:
                log.error("download_failed", event=event.event_id, source=src, error=str(e))
    return 0
