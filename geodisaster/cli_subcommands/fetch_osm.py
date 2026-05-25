"""`geodisaster fetch-osm` — roads / buildings / critical facilities for each event AOI."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..data.catalog import EventCatalog, HazardType
from ..data import osm as osm_mod
from ..utils.io import load_config, ensure_dir
from ..utils.logging import get_logger

log = get_logger("cli.fetch_osm")

LAYERS = {
    "roads":      osm_mod.fetch_roads,
    "buildings":  osm_mod.fetch_buildings,
    "facilities": osm_mod.fetch_facilities,
}


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser("geodisaster fetch-osm")
    p.add_argument("--catalog", default="data/catalog/japan_events.yaml")
    p.add_argument("--data-config", default="configs/data/japan_flood.yaml")
    p.add_argument("--event", action="append", default=None)
    p.add_argument("--layer", action="append", default=None,
                   help="subset of: roads, buildings, facilities")
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.data_config)
    catalog = EventCatalog.load(args.catalog)
    if args.event:
        wanted = set(args.event)
        events = [e for e in catalog if e.event_id in wanted]
    else:
        scope = cfg.get("scope", {})
        if scope.get("hazard"):
            catalog = catalog.filter(hazard=HazardType(scope["hazard"]))
        if scope.get("events"):
            ids = set(scope["events"])
            catalog = EventCatalog([e for e in catalog if e.event_id in ids])
        events = list(catalog)
    layers = args.layer or [l for l in ("roads", "buildings", "facilities")
                            if cfg.get(f"osm_{l}") is not None or l == "facilities"]

    for event in events:
        event_out = ensure_dir(Path(args.out_dir) / event.event_id)
        for layer in layers:
            if layer not in LAYERS:
                continue
            sub_cfg = cfg.get(f"osm_{layer}", {}) or {}
            log.info("osm_fetching", event=event.event_id, layer=layer)
            if args.dry_run:
                continue
            try:
                LAYERS[layer](event, sub_cfg, event_out)
            except Exception as e:
                log.error("osm_failed", event=event.event_id, layer=layer, error=str(e))
    return 0
