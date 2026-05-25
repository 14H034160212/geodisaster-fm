"""`geodisaster fetch-worldpop` — pull WorldPop population for each event AOI via GEE."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..data.catalog import EventCatalog, HazardType
from ..data.gee import worldpop
from ..utils.io import load_config, ensure_dir
from ..utils.logging import get_logger

log = get_logger("cli.worldpop")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser("geodisaster fetch-worldpop")
    p.add_argument("--catalog", default="data/catalog/japan_events.yaml")
    p.add_argument("--data-config", default="configs/data/japan_flood.yaml")
    p.add_argument("--default-config", default="configs/default.yaml")
    p.add_argument("--event", action="append", default=None)
    p.add_argument("--out-dir", default="data/raw")
    args = p.parse_args(argv)

    defaults = load_config(args.default_config)
    data_cfg = load_config(args.data_config)
    catalog = EventCatalog.load(args.catalog)
    if args.event:
        wanted = set(args.event)
        events = [e for e in catalog if e.event_id in wanted]
    else:
        scope = data_cfg.get("scope", {})
        if scope.get("hazard"):
            catalog = catalog.filter(hazard=HazardType(scope["hazard"]))
        if scope.get("events"):
            ids = set(scope["events"])
            catalog = EventCatalog([e for e in catalog if e.event_id in ids])
        events = list(catalog)

    cfg_block = data_cfg.get("worldpop", {}) or {}
    if defaults.gee.get("project") and "project" not in cfg_block:
        cfg_block["project"] = defaults.gee.project

    for event in events:
        event_out = ensure_dir(Path(args.out_dir) / event.event_id)
        log.info("worldpop_fetching", event=event.event_id)
        try:
            worldpop.download(event, cfg_block, event_out)
        except Exception as e:
            log.error("worldpop_failed", event=event.event_id, err=str(e))
    return 0
