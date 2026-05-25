"""`geodisaster list-events` — inspect the disaster event catalog."""
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..data.catalog import EventCatalog, HazardType


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser("geodisaster list-events")
    p.add_argument("--catalog", default="data/catalog/japan_events.yaml")
    p.add_argument("--hazard", choices=[h.value for h in HazardType], default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--country", default=None)
    p.add_argument("--format", choices=["table", "csv", "yaml"], default="table")
    args = p.parse_args(argv)

    cat = EventCatalog.load(args.catalog)
    if args.hazard:
        cat = cat.filter(hazard=HazardType(args.hazard))
    if args.region:
        cat = cat.filter(region=args.region)
    if args.country:
        cat = cat.filter(country=args.country)

    if args.format == "csv":
        cat.to_dataframe().to_csv(Path("/dev/stdout"), index=False)
        return 0
    if args.format == "yaml":
        for e in cat:
            print(f"- {e.event_id}: {e.name} ({e.hazard.value}) — {e.region}")
        return 0

    console = Console()
    t = Table(title=f"Disaster events ({len(cat)})")
    t.add_column("event_id", style="cyan")
    t.add_column("hazard", style="magenta")
    t.add_column("region")
    t.add_column("event_date")
    t.add_column("bbox")
    for e in cat:
        bbox = (
            f"{e.bbox[0]:.2f},{e.bbox[1]:.2f},{e.bbox[2]:.2f},{e.bbox[3]:.2f}"
            if e.bbox else "—"
        )
        t.add_row(
            e.event_id,
            e.hazard.value,
            e.region,
            str(e.event_date) if e.event_date else "—",
            bbox,
        )
    console.print(t)
    return 0
