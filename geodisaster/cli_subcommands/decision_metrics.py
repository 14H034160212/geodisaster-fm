"""`geodisaster decision-metrics` — compute exposure + accessibility for a predicted impact mask."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..decision import (
    affected_buildings, affected_road_length, affected_population, facility_exposure,
    road_disruption_graph, isolated_communities, rescue_priority,
)
from ..utils.io import ensure_dir
from ..utils.logging import get_logger

log = get_logger("cli.decision")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser("geodisaster decision-metrics")
    p.add_argument("--event", required=True)
    p.add_argument("--impact-mask", required=True, help="GeoTIFF binary impact mask")
    p.add_argument("--buildings", default=None)
    p.add_argument("--roads", default=None)
    p.add_argument("--facilities", default=None)
    p.add_argument("--population", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    out_path = Path(args.out or f"outputs/decision/{args.event}/summary.json")
    ensure_dir(out_path.parent)
    summary: dict = {"event": args.event, "metrics": []}

    if args.buildings:
        r = affected_buildings(args.event, args.impact_mask, args.buildings)
        summary["metrics"].append(r.as_dict())
    if args.roads:
        r = affected_road_length(args.event, args.impact_mask, args.roads)
        summary["metrics"].append(r.as_dict())
    if args.population:
        r = affected_population(args.event, args.impact_mask, args.population)
        summary["metrics"].append(r.as_dict())
    if args.facilities:
        r = facility_exposure(args.event, args.impact_mask, args.facilities)
        summary["metrics"].append(r.as_dict())
    if args.roads and args.population:
        _, H, _, disrupted = road_disruption_graph(args.roads, args.impact_mask)
        comps = isolated_communities(H, args.population, args.facilities)
        priority = rescue_priority(comps)
        summary["disrupted_edges"] = int(disrupted.sum())
        summary["components"] = [
            {"id": c.component_id, "n_nodes": c.n_nodes,
             "population": c.population, "has_facility": c.contains_facility}
            for c in priority[:50]
        ]

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("decision_done", event=args.event, out=str(out_path),
             n_metrics=len(summary["metrics"]))
    print(json.dumps(summary, indent=2))
    return 0
