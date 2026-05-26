"""Unified CLI: `geodisaster <subcommand>`.

Subcommands (filled in as P1+ tasks land):
    list-events            inspect the disaster event catalog
    download-gee           pull AlphaEarth / Sentinel / DEM rasters for an event
    fetch-osm              pull roads/buildings for an event AOI
    fetch-worldpop         pull WorldPop population for an event AOI
    ingest-labels          convert official polygons to per-event rasters
    tile-dataset           build aligned per-event patch dataset
    train                  train a model
    evaluate               evaluate a checkpoint
    run-few-shot           label-fraction sweep
    run-cross-domain       cross-region / cross-event / cross-hazard sweep
    decision-metrics       compute exposure + accessibility metrics
    make-figures           render Fig 1-5 from results

Each subcommand is dispatched to its module. Modules that aren't implemented
yet print a friendly NotImplementedError so the surface is discoverable.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from .utils.logging import get_logger, setup_logging

SUBCOMMANDS = {
    "list-events":        ("geodisaster.cli_subcommands.list_events",   "main"),
    "download-gee":       ("geodisaster.cli_subcommands.download_gee",  "main"),
    "fetch-osm":          ("geodisaster.cli_subcommands.fetch_osm",     "main"),
    "fetch-worldpop":     ("geodisaster.cli_subcommands.fetch_worldpop","main"),
    "ingest-labels":      ("geodisaster.cli_subcommands.ingest_labels", "main"),
    "ingest-sen1floods11": ("geodisaster.cli_subcommands.ingest_sen1floods11", "main"),
    "tile-dataset":       ("geodisaster.cli_subcommands.tile_dataset",  "main"),
    "compute-stats":      ("geodisaster.cli_subcommands.compute_stats", "main"),
    "train":              ("geodisaster.cli_subcommands.train",         "main"),
    "evaluate":           ("geodisaster.cli_subcommands.evaluate",      "main"),
    "run-few-shot":       ("geodisaster.cli_subcommands.run_few_shot",  "main"),
    "run-cross-domain":   ("geodisaster.cli_subcommands.run_cross_domain","main"),
    "decision-metrics":   ("geodisaster.cli_subcommands.decision_metrics","main"),
    "make-figures":       ("geodisaster.cli_subcommands.make_figures",  "main"),
    "build-report":       ("geodisaster.cli_subcommands.build_report",  "main"),
    "build-blog":         ("geodisaster.cli_subcommands.build_blog",    "main"),
    "dispatch":           ("geodisaster.cli_subcommands.dispatch",      "main"),
    "smoke":              ("geodisaster.cli_subcommands.smoke",         "main"),
}


def _print_help() -> None:
    print("geodisaster <subcommand> [args...]\n")
    print("Available subcommands:")
    for name in SUBCOMMANDS:
        print(f"  {name}")
    print("\nRun `geodisaster <subcommand> --help` for per-subcommand options.")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        _print_help()
        return 0

    subcommand, rest = argv[0], argv[1:]
    if subcommand not in SUBCOMMANDS:
        print(f"unknown subcommand: {subcommand}", file=sys.stderr)
        _print_help()
        return 2

    setup_logging()
    log = get_logger("cli")
    module_path, fn_name = SUBCOMMANDS[subcommand]
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        log.warning("subcommand_not_implemented", subcommand=subcommand,
                    note="module placeholder will be added in its priority slot")
        print(f"[{subcommand}] not implemented yet — see PLAN.md", file=sys.stderr)
        return 64
    fn = getattr(module, fn_name)
    return int(fn(rest) or 0)


if __name__ == "__main__":
    sys.exit(main())
