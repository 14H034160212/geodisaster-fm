"""`geodisaster make-figures` — render Fig 1-5 from cached experiment outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.catalog import EventCatalog
from ..viz import (
    render_fig1, render_fig2, render_fig3, render_fig4, render_fig5,
    write_manifest,
)
from ..utils.logging import get_logger

log = get_logger("cli.figures")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser("geodisaster make-figures")
    p.add_argument("--catalog", default="data/catalog/japan_events.yaml")
    p.add_argument("--out-dir", default="outputs/figures")
    p.add_argument("--few-shot-csv", action="append", default=None,
                   help="label=path pairs for Fig 3 overlay (e.g. 'AlphaEarth=outputs/few_shot_ae/few_shot_results.csv')")
    p.add_argument("--cross-domain-csv", default=None)
    p.add_argument("--impact-mask", default=None, help="for Fig 5")
    p.add_argument("--buildings", default=None)
    p.add_argument("--roads", default=None)
    p.add_argument("--reproducibility", action="store_true",
                   help="also write reproducibility manifest")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = EventCatalog.load(args.catalog)

    artefacts: dict[str, str] = {}

    p1 = render_fig1(out_dir / "fig1_paradigm.png")
    artefacts["fig1"] = str(p1)

    p2 = render_fig2(catalog, out_dir / "fig2_japan_events.png")
    artefacts["fig2"] = str(p2)

    if args.few_shot_csv:
        results = {}
        for spec in args.few_shot_csv:
            if "=" not in spec:
                continue
            k, v = spec.split("=", 1)
            results[k] = v
        if results:
            p3 = render_fig3(results, out_dir / "fig3_fewshot.png")
            artefacts["fig3"] = str(p3)

    if args.cross_domain_csv:
        p4 = render_fig4(args.cross_domain_csv, out_dir / "fig4_xdomain.png")
        artefacts["fig4"] = str(p4)

    if args.impact_mask:
        p5 = render_fig5(args.impact_mask, args.buildings, args.roads,
                         out_dir / "fig5_decision.png")
        artefacts["fig5"] = str(p5)

    if args.reproducibility:
        manifest = write_manifest(
            root=Path("."), artefacts=artefacts,
            out_path=out_dir / "reproducibility.json",
        )
        log.info("manifest_written", path=str(manifest))

    print(json.dumps(artefacts, indent=2))
    return 0
