"""`geodisaster dispatch` — run the Layer 2 neuro-symbolic emergency reasoner."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..dispatch.reasoner import EmergencyReasoner, save_report
from ..utils.logging import get_logger, setup_logging

log = get_logger("cli.dispatch")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser("geodisaster dispatch")
    p.add_argument("--flood-mask", required=True,
                   help="GeoTIFF flood prediction (uint8 mask)")
    p.add_argument("--event-id", default="unknown")
    p.add_argument("--worldpop", default=None)
    p.add_argument("--out", default=None,
                   help="output JSON (default: alongside flood mask)")
    p.add_argument("--briefing", default=None,
                   help="output text briefing (default: alongside flood mask)")
    args = p.parse_args(argv)

    setup_logging()
    reasoner = EmergencyReasoner(
        flood_mask_path=args.flood_mask,
        event_id=args.event_id,
        worldpop_path=args.worldpop,
    )
    report = reasoner.run()

    out_json = args.out or Path(args.flood_mask).with_suffix(".dispatch.json")
    out_brief = args.briefing or Path(args.flood_mask).with_suffix(".briefing.txt")
    save_report(report, out_json, out_brief)
    print(f"\nJSON  → {out_json}")
    print(f"Brief → {out_brief}")
    print("\n" + report.briefing())
    return 0
