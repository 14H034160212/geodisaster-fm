"""`geodisaster ingest-sen1floods11` — Sen1Floods11 -> GeoDisaster patch format."""
from __future__ import annotations

from ..data.labels.sen1floods11_bridge import main as bridge_main


def main(argv: list[str]) -> int:
    return bridge_main(argv)
