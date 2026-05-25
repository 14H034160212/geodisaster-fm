"""`geodisaster smoke` — synthetic-data end-to-end smoke test."""
from __future__ import annotations

from ..smoke import main as smoke_main


def main(argv: list[str]) -> int:
    return smoke_main(argv)
