"""Structured logging helpers."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog
from rich.console import Console
from rich.logging import RichHandler


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    handlers: list[logging.Handler] = [
        RichHandler(console=Console(stderr=True), rich_tracebacks=True, show_path=False)
    ]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        handlers.append(fh)

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        handlers=handlers,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
