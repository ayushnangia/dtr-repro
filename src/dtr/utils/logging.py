"""Logging configuration using rich."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger with rich handler."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger("dtr")


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(f"dtr.{name}")
