"""Logging configuration for the dsetgen pipeline.

Provides a structured JSON logger (via ``structlog`` if available) and
falls back to stdlib ``logging`` with a clean format otherwise.

Usage
-----
Call ``setup_logging()`` once at startup:

>>> from dsetgen.utils.logging import setup_logging
>>> setup_logging(level="DEBUG")
"""

from __future__ import annotations

import logging
import sys


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
) -> None:
    """Configure the root logger for the framework.

    Parameters
    ----------
    level:
        Log level name (``DEBUG``, ``INFO``, ``WARNING``, etc.).
    json_output:
        If ``True`` and ``structlog`` is installed, emit JSON lines.
        Otherwise fall back to a human-readable format.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if json_output:
        try:
            import structlog

            structlog.configure(
                processors=[
                    structlog.contextvars.merge_contextvars,
                    structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(),
                ],
                wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
                logger_factory=structlog.PrintLoggerFactory(),
            )
            return
        except ImportError:
            pass  # Fall through to stdlib

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "pdfplumber", "pdfminer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
