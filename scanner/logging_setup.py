"""Logging configuration.

Centralised so every module can simply call ``logging.getLogger(__name__)`` and
inherit consistent formatting, level and (optional) file output.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final, TextIO

_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES: Final[int] = 5 * 1024 * 1024
_BACKUP_COUNT: Final[int] = 3


def _utf8_stdout() -> TextIO:
    """Return stdout switched to UTF-8 where the platform allows it.

    On Windows, a redirected stdout defaults to the ANSI code page (cp1252),
    which mangles the em-dashes in log text and cannot encode the 🟢/🔴 markers
    in alert lines at all. ``errors="replace"`` keeps output flowing on any
    stream that still cannot represent a character.
    """
    stream: TextIO = sys.stdout
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass  # detached or non-reconfigurable stream; fall through as-is
    return stream


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Install console (and optionally rotating file) handlers on the root logger."""
    resolved = getattr(logging, level.upper(), logging.INFO)
    if not isinstance(resolved, int):
        resolved = logging.INFO

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(resolved)
    for handler in list(root.handlers):  # make re-configuration idempotent
        root.removeHandler(handler)

    console = logging.StreamHandler(stream=_utf8_stdout())
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # CCXT and urllib3 are chatty at DEBUG and drown out our own output.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
