"""Crypto Scanner Bot v2 — entrypoint.

Monitors a configurable set of USDT pairs (Bybit by default) and sends an alert
when an engulfing reversal closes *in agreement with* the SMA-200 trend regime
and on above-average volume.

Usage::

    python main.py                       # run continuously
    python main.py --once                # single pass, then exit
    python main.py --dry-run             # log alerts instead of sending them
    python main.py --timeframe 1h --symbols BTC/USDT,ETH/USDT
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from typing import Final, Sequence

from scanner.bot import ScannerBot
from scanner.config import ConfigError, Settings, parse_symbols, validate_timeframe_token
from scanner.exchange import MarketDataError
from scanner.logging_setup import configure_logging

logger: Final[logging.Logger] = logging.getLogger("scanner.main")

EXIT_OK: Final[int] = 0
EXIT_CONFIG_ERROR: Final[int] = 2
EXIT_RUNTIME_ERROR: Final[int] = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="crypto-scanner",
        description=(
            "Scan exchange pairs for engulfing reversals confirmed by the SMA-200 "
            "trend regime and above-average volume."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan pass and exit (useful for cron or smoke tests).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log alerts to the console instead of sending them to Telegram.",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Override the configured timeframe, e.g. 5m, 15m, 1h, 4h.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated override of the watchlist, e.g. 'BTC/USDT,ETH/USDT'.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override the configured log level.",
    )
    return parser.parse_args(argv)


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Layer CLI flags on top of environment-derived settings.

    ``--dry-run`` is deliberately absent: it is passed into
    :meth:`Settings.from_env` instead, since it governs whether credentials are
    required and so must be known during validation, not after it.
    """
    changes: dict[str, object] = {}

    if args.timeframe:
        changes["timeframe"] = validate_timeframe_token(args.timeframe)
    if args.symbols:
        changes["symbols"] = parse_symbols(args.symbols)
    if args.log_level:
        changes["log_level"] = args.log_level

    if not changes:
        return settings
    return dataclasses.replace(settings, **changes)  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        settings = apply_overrides(
            Settings.from_env(force_dry_run=args.dry_run), args
        )
    except ConfigError as exc:
        # Logging is not configured yet, so write directly to stderr.
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    configure_logging(settings.log_level, settings.log_file)

    bot: ScannerBot | None = None
    try:
        bot = ScannerBot(settings)
        bot.install_signal_handlers()
        bot.startup_checks()

        if args.once:
            signals = bot.scan_once()
            logger.info("Single pass complete: %d signal(s).", len(signals))
        else:
            bot.run_forever()
    except MarketDataError as exc:
        logger.error("Market data unavailable: %s", exc)
        return EXIT_RUNTIME_ERROR
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception:
        logger.exception("Fatal error — shutting down.")
        return EXIT_RUNTIME_ERROR
    finally:
        if bot is not None:
            bot.close()

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
