"""Crypto Scanner Bot v3.1 (Institutional MTF SMC) — entrypoint.

Watches a configurable set of pairs for extreme order blocks with real
displacement, then confirms entries on a lower timeframe before building an
order. Use --simulate to replay the pipeline over history and report the funnel.

v3.1 is deprecated (see README). --backtest-v5 evaluates the V5.0 research
model — daily Donchian trend-following — on fetched daily history; not live.

Usage::

    python main.py                       # run continuously
    python main.py --once                # single pass, then exit
    python main.py --dry-run             # log alerts instead of sending them
    python main.py --timeframe 1h --symbols BTC/USDT,ETH/USDT
    python main.py --backtest-v5                  # V5.0 research backtest, all history
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
            "Scan for Smart Money order blocks validated by fair value gaps, "
            "sized to a fixed fraction of account equity."
        ),
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "Replay the pipeline over historical candles, print the funnel and "
            "signal-frequency report, then exit. Implies --dry-run."
        ),
    )
    parser.add_argument(
        "--backtest-v5",
        action="store_true",
        help=(
            "Backtest the V5.0 daily trend-following research model on fetched "
            "daily history (10- and 20-day exits, spot costs), then exit. "
            "Implies --dry-run."
        ),
    )
    parser.add_argument(
        "--history",
        type=int,
        default=None,
        help=(
            "Candles per symbol for --simulate (default: CANDLE_LIMIT) or daily "
            "candles for --backtest-v5 (default: 3600, i.e. all Binance history)."
        ),
    )
    parser.add_argument(
        "--entries",
        choices=["table", "full", "csv", "none"],
        default="full",
        help=(
            "How to list the confirmed entries from --simulate: 'full' adds a "
            "per-entry breakdown, 'csv' prints machine-readable rows."
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


def run_trend_backtest(bot: ScannerBot, *, bars: int) -> None:
    """Fetch daily history and print the V5.0 backtest for both exit channels,
    benchmarked against BTC buy-and-hold from the first tradable day."""
    from scanner.trend import TrendParams, trend_frame
    from scanner.trend_backtest import (
        SPOT_COSTS,
        all_trades,
        buy_and_hold,
        render,
        simulate_portfolio,
    )

    frames = bot.fetch_history(bars=bars, timeframe="1d")
    if not frames:
        raise MarketDataError("no daily history could be fetched")
    name = "BTC/USDT" if "BTC/USDT" in frames else next(iter(frames))
    anchor = frames[name]
    ready = trend_frame(anchor, TrendParams()).dropna(subset=["sma"])
    if ready.empty:
        raise MarketDataError(f"{name} has too little history for a 200-day average")
    start = int(ready["timestamp"].iloc[0])
    benchmark = buy_and_hold(anchor, SPOT_COSTS, start_ms=start)
    for exit_channel in (10, 20):
        params = TrendParams(exit_channel=exit_channel)
        portfolio = simulate_portfolio(frames, params, SPOT_COSTS, start_ms=start)
        print(render(all_trades(frames, params), portfolio, benchmark,
                     params=params, benchmark_name=name.split("/")[0]))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        settings = apply_overrides(
            Settings.from_env(
                force_dry_run=args.dry_run or args.simulate or args.backtest_v5
            ),
            args,
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

        if args.backtest_v5:
            run_trend_backtest(bot, bars=args.history or 3_600)
        elif args.simulate:
            report = bot.simulate(candle_limit=args.history)
            print(report.render())
            if args.entries == "csv":
                print()
                print(report.entries_csv())
            elif args.entries != "none":
                print()
                print(report.render_entries(detailed=args.entries == "full"))
            logger.info("Filter funnel: %s", report.funnel_line())
        elif args.once:
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
