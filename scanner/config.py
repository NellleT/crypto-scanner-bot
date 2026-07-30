"""Configuration loading and validation.

All runtime configuration is sourced from environment variables (typically
populated from a local ``.env`` file). Nothing here reads from the network, so
settings can be validated at process start and fail fast on misconfiguration.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

from scanner.indicators import (
    DEFAULT_SMA_PERIOD,
    DEFAULT_VOLUME_SMA_PERIOD,
    required_candles,
)
from scanner.patterns import DEFAULT_MIN_BODY_RATIO
from scanner.risk import (
    DEFAULT_RR_TARGETS,
    DEFAULT_STOP_BUFFER_PCT,
    DEFAULT_STRUCTURAL_LOOKBACK,
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

DEFAULT_SYMBOLS: Final[tuple[str, ...]] = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
DEFAULT_TIMEFRAME: Final[str] = "4h"
DEFAULT_CANDLE_LIMIT: Final[int] = 300

#: Default venue. Bybit rather than Binance because Binance geo-blocks the IP
#: ranges GitHub Actions runners use and answers with HTTP 451, so a scheduled
#: workflow can never fetch candles from it. Bybit serves the same unified
#: symbols, timeframes and base-asset volumes, so the switch is configuration
#: only. Override with EXCHANGE_ID for any CCXT venue with public OHLCV.
DEFAULT_EXCHANGE_ID: Final[str] = "bybit"

#: Most venues, Bybit and Binance included, cap one OHLCV request at 1000 candles.
MAX_CANDLE_LIMIT: Final[int] = 1000

# Timeframe tokens: <int><unit> where unit is m/h/d/w/M.
_TIMEFRAME_RE: Final[re.Pattern[str]] = re.compile(r"^\d+[mhdwM]$")

_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "y", "on"})
_FALSEY: Final[frozenset[str]] = frozenset({"0", "false", "no", "n", "off"})


class ConfigError(RuntimeError):
    """Raised when the environment is missing or contains invalid settings."""


def _get_str(key: str, default: str | None = None, *, required: bool = False) -> str:
    raw = os.getenv(key)
    value = raw.strip() if raw is not None else ""
    if not value:
        if required:
            raise ConfigError(
                f"Missing required environment variable {key!r}. "
                "Copy .env.example to .env and fill it in."
            )
        return default if default is not None else ""
    return value


def _get_int(
    key: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = _get_str(key)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}.") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {value}.")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{key} must be <= {maximum}, got {value}.")
    return value


def _get_float(
    key: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = _get_str(key)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}.") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {value}.")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{key} must be <= {maximum}, got {value}.")
    return value


def _get_bool(key: str, default: bool) -> bool:
    raw = _get_str(key).lower()
    if not raw:
        return default
    if raw in _TRUTHY:
        return True
    if raw in _FALSEY:
        return False
    raise ConfigError(f"{key} must be a boolean-like value, got {raw!r}.")


def parse_rr_targets(raw: str) -> tuple[float, ...]:
    """Parse a comma-separated reward-to-risk ladder, e.g. ``2,3``.

    Returned in ascending order so the alert always lists nearer targets first,
    regardless of how they were written.
    """
    targets: list[float] = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ConfigError(
                f"RR_TARGETS must be a comma-separated list of numbers, got {token!r}."
            ) from exc
        if value <= 0.0:
            raise ConfigError(f"RR_TARGETS entries must be > 0, got {value:g}.")
        targets.append(value)

    if not targets:
        raise ConfigError("RR_TARGETS resolved to an empty list.")
    return tuple(sorted(set(targets)))


def validate_timeframe_token(timeframe: str) -> str:
    """Return ``timeframe`` if it is a well-formed token, else raise ``ConfigError``.

    Shared by the environment and ``--timeframe`` paths so a CLI override gets
    the same check as a ``.env`` value.
    """
    if not _TIMEFRAME_RE.match(timeframe):
        raise ConfigError(
            f"TIMEFRAME {timeframe!r} is not a valid token "
            "(expected e.g. '1m', '15m', '4h', '1d')."
        )
    return timeframe


def parse_symbols(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated symbol list, preserving order and de-duplicating."""
    seen: dict[str, None] = {}
    for chunk in raw.split(","):
        symbol = chunk.strip().upper()
        if not symbol:
            continue
        if "/" not in symbol:
            raise ConfigError(
                f"Symbol {symbol!r} must use CCXT unified format, e.g. 'BTC/USDT'."
            )
        seen.setdefault(symbol, None)
    if not seen:
        raise ConfigError("SYMBOLS resolved to an empty list.")
    return tuple(seen)


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable, fully validated runtime configuration."""

    telegram_bot_token: str
    telegram_chat_id: str
    symbols: tuple[str, ...]
    timeframe: str
    exchange_id: str
    candle_limit: int
    sma_period: int
    volume_sma_period: int
    min_body_ratio: float
    require_volume_expansion: bool
    structural_lookback: int
    stop_buffer_pct: float
    rr_targets: tuple[float, ...]
    poll_buffer_seconds: float
    request_delay_seconds: float
    max_retries: int
    retry_backoff_seconds: float
    http_timeout_seconds: float
    log_level: str
    log_file: Path | None
    dry_run: bool

    @classmethod
    def from_env(
        cls,
        *,
        env_file: Path | None = None,
        force_dry_run: bool = False,
    ) -> "Settings":
        """Load settings from the process environment, seeded by a ``.env`` file.

        Existing environment variables always win over ``.env`` values, which is
        what you want when running under systemd, Docker or CI.

        ``force_dry_run`` carries the ``--dry-run`` CLI flag. It must be known
        here rather than applied afterwards, because it decides whether Telegram
        credentials are mandatory — a dry run has to work before the user has
        set any up.
        """
        dotenv_path = env_file if env_file is not None else PROJECT_ROOT / ".env"
        if dotenv_path.is_file():
            load_dotenv(dotenv_path=dotenv_path, override=False)

        dry_run = force_dry_run or _get_bool("DRY_RUN", False)

        timeframe = validate_timeframe_token(_get_str("TIMEFRAME", DEFAULT_TIMEFRAME))

        symbols_raw = _get_str("SYMBOLS", ",".join(DEFAULT_SYMBOLS))

        sma_period = _get_int("SMA_PERIOD", DEFAULT_SMA_PERIOD, minimum=2, maximum=1000)
        volume_sma_period = _get_int(
            "VOLUME_SMA_PERIOD", DEFAULT_VOLUME_SMA_PERIOD, minimum=2, maximum=1000
        )
        structural_lookback = _get_int(
            "STRUCTURAL_LOOKBACK", DEFAULT_STRUCTURAL_LOOKBACK, minimum=2, maximum=1000
        )
        candle_limit = _get_int(
            "CANDLE_LIMIT",
            DEFAULT_CANDLE_LIMIT,
            minimum=3,
            maximum=MAX_CANDLE_LIMIT,
        )

        # The newest bar in a response is usually still forming and gets dropped
        # before analysis, so one extra candle must be fetched on top of what the
        # indicators and the structural lookback need.
        minimum_limit = (
            max(required_candles(sma_period, volume_sma_period), structural_lookback) + 1
        )
        if candle_limit < minimum_limit:
            raise ConfigError(
                f"CANDLE_LIMIT={candle_limit} is too small for SMA_PERIOD={sma_period}, "
                f"VOLUME_SMA_PERIOD={volume_sma_period} and "
                f"STRUCTURAL_LOOKBACK={structural_lookback}: at least {minimum_limit} "
                "candles are required, otherwise every signal is discarded as "
                "not-warmed-up. Raise CANDLE_LIMIT or lower the periods."
            )

        rr_targets = parse_rr_targets(
            _get_str("RR_TARGETS", ",".join(f"{r:g}" for r in DEFAULT_RR_TARGETS))
        )

        log_file_raw = _get_str("LOG_FILE")
        log_file = Path(log_file_raw).expanduser() if log_file_raw else None
        if log_file is not None and not log_file.is_absolute():
            log_file = PROJECT_ROOT / log_file

        return cls(
            # Credentials are only mandatory when we actually intend to send.
            telegram_bot_token=_get_str("TELEGRAM_BOT_TOKEN", required=not dry_run),
            telegram_chat_id=_get_str("TELEGRAM_CHAT_ID", required=not dry_run),
            symbols=parse_symbols(symbols_raw),
            timeframe=timeframe,
            exchange_id=_get_str("EXCHANGE_ID", DEFAULT_EXCHANGE_ID).lower(),
            candle_limit=candle_limit,
            sma_period=sma_period,
            volume_sma_period=volume_sma_period,
            min_body_ratio=_get_float(
                "MIN_BODY_RATIO",
                DEFAULT_MIN_BODY_RATIO,
                minimum=0.0,
                maximum=1.0,
            ),
            require_volume_expansion=_get_bool("REQUIRE_VOLUME_EXPANSION", True),
            structural_lookback=structural_lookback,
            stop_buffer_pct=_get_float(
                "STOP_BUFFER_PCT",
                DEFAULT_STOP_BUFFER_PCT,
                minimum=0.0,
                maximum=10.0,
            ),
            rr_targets=rr_targets,
            poll_buffer_seconds=_get_float("POLL_BUFFER_SECONDS", 10.0, minimum=0.0),
            request_delay_seconds=_get_float("REQUEST_DELAY_SECONDS", 0.25, minimum=0.0),
            max_retries=_get_int("MAX_RETRIES", 3, minimum=0),
            retry_backoff_seconds=_get_float("RETRY_BACKOFF_SECONDS", 2.0, minimum=0.1),
            http_timeout_seconds=_get_float("HTTP_TIMEOUT_SECONDS", 15.0, minimum=1.0),
            log_level=_get_str("LOG_LEVEL", "INFO").upper(),
            log_file=log_file,
            dry_run=dry_run,
        )

    def describe(self) -> str:
        """Human-readable, secret-free summary for the startup banner."""
        return (
            f"exchange={self.exchange_id} "
            f"timeframe={self.timeframe} "
            f"symbols={len(self.symbols)} ({', '.join(self.symbols)}) "
            f"sma={self.sma_period} vol_sma={self.volume_sma_period} "
            f"min_body_ratio={self.min_body_ratio:g} "
            f"vsa_expansion={self.require_volume_expansion} "
            f"candles={self.candle_limit} "
            f"stop={self.structural_lookback}-bar±{self.stop_buffer_pct:g}% "
            f"targets={'/'.join(f'1:{r:g}' for r in self.rr_targets)} "
            f"dry_run={self.dry_run}"
        )
