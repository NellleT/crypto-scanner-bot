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

from scanner.candles import DEFAULT_MIN_BODY_RATIO
from scanner.execution import to_unified_symbol
from scanner.risk import (
    DEFAULT_ACCOUNT_EQUITY,
    DEFAULT_MAX_STOP_PCT,
    DEFAULT_REWARD_RATIO,
    DEFAULT_RISK_PER_TRADE_PCT,
    DEFAULT_STOP_BUFFER_PCT,
)
from scanner.mtf import (
    DEFAULT_CONFIRM_WINDOW,
    DEFAULT_LTF_MIN_FVG_PCT,
)
from scanner.smc import (
    DEFAULT_MIN_FVG_PCT,
    DEFAULT_RANGE_LOOKBACK,
    DEFAULT_SWING_STRENGTH,
    STRUCTURE_LENGTH,
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Default venue — the one orders are executed on.
#:
#: v3.1 builds orders at exact order-block edges, so the levels
#: must come from the book they will rest in. Kraken and Binance disagree by up
#: to 1.3% on the thinner pairs; against a stop that is often ~1% wide, a level
#: taken from the wrong venue either fills instantly or never fills.
#:
#: Binance restricts the IP ranges GitHub Actions runners use, so a scheduled
#: workflow cannot reach it — v3.1 needs a host that Binance serves. Set
#: EXCHANGE_ID=kraken to run from CI, accepting that the levels are indicative
#: rather than executable.
DEFAULT_EXCHANGE_ID: Final[str] = "binance"

#: Default execution watchlist.
DEFAULT_SYMBOLS: Final[tuple[str, ...]] = (
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "AVAX/USDT",
)

#: Higher timeframe — where structure is read and zones are defined.
DEFAULT_TIMEFRAME: Final[str] = "1h"

#: Lower timeframe — where entries are confirmed. Must be strictly faster than
#: TIMEFRAME, or confirmation would be reading the same bars as the setup.
DEFAULT_LTF_TIMEFRAME: Final[str] = "15m"

#: HTF candles per request. The structure needs three, but premium/discount
#: needs the full dealing-range lookback behind it.
DEFAULT_CANDLE_LIMIT: Final[int] = 200

#: LTF candles per request — only the recent confirmation window is examined.
DEFAULT_LTF_CANDLE_LIMIT: Final[int] = 120

#: Hours a watched zone stays live before it is retired as stale.
DEFAULT_MAX_ZONE_AGE_HOURS: Final[float] = 72.0

#: Concurrent market-data fetches. CCXT is blocking I/O, so threads overlap the
#: waiting; the venue rate limiter still serialises what actually goes out.
DEFAULT_MAX_WORKERS: Final[int] = 4

#: Where the watched-zone state machine is persisted between runs.
DEFAULT_WATCHLIST_FILE: Final[str] = "watchlist.json"

#: Upper bound accepted for CANDLE_LIMIT. Individual venues cap lower — Kraken
#: returns at most ~720 candles — and :mod:`scanner.exchange` warns when a
#: response comes back materially short of what was requested.
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


#: Seconds per timeframe unit, for ordering HTF against LTF without CCXT.
_UNIT_SECONDS: Final[dict[str, int]] = {
    "m": 60,
    "h": 3_600,
    "d": 86_400,
    "w": 604_800,
    "M": 2_592_000,
}


def _timeframe_seconds(timeframe: str) -> int:
    """Duration of one candle in seconds. Assumes a validated token."""
    return int(timeframe[:-1]) * _UNIT_SECONDS[timeframe[-1]]


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
    """Parse a comma-separated symbol list into CCXT unified form.

    Accepts both the exchange-native spelling used by the Binance API
    (``BTCUSDT``) and the unified form CCXT needs (``BTC/USDT``), because the
    execution watchlist is naturally written the first way and the market-data
    layer requires the second. Order is preserved and duplicates collapse — so
    ``BTCUSDT`` and ``BTC/USDT`` in one list resolve to a single entry.
    """
    seen: dict[str, None] = {}
    for chunk in raw.split(","):
        token = chunk.strip().upper()
        if not token:
            continue
        try:
            seen.setdefault(to_unified_symbol(token), None)
        except ValueError as exc:
            raise ConfigError(f"Cannot parse symbol {token!r}: {exc}") from exc
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
    ltf_timeframe: str
    ltf_candle_limit: int
    min_body_ratio: float
    min_fvg_pct: float
    range_lookback: int
    require_extreme: bool
    swing_strength: int
    confirm_window: int
    ltf_min_fvg_pct: float
    max_zone_age_hours: float
    max_workers: int
    watchlist_file: Path
    stop_buffer_pct: float
    max_stop_pct: float
    reward_ratio: float
    account_equity: float
    risk_per_trade_pct: float
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
        ltf_timeframe = validate_timeframe_token(
            _get_str("LTF_TIMEFRAME", DEFAULT_LTF_TIMEFRAME)
        )
        if _timeframe_seconds(ltf_timeframe) >= _timeframe_seconds(timeframe):
            raise ConfigError(
                f"LTF_TIMEFRAME={ltf_timeframe} must be strictly faster than "
                f"TIMEFRAME={timeframe}. Confirming an entry on the same or a "
                "slower timeframe than the setup would just re-read the setup."
            )

        range_lookback = _get_int(
            "RANGE_LOOKBACK", DEFAULT_RANGE_LOOKBACK, minimum=4, maximum=1000
        )

        watchlist_raw = _get_str("WATCHLIST_FILE", DEFAULT_WATCHLIST_FILE)
        watchlist_file = Path(watchlist_raw).expanduser()
        if not watchlist_file.is_absolute():
            watchlist_file = PROJECT_ROOT / watchlist_file

        symbols_raw = _get_str("SYMBOLS", ",".join(DEFAULT_SYMBOLS))

        candle_limit = _get_int(
            "CANDLE_LIMIT",
            DEFAULT_CANDLE_LIMIT,
            minimum=STRUCTURE_LENGTH + 1,
            maximum=MAX_CANDLE_LIMIT,
        )

        # The newest bar is usually still forming and gets dropped, so one extra
        # candle is needed on top of the dealing-range lookback. Without the full
        # window, premium/discount is measured against a partial range and every
        # block looks like an extreme.
        minimum_limit = range_lookback + 1
        if candle_limit < minimum_limit:
            raise ConfigError(
                f"CANDLE_LIMIT={candle_limit} is too small for "
                f"RANGE_LOOKBACK={range_lookback}: at least {minimum_limit} candles "
                "are needed, otherwise the dealing range is measured against a "
                "partial window and premium/discount filtering is meaningless. "
                "Raise CANDLE_LIMIT or lower RANGE_LOOKBACK."
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
            ltf_timeframe=ltf_timeframe,
            ltf_candle_limit=_get_int(
                "LTF_CANDLE_LIMIT",
                DEFAULT_LTF_CANDLE_LIMIT,
                minimum=STRUCTURE_LENGTH + 1,
                maximum=MAX_CANDLE_LIMIT,
            ),
            min_fvg_pct=_get_float(
                "MIN_FVG_PCT", DEFAULT_MIN_FVG_PCT, minimum=0.0, maximum=100.0
            ),
            range_lookback=range_lookback,
            require_extreme=_get_bool("REQUIRE_EXTREME_OB", True),
            swing_strength=_get_int(
                "SWING_STRENGTH", DEFAULT_SWING_STRENGTH, minimum=1, maximum=50
            ),
            confirm_window=_get_int(
                "LTF_CONFIRM_WINDOW", DEFAULT_CONFIRM_WINDOW, minimum=STRUCTURE_LENGTH, maximum=500
            ),
            ltf_min_fvg_pct=_get_float(
                "LTF_MIN_FVG_PCT", DEFAULT_LTF_MIN_FVG_PCT, minimum=0.0, maximum=100.0
            ),
            max_zone_age_hours=_get_float(
                "MAX_ZONE_AGE_HOURS", DEFAULT_MAX_ZONE_AGE_HOURS, minimum=0.0
            ),
            max_workers=_get_int("MAX_WORKERS", DEFAULT_MAX_WORKERS, minimum=1, maximum=32),
            watchlist_file=watchlist_file,
            min_body_ratio=_get_float(
                "MIN_BODY_RATIO",
                DEFAULT_MIN_BODY_RATIO,
                minimum=0.0,
                maximum=1.0,
            ),
            stop_buffer_pct=_get_float(
                "STOP_BUFFER_PCT",
                DEFAULT_STOP_BUFFER_PCT,
                minimum=0.0,
                maximum=10.0,
            ),
            max_stop_pct=_get_float(
                "MAX_STOP_PCT", DEFAULT_MAX_STOP_PCT, minimum=0.0, maximum=100.0
            ),
            reward_ratio=_get_float(
                "REWARD_RATIO", DEFAULT_REWARD_RATIO, minimum=0.1, maximum=100.0
            ),
            account_equity=_get_float(
                "ACCOUNT_EQUITY", DEFAULT_ACCOUNT_EQUITY, minimum=0.01
            ),
            risk_per_trade_pct=_get_float(
                "RISK_PER_TRADE_PCT",
                DEFAULT_RISK_PER_TRADE_PCT,
                minimum=0.001,
                maximum=100.0,
            ),
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
            f"ltf={self.ltf_timeframe} "
            f"min_fvg={self.min_fvg_pct:g}% "
            f"range={self.range_lookback} extreme_only={self.require_extreme} "
            f"min_body_ratio={self.min_body_ratio:g} "
            f"candles={self.candle_limit}/{self.ltf_candle_limit} "
            f"stop=distal±{self.stop_buffer_pct:g}% max_stop={self.max_stop_pct:g}% "
            f"target=1:{self.reward_ratio:g} "
            f"risk={self.risk_per_trade_pct:g}% of {self.account_equity:,.2f} "
            f"dry_run={self.dry_run}"
        )
