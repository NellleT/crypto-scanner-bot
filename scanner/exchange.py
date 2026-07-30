"""Market data access via CCXT.

Wraps a CCXT exchange instance with the concerns a long-running scanner needs:
retry/backoff on transient failures, normalisation into a pandas ``DataFrame``,
and — critically — removal of the still-forming candle so that pattern logic
never evaluates an incomplete bar.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Final, TypeVar

import ccxt
import pandas as pd

from scanner.patterns import REQUIRED_COLUMNS

logger: Final[logging.Logger] = logging.getLogger(__name__)

T = TypeVar("T")

#: Transient failures worth retrying. Ordered most-specific first for clarity;
#: note RateLimitExceeded is itself a subclass of ExchangeError in CCXT.
_RETRYABLE: Final[tuple[type[Exception], ...]] = (
    ccxt.RateLimitExceeded,
    ccxt.DDoSProtection,
    ccxt.RequestTimeout,
    ccxt.ExchangeNotAvailable,
    ccxt.NetworkError,
)


class MarketDataError(RuntimeError):
    """Raised when market data could not be retrieved after all retries."""


class MarketDataClient:
    """Thin, retrying façade over a CCXT exchange for public OHLCV data.

    No API keys are required or accepted: this client only touches public
    market-data endpoints.
    """

    def __init__(
        self,
        exchange_id: str = "kraken",
        *,
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._exchange_id = exchange_id
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._stop_event = stop_event or threading.Event()
        self._markets_loaded = False
        self._short_response_warned: set[str] = set()

        try:
            exchange_class: type[ccxt.Exchange] = getattr(ccxt, exchange_id)
        except AttributeError as exc:
            raise MarketDataError(f"Unknown CCXT exchange id {exchange_id!r}.") from exc

        self._exchange: ccxt.Exchange = exchange_class(
            {
                "enableRateLimit": True,  # let CCXT throttle to the venue's limits
                "timeout": int(timeout_seconds * 1000),
                "options": {"defaultType": "spot"},
            }
        )

    @property
    def exchange_id(self) -> str:
        return self._exchange_id

    # ------------------------------------------------------------------
    # Retry plumbing
    # ------------------------------------------------------------------
    def _with_retries(self, description: str, operation: Callable[[], T]) -> T:
        """Run ``operation``, retrying transient CCXT failures with backoff."""
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 2):
            if self._stop_event.is_set():
                raise MarketDataError(f"{description} aborted: shutdown requested.")
            try:
                return operation()
            except _RETRYABLE as exc:
                last_error = exc
                if attempt > self._max_retries:
                    break
                delay = self._retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "%s failed (attempt %d/%d): %s: %s — retrying in %.1fs",
                    description,
                    attempt,
                    self._max_retries + 1,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                # Interruptible sleep so Ctrl-C is honoured during backoff.
                if self._stop_event.wait(delay):
                    raise MarketDataError(
                        f"{description} aborted: shutdown requested."
                    ) from exc
            except ccxt.BadSymbol as exc:
                # Permanent for this symbol — do not burn retries on it.
                raise MarketDataError(f"{description} failed: {exc}") from exc
            except ccxt.ExchangeError as exc:
                last_error = exc
                logger.error("%s failed with exchange error: %s", description, exc)
                break

        raise MarketDataError(
            f"{description} failed after {self._max_retries + 1} attempt(s): "
            f"{type(last_error).__name__ if last_error else 'unknown error'}: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_markets(self) -> None:
        """Load market metadata (idempotent). Enables symbol + precision support."""
        if self._markets_loaded:
            return
        self._with_retries("load_markets", lambda: self._exchange.load_markets())
        self._markets_loaded = True
        logger.info(
            "Loaded %d markets from %s.",
            len(self._exchange.markets or {}),
            self._exchange_id,
        )

    def timeframe_seconds(self, timeframe: str) -> int:
        """Duration of one candle in seconds, per the exchange's own parser."""
        try:
            return int(self._exchange.parse_timeframe(timeframe))
        except Exception as exc:  # ccxt raises bare NotSupported/ValueError here
            raise MarketDataError(
                f"Timeframe {timeframe!r} is not parseable: {exc}"
            ) from exc

    def validate_timeframe(self, timeframe: str) -> None:
        """Warn (do not fail) if the venue does not advertise ``timeframe``."""
        supported: dict[str, Any] | None = getattr(self._exchange, "timeframes", None)
        if supported and timeframe not in supported:
            logger.warning(
                "Timeframe %r is not advertised by %s. Supported: %s",
                timeframe,
                self._exchange_id,
                ", ".join(sorted(supported)),
            )

    def filter_supported_symbols(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        """Drop symbols the exchange does not list, logging each omission."""
        self.load_markets()
        markets = self._exchange.markets or {}
        if not markets:
            return symbols

        supported: list[str] = []
        for symbol in symbols:
            market = markets.get(symbol)
            if market is None:
                logger.error(
                    "Symbol %s is not listed on %s — skipping.",
                    symbol,
                    self._exchange_id,
                )
                continue
            if market.get("active") is False:
                logger.warning("Symbol %s is inactive on %s — skipping.", symbol, self._exchange_id)
                continue
            supported.append(symbol)
        return tuple(supported)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        *,
        drop_unclosed: bool = True,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles as a typed ``DataFrame`` sorted oldest → newest.

        When ``drop_unclosed`` is true (the default) the final bar is discarded
        if its close time has not yet passed. Kraken, Bybit and Binance all
        return the in-progress candle as the last element, and evaluating it
        would produce signals that vanish before the bar actually closes.
        """
        raw: list[list[float]] = self._with_retries(
            f"fetch_ohlcv({symbol}, {timeframe})",
            lambda: self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit),
        )

        if not raw:
            logger.warning("No OHLCV data returned for %s %s.", symbol, timeframe)
            return pd.DataFrame(columns=list(REQUIRED_COLUMNS))

        # Venues cap OHLCV history at different depths — Kraken tops out around
        # 720 candles regardless of what is asked for. Silently receiving fewer
        # bars than requested starves the moving averages, and the only symptom
        # would be a scanner that warms up forever and never alerts. Say so once.
        if len(raw) < limit * 0.95 and symbol not in self._short_response_warned:
            self._short_response_warned.add(symbol)
            logger.warning(
                "%s returned %d of the %d candles requested for %s %s. This is a "
                "venue history cap; make sure it still covers your indicator "
                "periods, or the scanner will never warm up.",
                self._exchange_id,
                len(raw),
                limit,
                symbol,
                timeframe,
            )

        df = pd.DataFrame(raw, columns=list(REQUIRED_COLUMNS))
        df = df.astype(
            {
                "timestamp": "int64",
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
            }
        )
        df = df.drop_duplicates(subset="timestamp", keep="last")
        df = df.sort_values("timestamp").reset_index(drop=True)

        if drop_unclosed:
            df = self._drop_unclosed_candle(df, timeframe)

        df["open_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def price_to_precision(self, symbol: str, price: float) -> str | None:
        """Format ``price`` using the venue's tick size, or ``None`` if unknown."""
        if not self._markets_loaded:
            return None
        try:
            return str(self._exchange.price_to_precision(symbol, price))
        except Exception:  # unknown symbol or missing precision metadata
            return None

    def close(self) -> None:
        """Release any underlying HTTP session held by CCXT."""
        closer = getattr(self._exchange, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:  # never let teardown mask a real error
                logger.debug("Ignoring error while closing exchange: %s", exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _drop_unclosed_candle(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if df.empty:
            return df
        duration_ms = self.timeframe_seconds(timeframe) * 1000
        now_ms = int(self._exchange.milliseconds())
        last_close_ms = int(df["timestamp"].iloc[-1]) + duration_ms
        if last_close_ms > now_ms:
            return df.iloc[:-1].reset_index(drop=True)
        return df
