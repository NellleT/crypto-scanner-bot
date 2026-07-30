"""Candlestick geometry and pattern recognition.

This module holds the pure primitives — bar geometry and the two-candle
engulfing rules — with no I/O and no trading policy. Combining a pattern with
trend and volume filters is :mod:`scanner.strategy`'s job; keeping the two apart
means the detection rules can be audited and tested on their own.

Column contract (see :mod:`scanner.exchange`)::

    timestamp (int, ms, UTC)  open  high  low  close  volume  open_time (datetime64[ns, UTC])
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Final

import pandas as pd

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

#: Default minimum real body, as a fraction of the candle's own high-low range,
#: required of BOTH candles in the pair.
#:
#: The engulfing inequalities are trivially satisfied when the previous body is
#: only a tick or two wide — "engulfing" a doji says nothing about who won the
#: bar. Measured over ~6,000 15m candle pairs across six major pairs, 5.5% of
#: raw signals had an engulfed body under 5% of its range, so this default
#: discards the degenerate cases while retaining ~94% of signals. Set to 0.0 for
#: the unfiltered mathematical definition, or raise it to demand decisive bars.
DEFAULT_MIN_BODY_RATIO: Final[float] = 0.05


class PatternType(str, Enum):
    """Supported patterns. The value is used verbatim in notifications."""

    BULLISH_ENGULFING = "Bullish Engulfing"
    BEARISH_ENGULFING = "Bearish Engulfing"

    @property
    def is_bullish(self) -> bool:
        return self is PatternType.BULLISH_ENGULFING


@dataclass(frozen=True, slots=True)
class Candle:
    """A single OHLCV bar with derived body/range geometry."""

    timestamp: int  # candle OPEN time, epoch milliseconds, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_row(cls, row: "pd.Series[float]") -> "Candle":
        return cls(
            timestamp=int(row["timestamp"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def body(self) -> float:
        """Absolute size of the real body."""
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        """High-to-low range of the bar."""
        return self.high - self.low

    @property
    def body_ratio(self) -> float:
        """Body size as a fraction of total range; 0.0 for a flat bar."""
        if self.range <= 0.0:
            return 0.0
        return self.body / self.range

    @property
    def open_time(self) -> datetime:
        return pd.Timestamp(self.timestamp, unit="ms", tz="UTC").to_pydatetime()


def _is_bullish_engulfing(previous: Candle, current: Candle) -> bool:
    """Previous bar bearish, current bar bullish, current body engulfs previous.

    The open may sit *at* the previous close (``<=``); the close must strictly
    exceed the previous open. See :func:`classify_engulfing` for why the open
    side is inclusive.
    """
    return (
        previous.is_bearish
        and current.is_bullish
        and current.open <= previous.close
        and current.close > previous.open
    )


def _is_bearish_engulfing(previous: Candle, current: Candle) -> bool:
    """Previous bar bullish, current bar bearish, current body engulfs previous.

    The open may sit *at* the previous close (``>=``); the close must strictly
    undercut the previous open. See :func:`classify_engulfing` for why the open
    side is inclusive.
    """
    return (
        previous.is_bullish
        and current.is_bearish
        and current.open >= previous.close
        and current.close < previous.open
    )


def classify_engulfing(
    previous: Candle,
    current: Candle,
    *,
    min_body_ratio: float = DEFAULT_MIN_BODY_RATIO,
) -> PatternType | None:
    """Classify a two-candle sequence, or return ``None`` if no pattern applies.

    Both candles must have a real body of at least ``min_body_ratio`` of their
    own high-low range; see :data:`DEFAULT_MIN_BODY_RATIO` for why. Pass ``0.0``
    to apply the engulfing inequalities with no quality filter at all.

    **Why the open comparison is inclusive.** The textbook form requires the
    signal candle to *gap* past the previous close (``open < prev.close`` for a
    bullish setup). That encodes an assumption from session-based markets, and
    it makes the rule depend on how a venue stitches its candles rather than on
    price action. Measured over 999 4h candles per symbol:

    ==========  ====================  =====================  ====================
    Venue       ``open == prev``      Patterns, strict ``<``  Patterns, ``<=``
    ==========  ====================  =====================  ====================
    Bybit       100%                  **0**                   216-240
    Binance     ~50%                  34-58                   151-177
    ==========  ====================  =====================  ====================

    Bybit publishes a continuous series where every candle opens exactly at the
    previous close, so ``open < prev.close`` is unsatisfiable and the strict form
    detects **nothing at all** — a silent, permanent dead end. The inclusive form
    yields comparable counts on both venues, which is what a rule about price
    action should do.

    Containment still holds with equality: if ``open == prev.close`` and
    ``close > prev.open``, the signal body spans the whole previous body. The
    close comparison stays strict, because merely matching the previous open is
    not engulfing it.
    """
    if previous.body_ratio < min_body_ratio or current.body_ratio < min_body_ratio:
        return None
    if previous.body <= 0.0 or current.body <= 0.0:
        return None
    if _is_bullish_engulfing(previous, current):
        return PatternType.BULLISH_ENGULFING
    if _is_bearish_engulfing(previous, current):
        return PatternType.BEARISH_ENGULFING
    return None


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Raise :class:`ValueError` if the frame does not meet the column contract."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {', '.join(missing)}")
