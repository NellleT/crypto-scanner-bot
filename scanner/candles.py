"""Candle geometry primitives.

Pure value objects over an OHLCV frame — no I/O, no trading policy. Replaces
the v2.x ``patterns`` module, which also held the engulfing rules that v3.0
retires in favour of :mod:`scanner.smc`.

Column contract (see :mod:`scanner.exchange`)::

    timestamp (int, ms, UTC)  open  high  low  close  volume  open_time
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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

#: Default minimum real body as a fraction of the candle's own high-low range.
#: A structure built on a doji carries no information about who was in control,
#: and the order-block rules would otherwise fire on noise.
DEFAULT_MIN_BODY_RATIO: Final[float] = 0.05


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


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Raise :class:`ValueError` if the frame does not meet the column contract."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {', '.join(missing)}")
