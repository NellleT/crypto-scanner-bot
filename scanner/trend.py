"""V5.0 — daily trend-following (Donchian breakout). Research, not live.

Rules, on closed **daily** candles, evaluated once per day at the close:

* **Regime** — the close is above its 200-day simple moving average. Applies to
  entries only; an open trade is managed by its exit alone.
* **Entry** — the close breaks above the highest high of the previous 20 days,
  while the regime holds. Filled at market on the next day's open.
* **Exit** — the close breaks below the lowest low of the previous 10 (or 20)
  days. Filled at market on the next day's open. No profit target: the trend is
  ridden until it bends.
* **Risk** — 1% of equity per trade, sized on the distance from entry to the
  initial trailing stop: the exit channel as it will stand on the fill day.

Both channels exclude the current candle — a close can never exceed its own
day's high, so "above the 20-day high" can only mean the 20 days before it. That
also makes every level known at the close it is judged at: no look-ahead by
construction. :func:`trend_frame` is the single implementation; the backtest
reads it over history, and :func:`latest_signal` reads its last row live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
import pandas as pd

from scanner.candles import validate_ohlcv
from scanner.risk import DEFAULT_ACCOUNT_EQUITY, DEFAULT_RISK_PER_TRADE_PCT, position_size

logger: Final[logging.Logger] = logging.getLogger(__name__)

#: Days in the regime moving average.
DEFAULT_TREND_SMA: Final[int] = 200

#: Days whose highest high a close must clear to enter.
DEFAULT_ENTRY_CHANNEL: Final[int] = 20

#: Days whose lowest low a close must break to exit. Ten is the classic pairing
#: with a 20-day breakout (Turtle System 1); 20 rides trends longer and gives
#: back more at the turn.
DEFAULT_EXIT_CHANNEL: Final[int] = 10


@dataclass(frozen=True, slots=True)
class TrendParams:
    sma: int = DEFAULT_TREND_SMA
    entry_channel: int = DEFAULT_ENTRY_CHANNEL
    exit_channel: int = DEFAULT_EXIT_CHANNEL

    def __post_init__(self) -> None:
        if min(self.sma, self.entry_channel, self.exit_channel) < 1:
            raise ValueError("SMA and channel lengths must be positive.")

    @property
    def warmup(self) -> int:
        """Candles before the first signal can exist."""
        return max(self.sma, self.entry_channel + 1, self.exit_channel + 1)


def trend_frame(df: pd.DataFrame, params: TrendParams | None = None) -> pd.DataFrame:
    """``df`` with the V5 indicators and signals attached, vectorised.

    * ``sma`` — close SMA over ``sma`` days, including today;
    * ``entry_level`` — highest high of the ``entry_channel`` days before today;
    * ``exit_level`` — lowest low of the ``exit_channel`` days before today;
    * ``next_stop`` — lowest low of the ``exit_channel`` days *ending* today:
      tomorrow's ``exit_level``, so the initial stop of a trade entered at
      tomorrow's open, known at today's close;
    * ``entry_signal`` / ``exit_signal`` — the rules above, judged at the close.

    Rows inside the warm-up have NaN levels and False signals.
    """
    params = params or TrendParams()
    validate_ohlcv(df)
    out = df.copy()
    out["sma"] = out["close"].rolling(params.sma, min_periods=params.sma).mean()
    out["entry_level"] = out["high"].shift(1).rolling(params.entry_channel).max()
    out["exit_level"] = out["low"].shift(1).rolling(params.exit_channel).min()
    out["next_stop"] = out["low"].rolling(params.exit_channel).min()
    out["regime"] = (out["close"] > out["sma"]).fillna(False)
    out["entry_signal"] = (out["regime"] & (out["close"] > out["entry_level"])).fillna(False)
    out["exit_signal"] = (out["close"] < out["exit_level"]).fillna(False)
    return out


class SignalKind(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"


@dataclass(frozen=True, slots=True)
class TrendSignal:
    """What the newest closed daily candle says to do."""

    kind: SignalKind
    candle_ms: int       # open of the daily candle whose close decided it
    close: float
    level: float         # the channel the close broke
    stop: float          # initial trailing stop, for an entry


def latest_signal(
    df: pd.DataFrame,
    params: TrendParams | None = None,
    *,
    in_position: bool,
) -> TrendSignal | None:
    """The live question, answered from the last row of :func:`trend_frame`.

    Holding: an exit signal or nothing. Flat: an entry signal or nothing — the
    two never coexist, since a close cannot be both above the 20-day high and
    below a shorter low.
    """
    params = params or TrendParams()
    if len(df) < params.warmup:
        return None
    last = trend_frame(df, params).iloc[-1]
    if in_position and bool(last["exit_signal"]):
        return TrendSignal(SignalKind.EXIT, int(last["timestamp"]), float(last["close"]),
                           float(last["exit_level"]), float("nan"))
    if not in_position and bool(last["entry_signal"]):
        return TrendSignal(SignalKind.ENTRY, int(last["timestamp"]), float(last["close"]),
                           float(last["entry_level"]), float(last["next_stop"]))
    return None


@dataclass(frozen=True, slots=True)
class TrendPlan:
    """A sized entry. There is no target: the exit channel trails the trade."""

    entry: float          # reference price — the signal close; the fill is next open
    initial_stop: float
    quantity: float
    equity: float
    risk_pct: float

    @property
    def risk_per_unit(self) -> float:
        return self.entry - self.initial_stop

    @property
    def stop_distance_pct(self) -> float:
        return self.risk_per_unit / self.entry * 100.0

    @property
    def notional(self) -> float:
        return self.quantity * self.entry

    @property
    def risk_amount(self) -> float:
        return self.equity * self.risk_pct / 100.0


def plan_trend_entry(
    signal: TrendSignal,
    *,
    equity: float = DEFAULT_ACCOUNT_EQUITY,
    risk_pct: float = DEFAULT_RISK_PER_TRADE_PCT,
) -> TrendPlan | None:
    """Size an entry so the initial stop costs ``risk_pct`` of equity.

    The exit is a *close* below the channel, filled at the next open, so a gap
    can lose more than the planned 1% — the backtest measures how often and by
    how much, rather than pretending the stop is a hard price.
    """
    if signal.kind is not SignalKind.ENTRY:
        return None
    risk = signal.close - signal.stop
    if not (risk > 0 and np.isfinite(risk)):
        return None
    quantity = position_size(equity=equity, risk_pct=risk_pct, risk_per_unit=risk)
    if quantity <= 0:
        return None
    return TrendPlan(signal.close, signal.stop, quantity, equity, risk_pct)


__all__ = [
    "DEFAULT_ENTRY_CHANNEL",
    "DEFAULT_EXIT_CHANNEL",
    "DEFAULT_TREND_SMA",
    "SignalKind",
    "TrendParams",
    "TrendPlan",
    "TrendSignal",
    "latest_signal",
    "plan_trend_entry",
    "trend_frame",
]
