"""Market regime classification — the kill switch for trend-following entries.

The order-block engine assumes a trending market: an impulse leaves an
imbalance, price retraces into the block, and the trend resumes. In a range
that assumption inverts. Impulses become liquidity sweeps that immediately
revert, "changes of character" are just the other side of the range, and a
strategy built on continuation bleeds.

This module decides whether the market is trending or ranging, using only
candles that had CLOSED at the moment the decision is made. Two independent
measurements are provided:

* **ADX** (Wilder's Average Directional Index) — a smoothed measure of how
  directional recent movement has been. Low ADX means movement is spread across
  both directions with no persistent bias: a range. Its smoothing is also what
  lets it see *through* a pullback inside a trend, which matters because every
  order-block entry is, by construction, a pullback.
* **Structural containment** (SMC) — the last confirmed major swing high and
  swing low bound a range; the market is ranging while no candle has closed
  with its *whole body* beyond either bound since that range formed. Wicks
  outside the range are sweeps, not breakouts, and do not count.

Both are vectorised over the full frame, and both are causal: the value at bar
``t`` depends only on bars ``<= t``. Swing pivots in particular are only treated
as known ``strength`` bars after they print, because that is when a live
scanner could first have confirmed them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
import pandas as pd

from scanner.candles import validate_ohlcv
from scanner.smc import swing_points

logger: Final[logging.Logger] = logging.getLogger(__name__)

#: Wilder's original lookback for the directional movement system.
DEFAULT_ADX_PERIOD: Final[int] = 14

#: ADX below this is treated as the absence of a trend. Wilder's convention is
#: that readings under 20 indicate a directionless market; 25+ a strong trend.
#: 20 is chosen a priori rather than fitted, so the backtest is not tuned to it.
DEFAULT_ADX_THRESHOLD: Final[float] = 20.0

#: Bars either side of a pivot for a swing to count as *major* for regime
#: purposes. Deliberately wider than the LTF CHoCH strength: regime is about the
#: structure a range is built from, not every intraday wiggle.
DEFAULT_REGIME_SWING_STRENGTH: Final[int] = 5

#: Closed HTF candles the regime is computed over. Matches the live
#: ``CANDLE_LIMIT`` so the backtest sees exactly what the scanner sees.
DEFAULT_REGIME_LOOKBACK: Final[int] = 200


class Regime(str, Enum):
    """What the market is doing at a given moment."""

    TRENDING = "trending"
    RANGING = "ranging"
    UNKNOWN = "unknown"   # not enough closed history to judge


class RegimeMethod(str, Enum):
    """Which measurement decides the regime."""

    OFF = "off"
    ADX = "adx"
    STRUCTURE = "structure"
    CONFLUENCE = "confluence"   # ranging only when BOTH measurements agree


class RegimeGate(str, Enum):
    """Where in the pipeline a ranging reading blocks the trade."""

    ADMISSION = "admission"   # no new watched zones while ranging
    ENTRY = "entry"           # no entries while ranging; zones may wait it out
    BOTH = "both"

    @property
    def at_admission(self) -> bool:
        return self in (RegimeGate.ADMISSION, RegimeGate.BOTH)

    @property
    def at_entry(self) -> bool:
        return self in (RegimeGate.ENTRY, RegimeGate.BOTH)


def adx_frame(df: pd.DataFrame, *, period: int = DEFAULT_ADX_PERIOD) -> pd.DataFrame:
    """Wilder's directional movement system, vectorised and causal.

    Returns ``plus_di``, ``minus_di`` and ``adx`` aligned to ``df``. Smoothing is
    Wilder's recursive moving average (``ewm`` with ``alpha = 1/period`` and
    ``adjust=False``), which is causal by construction: each value is a function
    of the previous one and the current bar only.

    The first ``2 * period`` ADX values are masked as NaN. ADX is a smoothed
    average of a smoothed ratio, and before that point it still carries the
    arbitrary seed of both stages.
    """
    validate_ohlcv(df)
    if period < 2:
        raise ValueError(f"ADX period must be >= 2, got {period}.")

    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    close = df["close"].astype("float64")
    previous_close = close.shift(1)

    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)

    alpha = 1.0 / period
    atr = true_range.ewm(alpha=alpha, adjust=False).mean().replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr

    # No directional movement at all (a perfectly flat stretch) is the purest
    # form of "no trend", so a zero denominator reads as DX = 0, not as missing.
    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / di_sum).fillna(0.0)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    warmed_up = pd.Series(np.arange(len(df)) >= 2 * period, index=df.index)
    return pd.DataFrame(
        {"plus_di": plus_di, "minus_di": minus_di, "adx": adx.where(warmed_up)},
        index=df.index,
    )


def structure_frame(
    df: pd.DataFrame, *, strength: int = DEFAULT_REGIME_SWING_STRENGTH
) -> pd.DataFrame:
    """Structural range containment, vectorised and causal.

    For every bar, the range is bounded by the most recent swing high and swing
    low that had been *confirmed* by then — a pivot needs ``strength`` bars on
    each side, so one printed at bar ``i`` becomes usable at bar ``i + strength``.

    The market is ``contained`` (ranging) while no candle since that range formed
    has closed with its whole body beyond either bound. A wick through a bound
    is a sweep of the liquidity resting there, which is exactly the deviation
    this filter exists to stop the engine mistaking for a breakout — so only a
    full-body close counts as escape.

    Columns: ``swing_high``, ``swing_low``, ``valid`` (both bounds known and the
    range not inverted) and ``contained``.
    """
    validate_ohlcv(df)
    is_high, is_low = swing_points(df, strength=strength)

    # A pivot is only KNOWN `strength` bars after it prints. Shifting the pivot
    # values forward by that much is what keeps the full-frame computation
    # identical to what a bar-by-bar replay would have seen.
    known_high = df["high"].where(is_high).shift(strength)
    known_low = df["low"].where(is_low).shift(strength)
    swing_high = known_high.ffill()
    swing_low = known_low.ffill()

    # Each newly confirmed pivot redraws the range, so escape is tracked per
    # segment: a breakout of an old range says nothing about the current one.
    segment = (known_high.notna() | known_low.notna()).cumsum()
    body_top = df[["open", "close"]].max(axis=1)
    body_bottom = df[["open", "close"]].min(axis=1)
    escaped = (body_bottom > swing_high) | (body_top < swing_low)
    broken = escaped.astype("int8").groupby(segment).cummax().astype(bool)

    # An inverted "range" (last swing high below last swing low) happens in a
    # strong trend where structure steps away faster than it is confirmed. That
    # is not containment.
    valid = swing_high.notna() & swing_low.notna() & (swing_high > swing_low)
    return pd.DataFrame(
        {
            "swing_high": swing_high,
            "swing_low": swing_low,
            "valid": valid,
            "contained": valid & ~broken,
        },
        index=df.index,
    )


@dataclass(frozen=True, slots=True)
class RegimeReading:
    """The regime at one moment, with the measurements behind it."""

    regime: Regime
    adx: float | None
    contained: bool | None
    swing_high: float | None
    swing_low: float | None
    reason: str

    @property
    def is_ranging(self) -> bool:
        return self.regime is Regime.RANGING


@dataclass(frozen=True, slots=True)
class RegimeFilter:
    """Classifies the regime and decides whether it blocks a trend entry.

    ``OFF`` never blocks. An ``UNKNOWN`` reading never blocks either: the only
    way to be unknown with the default lookback is short history, and a kill
    switch that silently halts all trading on a data problem is exactly the
    kind of quiet failure this scanner is built to avoid. It is logged instead.
    """

    method: RegimeMethod = RegimeMethod.OFF
    gate: RegimeGate = RegimeGate.ENTRY
    adx_period: int = DEFAULT_ADX_PERIOD
    adx_threshold: float = DEFAULT_ADX_THRESHOLD
    swing_strength: int = DEFAULT_REGIME_SWING_STRENGTH
    lookback: int = DEFAULT_REGIME_LOOKBACK

    @property
    def enabled(self) -> bool:
        return self.method is not RegimeMethod.OFF

    def read(self, df: pd.DataFrame) -> RegimeReading:
        """Regime as of the LAST row of ``df``, which must be a closed candle."""
        if not self.enabled:
            return RegimeReading(Regime.TRENDING, None, None, None, None, "filter off")
        if df.empty:
            return RegimeReading(Regime.UNKNOWN, None, None, None, None, "no candles")

        window = df.tail(self.lookback)
        adx_value: float | None = None
        contained: bool | None = None
        swing_high: float | None = None
        swing_low: float | None = None

        if self.method in (RegimeMethod.ADX, RegimeMethod.CONFLUENCE):
            raw = adx_frame(window, period=self.adx_period)["adx"].iloc[-1]
            adx_value = float(raw) if pd.notna(raw) else None

        if self.method in (RegimeMethod.STRUCTURE, RegimeMethod.CONFLUENCE):
            row = structure_frame(window, strength=self.swing_strength).iloc[-1]
            if bool(row["valid"]):
                contained = bool(row["contained"])
                swing_high = float(row["swing_high"])
                swing_low = float(row["swing_low"])

        adx_ranging = None if adx_value is None else adx_value < self.adx_threshold

        if self.method is RegimeMethod.ADX:
            ranging = adx_ranging
        elif self.method is RegimeMethod.STRUCTURE:
            ranging = contained
        elif adx_ranging is None or contained is None:
            ranging = None
        else:
            ranging = adx_ranging and contained

        if ranging is None:
            regime = Regime.UNKNOWN
        else:
            regime = Regime.RANGING if ranging else Regime.TRENDING
        return RegimeReading(
            regime=regime,
            adx=adx_value,
            contained=contained,
            swing_high=swing_high,
            swing_low=swing_low,
            reason=self._describe(regime, adx_value, contained, swing_high, swing_low),
        )

    def read_at(self, df: pd.DataFrame, timestamp_ms: int) -> RegimeReading:
        """Regime as a live scanner would have seen it at ``timestamp_ms``.

        Only candles that had fully CLOSED by then are used. The candle still
        forming at ``timestamp_ms`` is excluded, because its direction and range
        were not yet known.
        """
        if df.empty:
            return RegimeReading(Regime.UNKNOWN, None, None, None, None, "no candles")
        opens = df["timestamp"].to_numpy(dtype="int64")
        bar_ms = int(np.median(np.diff(opens))) if len(opens) > 1 else 0
        closed = df[opens + bar_ms <= timestamp_ms]
        return self.read(closed)

    def blocks(self, reading: RegimeReading) -> bool:
        """True when this reading must stop a trend-following trade."""
        if not self.enabled:
            return False
        if reading.regime is Regime.UNKNOWN:
            logger.warning("Regime unknown (%s); not blocking.", reading.reason)
            return False
        return reading.is_ranging

    def _describe(
        self,
        regime: Regime,
        adx_value: float | None,
        contained: bool | None,
        swing_high: float | None,
        swing_low: float | None,
    ) -> str:
        parts: list[str] = []
        if adx_value is not None:
            side = "<" if adx_value < self.adx_threshold else ">="
            parts.append(f"ADX({self.adx_period}) {adx_value:.1f} {side} {self.adx_threshold:g}")
        if contained is not None and swing_high is not None and swing_low is not None:
            state = "inside" if contained else "escaped"
            parts.append(f"price {state} range [{swing_low:g}, {swing_high:g}]")
        detail = "; ".join(parts) if parts else "insufficient history"
        return f"{regime.value}: {detail}"
