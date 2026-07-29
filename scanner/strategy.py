"""Multi-factor signal generation.

Composes three independent conditions into a single tradable signal:

===========  ==========================================  ==========================================
Filter       LONG                                        SHORT
===========  ==========================================  ==========================================
Trend        ``close > SMA_200``                         ``close < SMA_200``
Volume       ``volume > VOL_SMA_20``                     ``volume > VOL_SMA_20``
VSA          ``volume > previous volume``                ``volume > previous volume``
Pattern      Bullish Engulfing                           Bearish Engulfing
===========  ==========================================  ==========================================

All comparisons are strict. Both volume conditions are deliberately identical
for both directions: conviction is shown by participation, regardless of which
side is winning.

The two volume filters test different things and neither implies the other. The
first asks whether participation is high against the recent norm; the second
whether it *expanded* over the very bar being engulfed. A reversal that prints
less volume than the candle it swallowed reversed on fading participation, which
is the signature of a move drifting rather than being driven — it can clear a
20-period average comfortably and still fail this test.

A confirmed setup is then sized by :mod:`scanner.risk`, which anchors the stop
to the structural extreme of the recent range rather than to the signal candle.
A setup whose stop and targets cannot be expressed as real prices is rejected at
that final step rather than alerted without them.

Every evaluation returns a :class:`StrategyResult` carrying the reason it did or
did not fire, which makes the filter chain observable at DEBUG level instead of
signals silently disappearing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Final

import pandas as pd

from scanner.indicators import (
    DEFAULT_SMA_PERIOD,
    DEFAULT_VOLUME_SMA_PERIOD,
    required_candles,
    trend_column,
    volume_column,
)
from scanner.patterns import (
    DEFAULT_MIN_BODY_RATIO,
    Candle,
    PatternType,
    classify_engulfing,
    validate_ohlcv,
)
from scanner.risk import (
    DEFAULT_RR_TARGETS,
    DEFAULT_STOP_BUFFER_PCT,
    DEFAULT_STRUCTURAL_LOOKBACK,
    RiskPlan,
    build_risk_plan,
)

logger: Final[logging.Logger] = logging.getLogger(__name__)


class FilterStage(str, Enum):
    """Where an evaluation ended.

    Typed rather than inferred from the ``reason`` text: several rejection
    messages read "... is not above ...", so substring matching cannot tell them
    apart. Callers aggregating a funnel must compare this field.

    **Members are declared in filter-chain order** and depth is derived from that
    order, so inserting a stage does not require touching the ``reached_*``
    helpers. They previously enumerated members by hand, which meant adding a
    stage silently mis-attributed part of the funnel until every list was
    updated — the same failure mode the enum was introduced to prevent.
    """

    WARMUP = "warmup"          # indicators not yet available
    PATTERN = "pattern"        # no engulfing pattern
    TREND = "trend"            # pattern present, wrong side of the SMA
    VOLUME = "volume"          # volume below its own moving average
    VSA = "vsa"                # volume above average but not expanding bar-on-bar
    RISK = "risk"              # filters passed but no tradable stop/target set
    CONFIRMED = "confirmed"    # all filters passed

    @property
    def depth(self) -> int:
        """Position in the filter chain; a later stage means more filters passed."""
        return _STAGE_ORDER.index(self)

    def passed(self, stage: "FilterStage") -> bool:
        """True when evaluation got *past* ``stage`` — that filter accepted it.

        A recorded stage is where evaluation stopped, so stopping at ``TREND``
        means ``PATTERN`` was passed but ``TREND`` was not.
        """
        return self.depth > stage.depth

    @property
    def reached_pattern(self) -> bool:
        """True when an engulfing pattern was present, whatever happened next."""
        return self.passed(FilterStage.PATTERN)

    @property
    def reached_trend(self) -> bool:
        """True when the pattern also agreed with the trend regime."""
        return self.passed(FilterStage.TREND)

    @property
    def reached_volume(self) -> bool:
        """True when volume also cleared its moving average."""
        return self.passed(FilterStage.VOLUME)

    @property
    def reached_vsa(self) -> bool:
        """True when every entry filter passed, whatever risk sizing then said."""
        return self.passed(FilterStage.VSA)


#: Filter-chain order, taken from the declaration order above.
_STAGE_ORDER: Final[tuple[FilterStage, ...]] = tuple(FilterStage)


class SignalDirection(str, Enum):
    """Direction of a confirmed setup."""

    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def emoji(self) -> str:
        return "🟢" if self is SignalDirection.LONG else "🔴"

    @classmethod
    def for_pattern(cls, pattern: PatternType) -> "SignalDirection":
        return cls.LONG if pattern.is_bullish else cls.SHORT


@dataclass(frozen=True, slots=True)
class TradeSignal:
    """A setup that passed every filter, ready to be formatted and dispatched."""

    symbol: str
    timeframe: str
    direction: SignalDirection
    pattern: PatternType
    price: float
    trend_sma: float
    trend_sma_period: int
    volume: float
    volume_sma: float
    volume_sma_period: int
    previous_volume: float
    engulf_ratio: float
    candle: Candle
    risk: RiskPlan

    @property
    def volume_ratio(self) -> float:
        """Signal-candle volume as a multiple of its moving average."""
        if self.volume_sma <= 0.0:
            return 0.0
        return self.volume / self.volume_sma

    @property
    def volume_expansion_ratio(self) -> float:
        """Signal-candle volume as a multiple of the engulfed candle's.

        Always > 1.0 on a confirmed signal when the VSA rule is active; carried
        on the signal so the alert can show the follow-through it was accepted
        for rather than asserting it.
        """
        if self.previous_volume <= 0.0:
            return 0.0
        return self.volume / self.previous_volume

    @property
    def sma_distance_pct(self) -> float:
        """Signed distance from the trend average, in percent."""
        if self.trend_sma <= 0.0:
            return 0.0
        return (self.price - self.trend_sma) / self.trend_sma * 100.0

    @property
    def candle_open_time(self) -> datetime:
        return self.candle.open_time

    @property
    def dedup_key(self) -> tuple[str, str, int]:
        return (self.symbol, self.timeframe, self.candle.timestamp)


@dataclass(frozen=True, slots=True)
class StrategyResult:
    """Outcome of evaluating one symbol: a signal, or why there wasn't one."""

    signal: TradeSignal | None
    reason: str
    stage: FilterStage

    @property
    def matched(self) -> bool:
        return self.signal is not None

    @classmethod
    def rejected(cls, stage: FilterStage, reason: str) -> "StrategyResult":
        return cls(signal=None, reason=reason, stage=stage)


@dataclass(frozen=True, slots=True)
class EngulfingTrendStrategy:
    """Engulfing reversal, filtered by trend regime and volume conviction."""

    sma_period: int = DEFAULT_SMA_PERIOD
    volume_sma_period: int = DEFAULT_VOLUME_SMA_PERIOD
    min_body_ratio: float = DEFAULT_MIN_BODY_RATIO
    require_volume_expansion: bool = True
    structural_lookback: int = DEFAULT_STRUCTURAL_LOOKBACK
    stop_buffer_pct: float = DEFAULT_STOP_BUFFER_PCT
    rr_targets: tuple[float, ...] = DEFAULT_RR_TARGETS

    @property
    def required_candles(self) -> int:
        """Closed candles needed before the newest bar can be evaluated."""
        return max(
            required_candles(self.sma_period, self.volume_sma_period),
            self.structural_lookback,
        )

    def evaluate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> StrategyResult:
        """Evaluate the most recently CLOSED candle against every filter.

        ``df`` must contain only closed candles and must already carry the
        indicator columns (see :func:`scanner.indicators.add_indicators`).
        """
        validate_ohlcv(df)

        if len(df) < 2:
            return StrategyResult.rejected(
                FilterStage.WARMUP, "not enough candles for a two-bar pattern"
            )

        trend_col = trend_column(self.sma_period)
        volume_col = volume_column(self.volume_sma_period)
        for column in (trend_col, volume_col):
            if column not in df.columns:
                raise ValueError(
                    f"Indicator column {column!r} is missing — call add_indicators() first."
                )

        signal_row = df.iloc[-1]
        previous = Candle.from_row(df.iloc[-2])
        current = Candle.from_row(signal_row)

        trend_sma = float(signal_row[trend_col])
        volume_sma = float(signal_row[volume_col])

        # A partially warmed-up average would compare against a meaningless
        # number, so treat it as "no opinion" rather than passing or failing.
        if math.isnan(trend_sma) or math.isnan(volume_sma):
            return StrategyResult.rejected(
                FilterStage.WARMUP,
                f"indicators not warmed up (need {self.required_candles} closed candles, "
                f"have {len(df)})",
            )

        # 1. Pattern — cheapest discriminator and the rarest condition.
        pattern = classify_engulfing(previous, current, min_body_ratio=self.min_body_ratio)
        if pattern is None:
            return StrategyResult.rejected(FilterStage.PATTERN, "no engulfing pattern")

        direction = SignalDirection.for_pattern(pattern)

        # 2. Trend regime — only take reversals in the direction of the trend.
        if direction is SignalDirection.LONG and not current.close > trend_sma:
            return StrategyResult.rejected(
                FilterStage.TREND,
                f"{pattern.value} rejected: close {current.close:g} is not above "
                f"{trend_col} {trend_sma:g}",
            )
        if direction is SignalDirection.SHORT and not current.close < trend_sma:
            return StrategyResult.rejected(
                FilterStage.TREND,
                f"{pattern.value} rejected: close {current.close:g} is not below "
                f"{trend_col} {trend_sma:g}",
            )

        # 3. Volume conviction — participation against the recent norm.
        if not current.volume > volume_sma:
            return StrategyResult.rejected(
                FilterStage.VOLUME,
                f"{pattern.value} rejected: volume {current.volume:g} is not above "
                f"{volume_col} {volume_sma:g}",
            )

        # 4. VSA follow-through — expansion against the bar being engulfed.
        #    A candle can sit above a 20-period average and still print *less*
        #    volume than the bar it engulfed. Price reversed on fading
        #    participation, which is the signature of a move without size behind
        #    it rather than one being driven.
        if self.require_volume_expansion and not current.volume > previous.volume:
            return StrategyResult.rejected(
                FilterStage.VSA,
                f"{pattern.value} rejected: volume {current.volume:g} did not expand "
                f"over the engulfed candle's {previous.volume:g} "
                f"({current.volume / previous.volume:.2f}x) — no follow-through"
                if previous.volume > 0.0
                else f"{pattern.value} rejected: engulfed candle has no volume",
            )

        # 5. Risk structure — the entry filters have all passed, so size the
        #    trade. A setup with no expressible stop/target is not actionable.
        plan = build_risk_plan(
            df,
            is_long=direction is SignalDirection.LONG,
            entry=current.close,
            lookback=self.structural_lookback,
            buffer_pct=self.stop_buffer_pct,
            rr_targets=self.rr_targets,
        )
        if plan is None:
            return StrategyResult.rejected(
                FilterStage.RISK,
                f"{direction.value} rejected: no tradable stop/target structure "
                f"over the last {self.structural_lookback} bars",
            )

        engulf_ratio = current.body / previous.body if previous.body > 0.0 else 0.0

        return StrategyResult(
            signal=TradeSignal(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                pattern=pattern,
                price=current.close,
                trend_sma=trend_sma,
                trend_sma_period=self.sma_period,
                volume=current.volume,
                volume_sma=volume_sma,
                volume_sma_period=self.volume_sma_period,
                previous_volume=previous.volume,
                engulf_ratio=engulf_ratio,
                candle=current,
                risk=plan,
            ),
            reason=f"{direction.value} confirmed by trend, volume and pattern",
            stage=FilterStage.CONFIRMED,
        )
