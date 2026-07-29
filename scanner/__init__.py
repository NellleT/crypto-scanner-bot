"""Crypto Scanner Bot — multi-factor engulfing scanner for Binance spot markets.

Signals require an engulfing reversal to agree with the SMA-200 trend regime and
be backed by volume that is both above its own average and expanding over the
candle being engulfed. Each confirmed signal carries a risk plan
whose stop anchors to the recent structural extreme rather than to the signal
candle, to sit beyond the obvious pool of resting stops.
"""

from __future__ import annotations

from scanner.config import Settings
from scanner.indicators import (
    DEFAULT_SMA_PERIOD,
    DEFAULT_VOLUME_SMA_PERIOD,
    add_indicators,
    required_candles,
)
from scanner.patterns import (
    DEFAULT_MIN_BODY_RATIO,
    Candle,
    PatternType,
    classify_engulfing,
)
from scanner.risk import (
    DEFAULT_RR_TARGETS,
    DEFAULT_STOP_BUFFER_PCT,
    DEFAULT_STRUCTURAL_LOOKBACK,
    RiskPlan,
    TakeProfit,
    build_risk_plan,
)
from scanner.strategy import (
    EngulfingTrendStrategy,
    FilterStage,
    SignalDirection,
    StrategyResult,
    TradeSignal,
)

__all__ = [
    "DEFAULT_MIN_BODY_RATIO",
    "DEFAULT_RR_TARGETS",
    "DEFAULT_SMA_PERIOD",
    "DEFAULT_STOP_BUFFER_PCT",
    "DEFAULT_STRUCTURAL_LOOKBACK",
    "DEFAULT_VOLUME_SMA_PERIOD",
    "Candle",
    "EngulfingTrendStrategy",
    "FilterStage",
    "PatternType",
    "RiskPlan",
    "Settings",
    "SignalDirection",
    "StrategyResult",
    "TakeProfit",
    "TradeSignal",
    "add_indicators",
    "build_risk_plan",
    "classify_engulfing",
    "required_candles",
]

__version__ = "2.2.0"
