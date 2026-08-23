"""Crypto Scanner Bot — Smart Money Concepts scanner for crypto spot markets.

v3.0 retires the v2.x engulfing/SMA/volume chain. A signal now requires an
order block — the last opposing candle before a displacement — validated by the
fair value gap that displacement left behind. Each validated block produces a
resting limit order at its proximal edge, a stop beyond its distal edge, a fixed
R-multiple target, and a size derived from a fixed fraction of account equity.
"""

from __future__ import annotations

from scanner.candles import DEFAULT_MIN_BODY_RATIO, Candle, validate_ohlcv
from scanner.config import Settings
from scanner.execution import (
    ExecutionOrder,
    build_execution_order,
    to_binance_symbol,
    to_unified_symbol,
)
from scanner.risk import (
    DEFAULT_ACCOUNT_EQUITY,
    DEFAULT_REWARD_RATIO,
    DEFAULT_RISK_PER_TRADE_PCT,
    DEFAULT_STOP_BUFFER_PCT,
    TradePlan,
    build_trade_plan,
    position_size,
)
from scanner.smc import (
    STRUCTURE_LENGTH,
    Direction,
    FairValueGap,
    OrderBlock,
    detect_order_block,
    fvg_frame,
    order_block_mask,
)
from scanner.strategy import (
    FilterStage,
    OrderBlockStrategy,
    StrategyResult,
    TradeSignal,
)

__all__ = [
    "DEFAULT_ACCOUNT_EQUITY",
    "DEFAULT_MIN_BODY_RATIO",
    "DEFAULT_REWARD_RATIO",
    "DEFAULT_RISK_PER_TRADE_PCT",
    "DEFAULT_STOP_BUFFER_PCT",
    "STRUCTURE_LENGTH",
    "Candle",
    "Direction",
    "ExecutionOrder",
    "FairValueGap",
    "FilterStage",
    "OrderBlock",
    "OrderBlockStrategy",
    "Settings",
    "StrategyResult",
    "TradePlan",
    "TradeSignal",
    "build_execution_order",
    "build_trade_plan",
    "detect_order_block",
    "fvg_frame",
    "order_block_mask",
    "position_size",
    "to_binance_symbol",
    "to_unified_symbol",
    "validate_ohlcv",
]

__version__ = "3.0.0"
