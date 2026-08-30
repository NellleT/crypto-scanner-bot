"""Crypto Scanner Bot — institutional multi-timeframe SMC engine.

v3.1 stops entering blind. A higher-timeframe order block must show real
displacement (a fair value gap of at least ``MIN_FVG_PCT``) and sit at an
extreme of the dealing range (discount for longs, premium for shorts). Even
then it is only *watched*: an order is built when price returns to the zone and
the lower timeframe prints a change of character with its own gap. Zones expire
if the target is reached first, if structure breaks, or with age.
"""

from __future__ import annotations

from scanner.analytics import SimulationReport, simulate
from scanner.candles import DEFAULT_MIN_BODY_RATIO, Candle, validate_ohlcv
from scanner.config import Settings
from scanner.execution import (
    ExecutionOrder,
    build_execution_order,
    to_binance_symbol,
    to_unified_symbol,
)
from scanner.mtf import (
    DEFAULT_CONFIRM_WINDOW,
    DEFAULT_LTF_MIN_FVG_PCT,
    ConfirmationRejection,
    LtfTrigger,
    confirm_entry,
    find_choch,
    find_fvg_after,
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
    DEFAULT_MIN_FVG_PCT,
    DEFAULT_RANGE_LOOKBACK,
    DEFAULT_SWING_STRENGTH,
    STRUCTURE_LENGTH,
    Direction,
    FairValueGap,
    OrderBlock,
    RangeZone,
    SwingRange,
    detect_choch,
    detect_order_block,
    fvg_frame,
    order_block_mask,
    premium_discount_frame,
    swing_points,
    swing_range,
)
from scanner.strategy import (
    FilterStage,
    OrderBlockStrategy,
    StrategyResult,
    TradeSignal,
)
from scanner.watchlist import (
    InvalidationReason,
    WatchedZone,
    Watchlist,
    WatchState,
)

__all__ = [
    "DEFAULT_ACCOUNT_EQUITY",
    "DEFAULT_CONFIRM_WINDOW",
    "DEFAULT_LTF_MIN_FVG_PCT",
    "DEFAULT_MIN_BODY_RATIO",
    "DEFAULT_MIN_FVG_PCT",
    "DEFAULT_RANGE_LOOKBACK",
    "DEFAULT_REWARD_RATIO",
    "DEFAULT_RISK_PER_TRADE_PCT",
    "DEFAULT_STOP_BUFFER_PCT",
    "DEFAULT_SWING_STRENGTH",
    "STRUCTURE_LENGTH",
    "Candle",
    "ConfirmationRejection",
    "Direction",
    "ExecutionOrder",
    "FairValueGap",
    "FilterStage",
    "InvalidationReason",
    "LtfTrigger",
    "OrderBlock",
    "OrderBlockStrategy",
    "RangeZone",
    "Settings",
    "SimulationReport",
    "StrategyResult",
    "SwingRange",
    "TradePlan",
    "TradeSignal",
    "WatchState",
    "WatchedZone",
    "Watchlist",
    "build_execution_order",
    "build_trade_plan",
    "confirm_entry",
    "detect_choch",
    "detect_order_block",
    "find_choch",
    "find_fvg_after",
    "fvg_frame",
    "order_block_mask",
    "position_size",
    "premium_discount_frame",
    "simulate",
    "swing_points",
    "swing_range",
    "to_binance_symbol",
    "to_unified_symbol",
    "validate_ohlcv",
]

__version__ = "3.1.0"
