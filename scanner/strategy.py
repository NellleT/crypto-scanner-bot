"""Signal generation: order blocks validated by fair value gaps.

The v3.0 filter chain, in order:

==============  =====================================================
Stage           Requirement
==============  =====================================================
``warmup``      At least three closed candles
``order_block`` ``[-3]`` opposes the impulse at ``[-2]``, both with a
                real body
``fvg``         The displacement left a gap: ``low[-1] > high[-3]``
                for longs, ``high[-1] < low[-3]`` for shorts
``risk``        Entry, stop, target and size are all expressible
==============  =====================================================

This replaces the v2.x engulfing + SMA + volume chain entirely. Every
evaluation returns a :class:`StrategyResult` carrying the stage it stopped at,
so the funnel stays observable at DEBUG level instead of signals silently
disappearing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Final

import pandas as pd

from scanner.candles import DEFAULT_MIN_BODY_RATIO
from scanner.risk import (
    DEFAULT_ACCOUNT_EQUITY,
    DEFAULT_REWARD_RATIO,
    DEFAULT_RISK_PER_TRADE_PCT,
    DEFAULT_STOP_BUFFER_PCT,
    TradePlan,
    build_trade_plan,
)
from scanner.smc import STRUCTURE_LENGTH, Direction, OrderBlock, detect_order_block

logger: Final[logging.Logger] = logging.getLogger(__name__)


class FilterStage(str, Enum):
    """Where an evaluation ended.

    Members are declared in filter-chain order and depth is derived from that
    order, so inserting a stage does not require editing the ``reached_*``
    helpers by hand.
    """

    WARMUP = "warmup"            # not enough closed candles
    ORDER_BLOCK = "order_block"  # no opposing candle before a displacement
    FVG = "fvg"                  # displacement left no fair value gap
    RISK = "risk"                # structure valid but not sizeable
    CONFIRMED = "confirmed"      # all filters passed

    @property
    def depth(self) -> int:
        """Position in the filter chain; later means more filters passed."""
        return _STAGE_ORDER.index(self)

    def passed(self, stage: "FilterStage") -> bool:
        """True when evaluation got *past* ``stage`` — that filter accepted it."""
        return self.depth > stage.depth

    @property
    def reached_order_block(self) -> bool:
        """True when a structural order block was present, gap or not."""
        return self.passed(FilterStage.ORDER_BLOCK)

    @property
    def reached_fvg(self) -> bool:
        """True when the block was also validated by a fair value gap."""
        return self.passed(FilterStage.FVG)


#: Filter-chain order, taken from the declaration order above.
_STAGE_ORDER: Final[tuple[FilterStage, ...]] = tuple(FilterStage)

#: Maps a :class:`scanner.smc.StructureRejection` stage to a filter stage.
_SMC_STAGE: Final[dict[str, FilterStage]] = {
    "warmup": FilterStage.WARMUP,
    "order_block": FilterStage.ORDER_BLOCK,
    "fvg": FilterStage.FVG,
}


@dataclass(frozen=True, slots=True)
class TradeSignal:
    """A validated, sized setup ready to be alerted and routed."""

    symbol: str
    timeframe: str
    block: OrderBlock
    plan: TradePlan

    @property
    def direction(self) -> Direction:
        return self.block.direction

    @property
    def candle_open_time(self) -> datetime:
        """Open time of the order block candle — the zone's identity."""
        return self.block.open_time

    @property
    def confirmed_at(self) -> datetime:
        """Open time of the candle that completed the structure."""
        return self.block.confirmation.open_time

    @property
    def dedup_key(self) -> tuple[str, str, int, str]:
        return (self.symbol, self.timeframe, *self.block.dedup_key)


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
class OrderBlockStrategy:
    """Order blocks validated by fair value gaps, sized to fixed account risk."""

    min_body_ratio: float = DEFAULT_MIN_BODY_RATIO
    stop_buffer_pct: float = DEFAULT_STOP_BUFFER_PCT
    reward_ratio: float = DEFAULT_REWARD_RATIO
    account_equity: float = DEFAULT_ACCOUNT_EQUITY
    risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE_PCT

    @property
    def required_candles(self) -> int:
        """Closed candles needed before the newest bar can be evaluated."""
        return STRUCTURE_LENGTH

    def evaluate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> StrategyResult:
        """Evaluate the newest three CLOSED candles.

        ``df`` must contain only closed candles — a still-forming bar would let
        the gap appear and vanish while the candle develops.
        """
        block, rejection = detect_order_block(df, min_body_ratio=self.min_body_ratio)

        if block is None:
            assert rejection is not None  # detect returns one or the other
            return StrategyResult.rejected(
                _SMC_STAGE.get(rejection.stage, FilterStage.ORDER_BLOCK),
                rejection.reason,
            )

        plan = build_trade_plan(
            block,
            buffer_pct=self.stop_buffer_pct,
            reward_ratio=self.reward_ratio,
            equity=self.account_equity,
            risk_pct=self.risk_per_trade_pct,
        )
        if plan is None:
            return StrategyResult.rejected(
                FilterStage.RISK,
                f"{block.direction.value} order block rejected: no tradable "
                "entry/stop/target structure",
            )

        return StrategyResult(
            signal=TradeSignal(
                symbol=symbol, timeframe=timeframe, block=block, plan=plan
            ),
            reason=f"{block.direction.value} order block validated by fair value gap",
            stage=FilterStage.CONFIRMED,
        )
