"""Higher-timeframe signal generation: which order blocks become watched zones.

The v3.1 HTF filter chain, in order:

=================  =====================================================
Stage              Requirement
=================  =====================================================
``warmup``         Enough closed candles to describe a structure
``order_block``    ``[-3]`` opposes the impulse at ``[-2]``, both with a
                   real body
``fvg``            The displacement left a gap at all
``displacement``   That gap is at least ``min_fvg_pct`` of price
``premium_discount`` The block sits in the correct half of the dealing
                   range — discount for longs, premium for shorts
``stop_width``     The resulting stop is within ``max_stop_pct`` of entry
``risk``           Entry, stop, target and size are all expressible
``watchlist``      Accepted; the zone is now tracked toward an entry
=================  =====================================================

Reaching ``watchlist`` is where this module stops. No order is built here — v3.1
does not enter blind. The zone is handed to :mod:`scanner.watchlist`, and only a
lower-timeframe confirmation (:mod:`scanner.mtf`) turns it into an order.
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
    DEFAULT_MAX_STOP_PCT,
    DEFAULT_REWARD_RATIO,
    DEFAULT_RISK_PER_TRADE_PCT,
    DEFAULT_STOP_BUFFER_PCT,
    TradePlan,
    build_trade_plan,
)
from scanner.smc import (
    DEFAULT_MIN_FVG_PCT,
    DEFAULT_RANGE_LOOKBACK,
    STRUCTURE_LENGTH,
    Direction,
    OrderBlock,
    detect_order_block,
)

logger: Final[logging.Logger] = logging.getLogger(__name__)


class FilterStage(str, Enum):
    """Where an evaluation ended.

    Members are declared in filter-chain order and depth is derived from that
    order, so inserting a stage does not require editing the ``reached_*``
    helpers by hand.
    """

    WARMUP = "warmup"
    ORDER_BLOCK = "order_block"
    FVG = "fvg"
    DISPLACEMENT = "displacement"
    PREMIUM_DISCOUNT = "premium_discount"
    STOP_WIDTH = "stop_width"
    RISK = "risk"
    WATCHLIST = "watchlist"

    @property
    def depth(self) -> int:
        """Position in the filter chain; later means more filters passed."""
        return _STAGE_ORDER.index(self)

    def passed(self, stage: "FilterStage") -> bool:
        """True when evaluation got *past* ``stage`` — that filter accepted it."""
        return self.depth > stage.depth

    @property
    def reached_order_block(self) -> bool:
        """True when a structural order block was present, whatever followed."""
        return self.passed(FilterStage.ORDER_BLOCK)

    @property
    def reached_fvg(self) -> bool:
        """True when a gap existed, before the size threshold was applied."""
        return self.passed(FilterStage.FVG)

    @property
    def reached_displacement(self) -> bool:
        """True when the gap also cleared the displacement threshold."""
        return self.passed(FilterStage.DISPLACEMENT)

    @property
    def reached_spatial(self) -> bool:
        """True when the block was also on the right side of the range."""
        return self.passed(FilterStage.PREMIUM_DISCOUNT)

    @property
    def reached_stop_width(self) -> bool:
        """True when the resulting stop was also tight enough to be tradable."""
        return self.passed(FilterStage.STOP_WIDTH)


#: Filter-chain order, taken from the declaration order above.
_STAGE_ORDER: Final[tuple[FilterStage, ...]] = tuple(FilterStage)

#: Maps a :class:`scanner.smc.StructureRejection` stage to a filter stage.
_SMC_STAGE: Final[dict[str, FilterStage]] = {
    "warmup": FilterStage.WARMUP,
    "order_block": FilterStage.ORDER_BLOCK,
    "fvg": FilterStage.FVG,
    "displacement": FilterStage.DISPLACEMENT,
    "premium_discount": FilterStage.PREMIUM_DISCOUNT,
}


@dataclass(frozen=True, slots=True)
class TradeSignal:
    """A validated HTF setup, sized and ready to be watched."""

    symbol: str
    timeframe: str
    block: OrderBlock
    plan: TradePlan

    @property
    def direction(self) -> Direction:
        return self.block.direction

    @property
    def candle_open_time(self) -> datetime:
        """Open time of the order block candle — the identity of the zone."""
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
    """Extreme order blocks with displacement, sized to fixed account risk."""

    min_body_ratio: float = DEFAULT_MIN_BODY_RATIO
    min_fvg_pct: float = DEFAULT_MIN_FVG_PCT
    range_lookback: int = DEFAULT_RANGE_LOOKBACK
    require_extreme: bool = True
    stop_buffer_pct: float = DEFAULT_STOP_BUFFER_PCT
    max_stop_pct: float = DEFAULT_MAX_STOP_PCT
    reward_ratio: float = DEFAULT_REWARD_RATIO
    account_equity: float = DEFAULT_ACCOUNT_EQUITY
    risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE_PCT

    @property
    def required_candles(self) -> int:
        """Closed candles needed before the newest bar can be evaluated.

        The structure needs three; the dealing range wants its full lookback
        before premium/discount means anything.
        """
        return max(STRUCTURE_LENGTH, self.range_lookback)

    def evaluate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> StrategyResult:
        """Evaluate the newest closed candles for a watchable order block.

        ``df`` must contain only closed candles — a still-forming bar would let
        the gap appear and vanish while the candle develops.
        """
        block, rejection = detect_order_block(
            df,
            min_body_ratio=self.min_body_ratio,
            min_fvg_pct=self.min_fvg_pct,
            range_lookback=self.range_lookback,
            require_extreme=self.require_extreme,
        )

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

        # Width policy, applied after the arithmetic rather than inside it: the
        # plan is expressible, it is just too loose to be worth taking. Sizing
        # holds the loss at RISK_PER_TRADE_PCT whatever the stop width, so this
        # is not about account risk — a zone this thick has not located anything
        # precise enough to trade, and the position it implies is negligible.
        if 0.0 < self.max_stop_pct < plan.risk_pct_of_entry:
            return StrategyResult.rejected(
                FilterStage.STOP_WIDTH,
                f"{block.direction.value} order block rejected: stop is "
                f"{plan.risk_pct_of_entry:.2f}% from entry, wider than the "
                f"{self.max_stop_pct:g}% limit — the zone "
                f"[{block.candle.low:g}, {block.candle.high:g}] is too thick to "
                "have located anything",
            )

        zone_name = block.zone.value if block.zone is not None else "unknown"
        return StrategyResult(
            signal=TradeSignal(
                symbol=symbol, timeframe=timeframe, block=block, plan=plan
            ),
            reason=(
                f"{block.direction.value} extreme order block accepted: "
                f"{block.fvg.pct:.2f}% displacement, {zone_name} half — "
                "watching for a lower-timeframe entry"
            ),
            stage=FilterStage.WATCHLIST,
        )
