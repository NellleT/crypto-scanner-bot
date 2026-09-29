"""V4.0 — liquidity-sweep deviations (mean reversion). Research, not live.

Thesis: resting stops cluster just beyond obvious swing highs (buy-side
liquidity, BSL) and swing lows (sell-side liquidity, SSL). When price runs those
stops and cannot hold beyond them, the break was a liquidity grab rather than a
breakout, and price tends to revert into the range. V4 fades that failure.

On closed candles of a single timeframe:

1. **Liquidity pools.** Swing highs and lows confirmed by ``strength`` candles on
   each side, formed within the last ``lookback`` candles, and not yet traded
   through. The dealing range runs from the lowest untaken swing low (SSL) to the
   highest untaken swing high (BSL); equilibrium is its midpoint.
2. **Sweep.** A candle trades beyond a pool. If it closes back inside the range
   it is a deviation at once — the wick was the grab. If it closes outside, the
   deviation is only confirmed by a close back inside within ``reclaim_bars``
   candles; otherwise the break was accepted and nothing is traded.
3. **Entry** at market on the close of the confirming candle.
4. **Stop** just beyond the most extreme price the sweep reached.
5. **Target** equilibrium, or the opposite pool.

Causality is structural, not a matter of care: a pivot is added to the pool set
only once its right-hand candles have closed, and every candle is judged against
the range as it stood *before* that candle opened. :func:`scan_deviations` is the
single implementation — the backtest runs it over history, and the live path
(:func:`latest_deviation`) reads the newest candle from the very same function.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Final

import pandas as pd

from scanner.candles import validate_ohlcv
from scanner.risk import (
    DEFAULT_ACCOUNT_EQUITY,
    DEFAULT_MAX_STOP_PCT,
    DEFAULT_RISK_PER_TRADE_PCT,
    TradePlan,
    position_size,
)
from scanner.smc import Direction, swing_points

logger: Final[logging.Logger] = logging.getLogger(__name__)

#: Candles back that a swing may have formed and still define the range.
DEFAULT_DEV_LOOKBACK: Final[int] = 50

#: Candles either side of a pivot. Three makes a swing structural rather than a
#: wiggle, at the cost of three candles of confirmation lag.
DEFAULT_DEV_STRENGTH: Final[int] = 3

#: Candles allowed to close back inside after a sweep candle closed outside.
DEFAULT_RECLAIM_BARS: Final[int] = 2

#: Offset beyond the sweep's extreme, in percent. Small on purpose: the extreme
#: is where the grab ran out of orders, so the stop only needs to clear the tick
#: noise around it.
DEFAULT_DEV_STOP_BUFFER_PCT: Final[float] = 0.1


class PoolSide(str, Enum):
    """Which side of the range a pool of resting orders sits on."""

    BUY_SIDE = "BSL"    # buy stops above a swing high
    SELL_SIDE = "SSL"   # sell stops below a swing low

    @property
    def fade(self) -> Direction:
        """Direction of the reversion trade after this pool is swept."""
        return Direction.SHORT if self is PoolSide.BUY_SIDE else Direction.LONG


class TargetMode(str, Enum):
    """Where a deviation trade takes profit."""

    EQUILIBRIUM = "equilibrium"   # the range midpoint
    OPPOSITE = "opposite"         # the liquidity pool on the far side


@dataclass(frozen=True, slots=True)
class DeviationParams:
    """Every tunable of the model, validated once."""

    lookback: int = DEFAULT_DEV_LOOKBACK
    strength: int = DEFAULT_DEV_STRENGTH
    reclaim_bars: int = DEFAULT_RECLAIM_BARS
    stop_buffer_pct: float = DEFAULT_DEV_STOP_BUFFER_PCT
    target: TargetMode = TargetMode.EQUILIBRIUM
    max_stop_pct: float = DEFAULT_MAX_STOP_PCT
    min_reward_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.strength < 1:
            raise ValueError(f"strength must be >= 1, got {self.strength}.")
        if self.lookback <= 2 * self.strength:
            raise ValueError(
                f"lookback ({self.lookback}) must exceed 2 * strength "
                f"({2 * self.strength}), or no pivot can form inside it."
            )
        if self.reclaim_bars < 0:
            raise ValueError(f"reclaim_bars must be >= 0, got {self.reclaim_bars}.")
        if self.stop_buffer_pct < 0 or self.max_stop_pct < 0 or self.min_reward_ratio < 0:
            raise ValueError("buffer, max stop and minimum reward ratio must be >= 0.")

    @property
    def required_candles(self) -> int:
        """Candles a live frame needs so its newest verdict matches full history:
        every pool it can use, plus the longest reclaim still pending."""
        return self.lookback + self.strength + self.reclaim_bars + 1


@dataclass(frozen=True, slots=True)
class LiquidityPool:
    """Resting orders beyond one confirmed swing point."""

    side: PoolSide
    price: float
    formed_ms: int   # open of the pivot candle


@dataclass(frozen=True, slots=True)
class DealingRange:
    """The range a sweep is judged against, as it stood before the sweep."""

    high: LiquidityPool   # BSL
    low: LiquidityPool    # SSL

    @property
    def equilibrium(self) -> float:
        return (self.high.price + self.low.price) / 2.0

    @property
    def height_pct(self) -> float:
        return (self.high.price - self.low.price) / self.low.price * 100.0

    def pool(self, side: PoolSide) -> LiquidityPool:
        return self.high if side is PoolSide.BUY_SIDE else self.low

    def opposite(self, side: PoolSide) -> LiquidityPool:
        return self.low if side is PoolSide.BUY_SIDE else self.high


@dataclass(frozen=True, slots=True)
class Deviation:
    """A confirmed sweep of one liquidity pool that failed to hold."""

    swept: PoolSide
    range: DealingRange
    sweep_ms: int        # open of the candle that pierced the pool
    confirm_ms: int      # open of the candle whose close confirmed the failure
    extreme: float       # furthest price the sweep reached
    close: float         # the confirming close — the market entry
    reclaim_bars: int    # 0 when the sweep candle itself closed back inside

    @property
    def direction(self) -> Direction:
        return self.swept.fade

    @property
    def pool(self) -> LiquidityPool:
        return self.range.pool(self.swept)

    @property
    def depth_pct(self) -> float:
        """How far beyond the pool the sweep ran, in percent of the pool."""
        return abs(self.extreme - self.pool.price) / self.pool.price * 100.0


@dataclass(slots=True)
class _Pending:
    """A sweep candle closed outside; waiting for a close back inside."""

    side: PoolSide
    range: DealingRange
    sweep_index: int
    extreme: float


def scan_deviations(
    df: pd.DataFrame, params: DeviationParams | None = None
) -> list[Deviation]:
    """Every deviation in ``df``, in the order they confirmed.

    Candle ``t`` is judged against the pools known before it opened: pivots
    whose ``strength`` right-hand candles had all closed by ``t - 1``, formed at
    or after ``t - lookback``, and never traded through since. Only after that
    judgement does candle ``t`` update the pools — removing any it traded
    through, and adding the pivot its own close has just confirmed. Appending
    later candles therefore never changes an earlier verdict.

    A candle that pierces both ends of the range at once says nothing about
    which side failed, so it is not traded.
    """
    params = params or DeviationParams()
    validate_ohlcv(df)
    if df.empty:
        return []

    timestamps = df["timestamp"].to_numpy(dtype="int64")
    highs = df["high"].to_numpy(dtype="float64")
    lows = df["low"].to_numpy(dtype="float64")
    closes = df["close"].to_numpy(dtype="float64")
    is_high, is_low = (s.to_numpy(dtype=bool) for s in swing_points(df, strength=params.strength))

    buy_side: list[tuple[int, float]] = []    # (pivot index, price), untaken
    sell_side: list[tuple[int, float]] = []
    pending: dict[PoolSide, _Pending] = {}
    found: list[Deviation] = []

    def pool(side: PoolSide, index: int, price: float) -> LiquidityPool:
        return LiquidityPool(side, price, int(timestamps[index]))

    for t in range(len(df)):
        # 1. The range as it stood before candle t opened.
        cutoff = t - params.lookback
        buy_side = [p for p in buy_side if p[0] >= cutoff]
        sell_side = [p for p in sell_side if p[0] >= cutoff]
        current: DealingRange | None = None
        if buy_side and sell_side:
            top = max(buy_side, key=lambda p: p[1])
            bottom = min(sell_side, key=lambda p: p[1])
            if top[1] > bottom[1]:
                current = DealingRange(
                    pool(PoolSide.BUY_SIDE, *top), pool(PoolSide.SELL_SIDE, *bottom)
                )

        # 2. Sweeps that closed outside: did this candle close back inside?
        for side, wait in list(pending.items()):
            level = wait.range.pool(side).price
            if side is PoolSide.BUY_SIDE:
                wait.extreme = max(wait.extreme, float(highs[t]))
                back_inside = closes[t] < level
            else:
                wait.extreme = min(wait.extreme, float(lows[t]))
                back_inside = closes[t] > level
            if back_inside:
                found.append(
                    Deviation(side, wait.range, int(timestamps[wait.sweep_index]),
                              int(timestamps[t]), wait.extreme, float(closes[t]),
                              t - wait.sweep_index)
                )
                del pending[side]
            elif t - wait.sweep_index >= params.reclaim_bars:
                del pending[side]   # held outside: an accepted break, not a grab

        # 3. New sweeps by candle t.
        if current is not None:
            above = highs[t] > current.high.price
            below = lows[t] < current.low.price
            if above != below:
                side = PoolSide.BUY_SIDE if above else PoolSide.SELL_SIDE
                level = current.pool(side).price
                extreme = float(highs[t] if above else lows[t])
                inside = closes[t] < level if above else closes[t] > level
                if inside:
                    found.append(
                        Deviation(side, current, int(timestamps[t]), int(timestamps[t]),
                                  extreme, float(closes[t]), 0)
                    )
                elif params.reclaim_bars > 0 and side not in pending:
                    pending[side] = _Pending(side, current, t, extreme)

        # 4. Candle t updates the pools: what it traded through is gone, and
        #    its close confirms the pivot `strength` candles back.
        buy_side = [p for p in buy_side if highs[t] <= p[1]]
        sell_side = [p for p in sell_side if lows[t] >= p[1]]
        pivot = t - params.strength
        if pivot >= 0:
            if is_high[pivot]:
                buy_side.append((pivot, float(highs[pivot])))
            if is_low[pivot]:
                sell_side.append((pivot, float(lows[pivot])))

    return found


def latest_deviation(
    df: pd.DataFrame, params: DeviationParams | None = None
) -> Deviation | None:
    """The deviation confirmed by the newest closed candle, if any — the live
    scanner's question, answered by the same code the backtest runs."""
    params = params or DeviationParams()
    if len(df) < params.required_candles:
        return None
    found = scan_deviations(df, params)
    if found and found[-1].confirm_ms == int(df["timestamp"].iloc[-1]):
        return found[-1]
    return None


@dataclass(frozen=True, slots=True)
class DeviationRejection:
    """Why a confirmed deviation does not become a trade."""

    stage: str   # no_reward | stop_width | reward_ratio | risk
    reason: str


def plan_deviation(
    deviation: Deviation,
    params: DeviationParams | None = None,
    *,
    equity: float = DEFAULT_ACCOUNT_EQUITY,
    risk_pct: float = DEFAULT_RISK_PER_TRADE_PCT,
) -> tuple[TradePlan | None, DeviationRejection | None]:
    """Size a deviation as a market entry with a stop beyond the sweep.

    The target is equilibrium or the opposite pool; the reward ratio is whatever
    that geometry gives, not a fixed multiple. A confirming close already past
    the target leaves nothing to take, and is rejected rather than inverted.
    """
    params = params or DeviationParams()
    entry = deviation.close
    offset = params.stop_buffer_pct / 100.0
    short = deviation.direction is Direction.SHORT
    stop = deviation.extreme * (1.0 + offset) if short else deviation.extreme * (1.0 - offset)
    target = (
        deviation.range.equilibrium
        if params.target is TargetMode.EQUILIBRIUM
        else deviation.range.opposite(deviation.swept).price
    )
    risk = (stop - entry) if short else (entry - stop)
    reward = (entry - target) if short else (target - entry)

    if not (risk > 0 and math.isfinite(stop)):
        return None, DeviationRejection("risk", f"stop {stop:g} is not beyond entry {entry:g}")
    if reward <= 0:
        return None, DeviationRejection(
            "no_reward",
            f"entry {entry:g} is already past the {params.target.value} target {target:g}",
        )
    stop_pct = risk / entry * 100.0
    if params.max_stop_pct > 0 and stop_pct > params.max_stop_pct:
        return None, DeviationRejection(
            "stop_width", f"stop {stop_pct:.2f}% wide exceeds {params.max_stop_pct:g}%"
        )
    ratio = reward / risk
    if ratio < params.min_reward_ratio:
        return None, DeviationRejection(
            "reward_ratio", f"1:{ratio:.2f} is below the 1:{params.min_reward_ratio:g} floor"
        )
    quantity = position_size(equity=equity, risk_pct=risk_pct, risk_per_unit=risk)
    if quantity <= 0:
        return None, DeviationRejection("risk", "position size resolved to zero")
    return (
        TradePlan(
            direction=deviation.direction,
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            reward_ratio=ratio,
            buffer_pct=params.stop_buffer_pct,
            equity=equity,
            risk_pct=risk_pct,
            quantity=quantity,
        ),
        None,
    )


__all__ = [
    "DEFAULT_DEV_LOOKBACK",
    "DEFAULT_DEV_STOP_BUFFER_PCT",
    "DEFAULT_DEV_STRENGTH",
    "DEFAULT_RECLAIM_BARS",
    "DealingRange",
    "Deviation",
    "DeviationParams",
    "DeviationRejection",
    "LiquidityPool",
    "PoolSide",
    "TargetMode",
    "latest_deviation",
    "plan_deviation",
    "scan_deviations",
]
