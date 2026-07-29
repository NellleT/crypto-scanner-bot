"""Structural stop-loss placement and R:R take-profit targets.

**Why structural, not candle-based.** A stop placed just under the signal
candle's own low sits exactly where every other reader of the same pattern puts
theirs. That cluster is visible resting liquidity, and price routinely wicks
through it before continuing in the original direction. Anchoring instead to the
extreme of the last ``lookback`` bars puts the stop beyond the recent swing —
past the pool rather than inside it — and the small percentage buffer keeps it
off the exact tick where orders pile up.

The trade-off is honest: a structural stop is wider, so position size must be
smaller for the same account risk. That is the point. It trades a worse entry
ratio for a materially lower chance of being swept out of a correct call.

This module deliberately imports nothing from :mod:`scanner.strategy` — the
dependency runs strategy → risk, so direction is passed as a plain ``is_long``
flag rather than a ``SignalDirection``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final, Sequence

import pandas as pd

logger: Final[logging.Logger] = logging.getLogger(__name__)

#: Bars scanned for the structural extreme, including the signal candle.
DEFAULT_STRUCTURAL_LOOKBACK: Final[int] = 10

#: Buffer beyond the structural extreme, in **percent** (0.1 == 0.1%).
DEFAULT_STOP_BUFFER_PCT: Final[float] = 0.1

#: Reward-to-risk multiples for the take-profit ladder.
DEFAULT_RR_TARGETS: Final[tuple[float, ...]] = (2.0, 3.0)


@dataclass(frozen=True, slots=True)
class TakeProfit:
    """One rung of the take-profit ladder."""

    ratio: float
    price: float

    @property
    def label(self) -> str:
        """Human-readable R:R, e.g. ``1:2``."""
        return f"1:{self.ratio:g}"


@dataclass(frozen=True, slots=True)
class RiskPlan:
    """Entry, structural stop and R-multiple targets for one signal."""

    is_long: bool
    entry: float
    stop_loss: float
    structural_level: float
    lookback: int
    buffer_pct: float
    take_profits: tuple[TakeProfit, ...]

    @property
    def risk_per_unit(self) -> float:
        """Absolute distance from entry to stop, in quote currency per unit."""
        return abs(self.entry - self.stop_loss)

    @property
    def risk_pct(self) -> float:
        """Risk as a percentage of entry — the move that would stop the trade out."""
        if self.entry <= 0.0:
            return 0.0
        return self.risk_per_unit / self.entry * 100.0

    @property
    def structural_label(self) -> str:
        """``low`` for longs, ``high`` for shorts — the extreme the stop anchors to."""
        return "low" if self.is_long else "high"

    @property
    def stop_description(self) -> str:
        """e.g. ``10-bar low`` — what the stop is anchored to."""
        return f"{self.lookback}-bar {self.structural_label}"

    def reward_pct(self, take_profit: TakeProfit) -> float:
        """Gain at the target, as a positive percentage of entry.

        Always positive: a short profits from a fall, and the magnitude is what
        sizing cares about.
        """
        if self.entry <= 0.0:
            return 0.0
        return abs(take_profit.price - self.entry) / self.entry * 100.0

    def price_move_pct(self, take_profit: TakeProfit) -> float:
        """Signed move price must make to reach the target — up positive.

        Distinct from :meth:`reward_pct` on purpose. A short's 1:2 target is a
        gain of 5.74% but a price move of **-5.74%**; printing "+5.74%" beside a
        target below the entry invites exactly the wrong read.
        """
        if self.entry <= 0.0:
            return 0.0
        return (take_profit.price - self.entry) / self.entry * 100.0

    def prices(self) -> tuple[float, ...]:
        """Every price in the plan, for consistent group formatting."""
        return (self.entry, self.stop_loss, *(tp.price for tp in self.take_profits))


def structural_extreme(df: pd.DataFrame, *, is_long: bool, lookback: int) -> float:
    """Lowest low (long) or highest high (short) over the last ``lookback`` bars.

    ``df`` must end at the signal candle, which is included in the window per the
    structural-stop definition. ``tail`` clamps naturally when fewer bars exist.
    """
    column = "low" if is_long else "high"
    if column not in df.columns:
        raise ValueError(f"Cannot locate structural level: missing {column!r} column.")
    if df.empty:
        raise ValueError("Cannot locate structural level: no candles supplied.")

    window = df[column].tail(max(lookback, 1))
    extreme = float(window.min() if is_long else window.max())
    if math.isnan(extreme):
        raise ValueError(f"Structural {column} is NaN over the lookback window.")
    return extreme


def build_risk_plan(
    df: pd.DataFrame,
    *,
    is_long: bool,
    entry: float,
    lookback: int = DEFAULT_STRUCTURAL_LOOKBACK,
    buffer_pct: float = DEFAULT_STOP_BUFFER_PCT,
    rr_targets: Sequence[float] = DEFAULT_RR_TARGETS,
) -> RiskPlan | None:
    """Build the stop and target ladder for a signal, or ``None`` if degenerate.

    ``df`` must contain only closed candles and end at the signal candle.

    A plan is rejected when the arithmetic cannot describe a tradable setup:
    non-positive risk, or any price at or below zero. The latter is not
    hypothetical — a short whose stop sits more than ``1/max(rr)`` of entry away
    produces a negative target, which no venue can express. Rejecting is safer
    than emitting a nonsensical number into an alert someone may act on.
    """
    if entry <= 0.0:
        logger.warning("Rejecting risk plan: non-positive entry %.10g.", entry)
        return None

    extreme = structural_extreme(df, is_long=is_long, lookback=lookback)

    # Push the stop just beyond the swing so it is not resting on the exact tick.
    buffer_multiplier = buffer_pct / 100.0
    stop_loss = (
        extreme * (1.0 - buffer_multiplier) if is_long else extreme * (1.0 + buffer_multiplier)
    )

    if stop_loss <= 0.0:
        logger.warning(
            "Rejecting risk plan: stop-loss %.10g is not positive (structural %.10g).",
            stop_loss,
            extreme,
        )
        return None

    # For a valid engulfing signal the entry always sits inside the structural
    # range (a bullish bar closes above its own low), so risk is positive by
    # construction. The guard covers malformed or stale data rather than normal
    # market conditions.
    risk_per_unit = entry - stop_loss if is_long else stop_loss - entry
    if risk_per_unit <= 0.0:
        logger.warning(
            "Rejecting risk plan: stop %.10g is on the wrong side of entry %.10g.",
            stop_loss,
            entry,
        )
        return None

    take_profits: list[TakeProfit] = []
    for ratio in rr_targets:
        if ratio <= 0.0:
            logger.warning("Skipping non-positive reward ratio %.10g.", ratio)
            continue
        price = entry + risk_per_unit * ratio if is_long else entry - risk_per_unit * ratio
        if price <= 0.0:
            logger.warning(
                "Rejecting risk plan: 1:%g target for a short would be %.10g, "
                "which is not a real price (risk is %.2f%% of entry).",
                ratio,
                price,
                risk_per_unit / entry * 100.0,
            )
            return None
        take_profits.append(TakeProfit(ratio=ratio, price=price))

    if not take_profits:
        logger.warning("Rejecting risk plan: no valid reward targets configured.")
        return None

    return RiskPlan(
        is_long=is_long,
        entry=entry,
        stop_loss=stop_loss,
        structural_level=extreme,
        lookback=lookback,
        buffer_pct=buffer_pct,
        take_profits=tuple(take_profits),
    )
