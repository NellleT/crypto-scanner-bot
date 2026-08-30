"""Order-block trade construction: entry, stop, target and position size.

Levels come from the order block itself rather than from a lookback window:

* **Entry** — a resting limit order at the block's *proximal* edge, the side
  price returns to first. Nothing is bought at market; the setup only becomes a
  position if price comes back into the zone.
* **Stop** — just beyond the *distal* edge, offset by ``buffer_pct``. Sitting
  exactly on the edge puts the stop inside the pool of orders that a liquidity
  sweep is aiming at.
* **Target** — a fixed multiple of the risk distance.
* **Size** — a fixed fraction of account equity, derived only from the distance
  between entry and stop.

This module imports nothing from :mod:`scanner.strategy`; the dependency runs
strategy → risk.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final

from scanner.smc import Direction, OrderBlock

logger: Final[logging.Logger] = logging.getLogger(__name__)

#: Offset beyond the block's distal edge, in **percent** (0.2 == 0.2%).
DEFAULT_STOP_BUFFER_PCT: Final[float] = 0.2

#: Reward-to-risk multiple for the take-profit.
DEFAULT_REWARD_RATIO: Final[float] = 4.0

#: Share of account equity risked per trade, in **percent**.
DEFAULT_RISK_PER_TRADE_PCT: Final[float] = 1.0

#: Widest stop, as a percentage of entry, that still counts as a tradable
#: structure. Fixed-fraction sizing keeps the *loss* constant however wide the
#: stop is, so an unusually thick order block does not blow up account risk —
#: it just sizes down to a position too small to be worth the fees, on a zone
#: too loose to have located anything. Rejecting is more honest than sizing it.
#: Set to 0.0 to disable the check.
DEFAULT_MAX_STOP_PCT: Final[float] = 3.5

#: Fallback account equity, in quote currency, when none is configured.
DEFAULT_ACCOUNT_EQUITY: Final[float] = 10_000.0


@dataclass(frozen=True, slots=True)
class TradePlan:
    """A fully sized order-block trade, ready to be routed or displayed."""

    direction: Direction
    entry: float
    stop_loss: float
    take_profit: float
    reward_ratio: float
    buffer_pct: float
    equity: float
    risk_pct: float
    quantity: float

    @property
    def risk_per_unit(self) -> float:
        """Distance from entry to stop, in quote currency per unit."""
        return abs(self.entry - self.stop_loss)

    @property
    def risk_pct_of_entry(self) -> float:
        """Stop distance as a percentage of entry — the move that stops us out."""
        if self.entry <= 0.0:
            return 0.0
        return self.risk_per_unit / self.entry * 100.0

    @property
    def risk_amount(self) -> float:
        """Quote-currency loss if the stop is hit, by construction."""
        return self.equity * self.risk_pct / 100.0

    @property
    def reward_amount(self) -> float:
        """Quote-currency gain if the target is hit."""
        return self.risk_amount * self.reward_ratio

    @property
    def notional(self) -> float:
        """Position value at entry."""
        return self.quantity * self.entry

    @property
    def leverage_required(self) -> float:
        """Notional as a multiple of equity.

        Sizing deliberately ignores leverage mechanics, so a tight stop can imply
        a notional well above equity. Reported rather than clamped: whether that
        is fundable is an account question, not a strategy one.
        """
        if self.equity <= 0.0:
            return 0.0
        return self.notional / self.equity

    @property
    def take_profit_move_pct(self) -> float:
        """Signed move price must make to reach the target — up positive."""
        if self.entry <= 0.0:
            return 0.0
        return (self.take_profit - self.entry) / self.entry * 100.0

    def prices(self) -> tuple[float, ...]:
        """Every price in the plan, for consistent group formatting."""
        return (self.entry, self.stop_loss, self.take_profit)


def build_trade_plan(
    block: OrderBlock,
    *,
    buffer_pct: float = DEFAULT_STOP_BUFFER_PCT,
    reward_ratio: float = DEFAULT_REWARD_RATIO,
    equity: float = DEFAULT_ACCOUNT_EQUITY,
    risk_pct: float = DEFAULT_RISK_PER_TRADE_PCT,
) -> TradePlan | None:
    """Size a validated order block, or return ``None`` if it is not tradable.

    The stop is placed *beyond* the distal edge — below it for a long, above it
    for a short. The requirement reads "distal + buffer", but a long stop moved
    up into the block would sit closer to the sweep it is meant to survive; the
    buffer only does its job pushing away from the zone.
    """
    entry = block.proximal
    distal = block.distal

    if entry <= 0.0 or distal <= 0.0:
        logger.warning("Rejecting trade plan: non-positive block edges.")
        return None

    offset = buffer_pct / 100.0
    if block.direction is Direction.LONG:
        stop_loss = distal * (1.0 - offset)
        risk_per_unit = entry - stop_loss
    else:
        stop_loss = distal * (1.0 + offset)
        risk_per_unit = stop_loss - entry

    if stop_loss <= 0.0 or not math.isfinite(stop_loss):
        logger.warning("Rejecting trade plan: stop-loss %.10g is not a real price.", stop_loss)
        return None

    if risk_per_unit <= 0.0:
        # A block whose proximal and distal edges coincide has no depth.
        logger.warning(
            "Rejecting trade plan: stop %.10g is on the wrong side of entry %.10g.",
            stop_loss,
            entry,
        )
        return None

    if block.direction is Direction.LONG:
        take_profit = entry + risk_per_unit * reward_ratio
    else:
        take_profit = entry - risk_per_unit * reward_ratio

    if take_profit <= 0.0:
        logger.warning(
            "Rejecting trade plan: 1:%g target would be %.10g, which is not a real "
            "price (risk is %.2f%% of entry).",
            reward_ratio,
            take_profit,
            risk_per_unit / entry * 100.0,
        )
        return None

    quantity = position_size(
        equity=equity, risk_pct=risk_pct, risk_per_unit=risk_per_unit
    )
    if quantity <= 0.0:
        logger.warning(
            "Rejecting trade plan: position size resolved to %.10g units.", quantity
        )
        return None

    return TradePlan(
        direction=block.direction,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reward_ratio=reward_ratio,
        buffer_pct=buffer_pct,
        equity=equity,
        risk_pct=risk_pct,
        quantity=quantity,
    )


def position_size(*, equity: float, risk_pct: float, risk_per_unit: float) -> float:
    """Units to trade so that being stopped out costs ``risk_pct`` of equity.

    Depends only on the entry-to-stop distance. Leverage, margin mode and
    contract multipliers are deliberately out of scope — the same number is
    correct on spot and on any leverage setting, because it is derived from the
    loss the stop implies rather than from the capital committed.
    """
    if equity <= 0.0 or risk_pct <= 0.0 or risk_per_unit <= 0.0:
        return 0.0
    return (equity * risk_pct / 100.0) / risk_per_unit
