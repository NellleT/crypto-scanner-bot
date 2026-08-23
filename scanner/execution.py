"""Order payloads for the Binance execution engine.

Converts a :class:`~scanner.risk.TradePlan` into the parameter dictionaries the
Binance REST API expects. This module builds and validates payloads; it never
sends them. Routing, signing and keys belong to the execution engine.

Two orders describe one setup:

* the **entry**, a resting ``LIMIT`` order at the block's proximal edge;
* the **protective** pair, a ``STOP_LOSS_LIMIT`` and ``TAKE_PROFIT_LIMIT``
  that must only be submitted *after* the entry fills.

They are kept separate because submitting the protective legs early would have
them working against a position that does not exist yet. On Binance spot the
usual shape is an OCO placed on fill; :meth:`ExecutionOrder.oco_payload`
renders that form.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Final

from scanner.risk import TradePlan
from scanner.smc import Direction, OrderBlock

#: Quote assets recognised when converting a unified symbol to Binance's form.
#: Longest first so USDT is matched before USD.
_QUOTE_ASSETS: Final[tuple[str, ...]] = (
    "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USD", "BTC", "ETH", "BNB", "EUR", "GBP",
)

_SYMBOL_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z0-9]{2,20}$")


def to_binance_symbol(symbol: str) -> str:
    """``BTC/USDT`` → ``BTCUSDT``; already-native symbols pass through.

    Any settlement suffix (``BTC/USDT:USDT``) is dropped — the execution engine
    is told the instrument, not the margin mode.
    """
    native = symbol.split(":")[0].replace("/", "").replace("-", "").upper()
    if not _SYMBOL_RE.match(native):
        raise ValueError(f"Cannot express {symbol!r} as a Binance symbol.")
    return native


def to_unified_symbol(symbol: str) -> str:
    """``BTCUSDT`` → ``BTC/USDT``; already-unified symbols pass through.

    Splits on the longest recognised quote asset, so ``BTCUSDT`` resolves to
    ``BTC/USDT`` rather than ``BTCUSD`` + ``T``.
    """
    cleaned = symbol.strip().upper()
    if "/" in cleaned:
        return cleaned
    for quote in _QUOTE_ASSETS:
        if cleaned.endswith(quote) and len(cleaned) > len(quote):
            return f"{cleaned[: -len(quote)]}/{quote}"
    raise ValueError(
        f"Cannot split {symbol!r} into base/quote. Use unified form, e.g. 'BTC/USDT'."
    )


@dataclass(frozen=True, slots=True)
class ExecutionOrder:
    """A routable description of one order-block setup."""

    symbol: str          # Binance-native, e.g. BTCUSDT
    side: str            # BUY | SELL
    quantity: str        # venue-precision strings, never floats
    entry: str
    stop_loss: str
    take_profit: str
    reward_ratio: float
    risk_pct: float
    risk_amount: float
    time_in_force: str = "GTC"

    @property
    def exit_side(self) -> str:
        """Side that closes the position."""
        return "SELL" if self.side == "BUY" else "BUY"

    def entry_payload(self) -> dict[str, Any]:
        """``POST /api/v3/order`` parameters for the resting entry."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "type": "LIMIT",
            "timeInForce": self.time_in_force,
            "quantity": self.quantity,
            "price": self.entry,
        }

    def oco_payload(self) -> dict[str, Any]:
        """``POST /api/v3/orderList/oco`` parameters, to submit once filled.

        One cancels the other, so the target and the stop cannot both execute.
        """
        return {
            "symbol": self.symbol,
            "side": self.exit_side,
            "quantity": self.quantity,
            "aboveType": "LIMIT_MAKER" if self.side == "BUY" else "STOP_LOSS_LIMIT",
            "belowType": "STOP_LOSS_LIMIT" if self.side == "BUY" else "LIMIT_MAKER",
            "abovePrice": self.take_profit if self.side == "BUY" else self.stop_loss,
            "belowPrice": self.stop_loss if self.side == "BUY" else self.take_profit,
            "belowStopPrice": self.stop_loss if self.side == "BUY" else self.take_profit,
            "belowTimeInForce": self.time_in_force,
        }

    def to_dict(self) -> dict[str, Any]:
        """Complete setup: both payloads plus the risk context behind them."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_order": self.entry_payload(),
            "exit_oco": self.oco_payload(),
            "risk": {
                "reward_ratio": self.reward_ratio,
                "risk_pct_of_equity": self.risk_pct,
                "risk_amount": round(self.risk_amount, 8),
            },
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


def build_execution_order(
    symbol: str,
    plan: TradePlan,
    block: OrderBlock,
    *,
    price_to_precision: Callable[[str, float], str | None] | None = None,
    amount_to_precision: Callable[[str, float], str | None] | None = None,
) -> ExecutionOrder:
    """Render ``plan`` as a routable order.

    Prices and quantity are emitted as **strings** at the venue's tick and lot
    precision. Binance rejects orders that violate ``PRICE_FILTER`` or
    ``LOT_SIZE``, and float repr is exactly how an over-precise value slips
    through — ``0.1 + 0.2`` is not ``0.3``.
    """

    def price(value: float) -> str:
        if price_to_precision is not None:
            rendered = price_to_precision(symbol, value)
            if rendered is not None:
                return rendered
        return f"{value:.8f}".rstrip("0").rstrip(".") or "0"

    def amount(value: float) -> str:
        if amount_to_precision is not None:
            rendered = amount_to_precision(symbol, value)
            if rendered is not None:
                return rendered
        return f"{value:.8f}".rstrip("0").rstrip(".") or "0"

    return ExecutionOrder(
        symbol=to_binance_symbol(symbol),
        side=block.direction.binance_side,
        quantity=amount(plan.quantity),
        entry=price(plan.entry),
        stop_loss=price(plan.stop_loss),
        take_profit=price(plan.take_profit),
        reward_ratio=plan.reward_ratio,
        risk_pct=plan.risk_pct,
        risk_amount=plan.risk_amount,
    )


__all__ = [
    "Direction",
    "ExecutionOrder",
    "build_execution_order",
    "to_binance_symbol",
    "to_unified_symbol",
]
