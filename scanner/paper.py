"""V5.0 paper trading — one daily pass. Paper only: nothing is ever routed.

**State by replay.** The paper account is a pure function of four things: the
rules, the start date, the starting equity, and the market's closed daily
candles. Every run rebuilds it from ``PAPER_START`` to the newest close through
:class:`~scanner.trend_backtest.TrendPortfolio` — the engine the backtest uses —
so there is no state file to lose on an ephemeral CI runner, and the paper
record cannot drift from the tested model. The state is still written to JSON
on every run, to be read and audited.

A run reports three things:

* **BUY** — the newest close broke out; the paper account buys at the next open.
* **SELL (Trailing Stop Hit)** — the newest close broke the exit channel; the
  paper account sells at the next open.
* **Status** — equity, cash, and every open position with its trailing stop.

Orders queued at the newest close fill at the next day's open, which the next
run sees as a closed candle and books at its real price.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping

import pandas as pd

from scanner.notifier import format_money, format_price, format_quantity
from scanner.trend import TrendParams
from scanner.trend_backtest import SPOT_COSTS, Costs, Order, TrendPortfolio

TAG: Final[str] = "[PAPER]"


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _base(symbol: str) -> str:
    return symbol.split("/")[0]


@dataclass(frozen=True, slots=True)
class PaperPosition:
    """An open paper position, marked at the newest close."""

    symbol: str
    quantity: float
    entry_ms: int
    entry_price: float     # fill, slippage included
    cost: float            # cash spent, fee included
    initial_stop: float
    close: float           # newest close
    exit_below: float      # the trailing stop: a daily close under it sells

    @property
    def value(self) -> float:
        return self.quantity * self.close

    @property
    def unrealized(self) -> float:
        return self.value - self.cost

    @property
    def unrealized_pct(self) -> float:
        return self.unrealized / self.cost * 100.0

    @property
    def stop_distance_pct(self) -> float:
        return (self.exit_below / self.close - 1.0) * 100.0


@dataclass(frozen=True, slots=True)
class PaperState:
    """The paper account after its newest close."""

    as_of_ms: int | None          # open of the newest daily candle replayed
    start_ms: int
    start_equity: float
    equity: float
    cash: float
    positions: tuple[PaperPosition, ...]
    orders: tuple[Order, ...]     # queued for the next open
    closed: tuple[dict, ...]      # completed round trips
    params: TrendParams
    costs: Costs
    risk_pct: float

    @property
    def return_pct(self) -> float:
        return (self.equity / self.start_equity - 1.0) * 100.0

    def position(self, symbol: str) -> PaperPosition | None:
        return next((p for p in self.positions if p.symbol == symbol), None)


def replay(
    frames: Mapping[str, pd.DataFrame],
    *,
    start_ms: int,
    equity: float,
    params: TrendParams | None = None,
    costs: Costs = SPOT_COSTS,
    risk_pct: float = 1.0,
) -> PaperState:
    """Rebuild the paper account from ``start_ms`` to the newest close.

    The account starts flat: a trend already under way at the start is only
    joined on its next breakout. Candles before ``start_ms`` only warm up the
    indicators.
    """
    params = params or TrendParams()
    book = TrendPortfolio(frames, params, costs, equity=equity, risk_pct=risk_pct)
    for day in book.days(start_ms):
        book.step(day)

    positions = []
    for symbol, held in sorted(book.holdings.items()):
        frame = book.data[symbol]
        row = frame.loc[book.last_day] if book.last_day in frame.index else frame.iloc[-1]
        positions.append(PaperPosition(
            symbol, held.quantity, held.entry_ms, held.entry_price, held.cost,
            held.initial_stop, float(row["close"]), float(row["next_stop"]),
        ))
    orders = tuple(sorted(book.orders.values(), key=lambda o: (o.side != "SELL", o.symbol)))
    return PaperState(
        as_of_ms=book.last_day, start_ms=start_ms, start_equity=equity,
        equity=book.equity, cash=book.cash, positions=tuple(positions), orders=orders,
        closed=tuple(book.fills), params=params, costs=costs, risk_pct=risk_pct,
    )


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
def _buy_text(state: PaperState, order: Order) -> str:
    quantity = order.risk_amount / (order.close - order.stop)
    notional = quantity * order.close
    note = ""
    freed = sum(p.value for p in state.positions
                if any(o.side == "SELL" and o.symbol == p.symbol for o in state.orders))
    if notional > state.cash + freed:
        note = f" (more than the {format_money(state.cash + freed)} cash available: it will be scaled down)"
    return (
        f"{TAG} BUY {order.symbol} — daily close {format_price(order.close)} broke the "
        f"{state.params.entry_channel}-day high {format_price(order.level)}, above its "
        f"{state.params.sma}-day average. Paper order at the next open: "
        f"{format_quantity(quantity)} {_base(order.symbol)} ≈ ${format_money(notional)}{note}; "
        f"trailing stop {format_price(order.stop)} "
        f"({(order.stop / order.close - 1) * 100:+.1f}%), risking ${format_money(order.risk_amount)} "
        f"({state.risk_pct:g}% of equity)."
    )


def _sell_text(state: PaperState, order: Order) -> str:
    held = state.position(order.symbol)
    detail = ""
    if held is not None:
        proceeds = held.quantity * order.close * (1 - state.costs.slippage) * (1 - state.costs.fee)
        pnl = proceeds - held.cost
        detail = (
            f" Held {format_quantity(held.quantity)} {_base(order.symbol)} since "
            f"{_day(held.entry_ms)} at {format_price(held.entry_price)}: about "
            f"{pnl / held.cost * 100:+.1f}% ({'+' if pnl >= 0 else '-'}${format_money(abs(pnl))}) "
            "at this close, after costs."
        )
    return (
        f"{TAG} SELL {order.symbol} (Trailing Stop Hit) — daily close "
        f"{format_price(order.close)} fell below the {state.params.exit_channel}-day low "
        f"{format_price(order.level)}. Paper order at the next open.{detail}"
    )


def alerts(state: PaperState) -> list[str]:
    """One message per order queued at the newest close — sells first."""
    return [_sell_text(state, o) if o.side == "SELL" else _buy_text(state, o)
            for o in state.orders]


def _order_summary(state: PaperState) -> str:
    if not state.orders:
        return "no orders"
    return " · ".join(
        f"SELL {o.symbol} (Trailing Stop Hit)" if o.side == "SELL" else f"BUY {o.symbol}"
        for o in state.orders
    )


def status(state: PaperState) -> str:
    """The daily status block: account, open positions, next-open orders."""
    if state.as_of_ms is None:
        return (f"{TAG} V5 paper account starts at the daily close of {_day(state.start_ms)} "
                f"with ${format_money(state.start_equity)}; no close to evaluate yet.")
    lines = [
        f"{TAG} V5 daily status — close of {_day(state.as_of_ms)} (UTC)",
        f"Equity ${format_money(state.equity)} ({state.return_pct:+.2f}% since "
        f"{_day(state.start_ms)}) · cash ${format_money(state.cash)} · "
        f"{len(state.positions)} open · {len(state.closed)} closed",
    ]
    if not state.positions:
        lines.append("No open positions — waiting for a breakout.")
    for p in state.positions:
        lines.append(
            f"  {p.symbol:<10} {format_quantity(p.quantity)} {_base(p.symbol)} since "
            f"{_day(p.entry_ms)} @ "
            f"{format_price(p.entry_price)} · close {format_price(p.close)} "
            f"({p.unrealized_pct:+.1f}%) · sells on a close below {format_price(p.exit_below)} "
            f"({p.stop_distance_pct:+.1f}%)"
        )
    lines.append(f"Next open: {_order_summary(state)}")
    return "\n".join(lines)


def markdown(state: PaperState) -> str:
    """The same status as Markdown, for the GitHub Actions run summary."""
    if state.as_of_ms is None:
        return f"## V5 paper trading\n\n{status(state)}\n"
    out = [
        f"## V5 paper trading — close of {_day(state.as_of_ms)} (UTC)",
        "",
        f"**Equity** ${format_money(state.equity)} ({state.return_pct:+.2f}% since "
        f"{_day(state.start_ms)}) · **cash** ${format_money(state.cash)} · "
        f"**closed trades** {len(state.closed)}",
        "",
    ]
    if state.positions:
        out += ["| Symbol | Quantity | Entry | Close | P&L | Sells on a close below |",
                "| --- | --- | --- | --- | --- | --- |"]
        for p in state.positions:
            out.append(
                f"| {p.symbol} | {format_quantity(p.quantity)} {_base(p.symbol)} | "
                f"{format_price(p.entry_price)} "
                f"({_day(p.entry_ms)}) | {format_price(p.close)} | {p.unrealized_pct:+.1f}% | "
                f"{format_price(p.exit_below)} ({p.stop_distance_pct:+.1f}%) |"
            )
    else:
        out.append("No open positions — waiting for a breakout.")
    out += ["", f"**Orders for the next open:** {_order_summary(state)}"]
    out += [f"- {text.removeprefix(TAG + ' ')}" for text in alerts(state)]
    out += ["", "_Paper trading only — DRY_RUN=true, no order is ever sent._", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------
def _clean(value: Any) -> Any:
    """JSON-safe number: NaN becomes null, float noise is trimmed to 10
    significant digits (0.24534509999999998 -> 0.2453451)."""
    if isinstance(value, float):
        return float(f"{value:.10g}") if math.isfinite(value) else None
    return value


def to_dict(state: PaperState) -> dict[str, Any]:
    """The JSON record of one run. Informational: the next run replays rather
    than reads it."""
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of": _day(state.as_of_ms) if state.as_of_ms is not None else None,
        "strategy": {
            "name": "V5.0 daily trend-following", "sma": state.params.sma,
            "entry_channel": state.params.entry_channel,
            "exit_channel": state.params.exit_channel, "risk_pct": state.risk_pct,
        },
        "costs": {"fee": state.costs.fee, "slippage": state.costs.slippage},
        "account": {
            "start": _day(state.start_ms), "start_equity": round(state.start_equity, 2),
            "equity": round(state.equity, 2), "cash": round(state.cash, 2),
            "return_pct": round(state.return_pct, 4),
        },
        "positions": [
            {"symbol": p.symbol, "quantity": _clean(p.quantity), "entry_date": _day(p.entry_ms),
             "entry_price": _clean(p.entry_price), "cost": round(p.cost, 2),
             "initial_stop": _clean(p.initial_stop), "close": _clean(p.close),
             "unrealized_pct": round(p.unrealized_pct, 4), "exit_below": _clean(p.exit_below)}
            for p in state.positions
        ],
        "orders_for_next_open": [
            {"side": o.side, "symbol": o.symbol, "signal_date": _day(o.signal_ms),
             "close": _clean(o.close), "channel": _clean(o.level), "initial_stop": _clean(o.stop),
             "risk_amount": round(o.risk_amount, 2)}
            for o in state.orders
        ],
        "closed_trades": [
            {"symbol": f["symbol"], "entry_date": _day(f["entry_ms"]),
             "exit_date": _day(f["exit_ms"]), "quantity": _clean(f["quantity"]),
             "entry_price": _clean(f["entry_price"]), "exit_price": _clean(f["exit_price"]),
             "pnl": round(f["proceeds"] - f["cost"], 2),
             "pnl_pct": round((f["proceeds"] / f["cost"] - 1) * 100, 4), "exit": f["kind"]}
            for f in state.closed
        ],
    }


def save(state: PaperState, path: Path) -> None:
    """Write the record atomically, so a crash mid-write leaves the old one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(to_dict(state), stream, indent=2)
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


__all__ = [
    "PaperPosition",
    "PaperState",
    "alerts",
    "markdown",
    "replay",
    "save",
    "status",
    "to_dict",
]
