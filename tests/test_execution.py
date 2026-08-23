"""Unit tests for the Binance execution payloads."""

from __future__ import annotations

import json

import pytest

from scanner.execution import (
    build_execution_order,
    to_binance_symbol,
    to_unified_symbol,
)
from scanner.risk import build_trade_plan
from scanner.smc import Direction
from tests.test_risk import make_block


# ---------------------------------------------------------------------------
# Symbol conversion
# ---------------------------------------------------------------------------
def test_unified_to_binance() -> None:
    assert to_binance_symbol("BTC/USDT") == "BTCUSDT"
    assert to_binance_symbol("BTCUSDT") == "BTCUSDT"
    assert to_binance_symbol("btc/usdt") == "BTCUSDT"


def test_settlement_suffix_is_dropped() -> None:
    """The engine is told the instrument, not the margin mode."""
    assert to_binance_symbol("BTC/USDT:USDT") == "BTCUSDT"


def test_binance_to_unified_splits_on_the_longest_quote() -> None:
    """BTCUSDT must not resolve to BTCUSD + T."""
    assert to_unified_symbol("BTCUSDT") == "BTC/USDT"
    assert to_unified_symbol("BTCUSD") == "BTC/USD"
    assert to_unified_symbol("ETHBTC") == "ETH/BTC"
    assert to_unified_symbol("BTC/USDT") == "BTC/USDT"


def test_unrecognised_symbol_is_rejected() -> None:
    with pytest.raises(ValueError, match="Cannot split"):
        to_unified_symbol("NOTAPAIR")


# ---------------------------------------------------------------------------
# Order payloads
# ---------------------------------------------------------------------------
def make_order(direction: Direction = Direction.LONG):
    block = make_block(direction)
    plan = build_trade_plan(block, equity=10_000.0, risk_pct=1.0, reward_ratio=4.0)
    assert plan is not None
    return build_execution_order("BTC/USDT", plan, block), plan


def test_long_entry_payload_is_a_resting_limit_buy() -> None:
    order, plan = make_order(Direction.LONG)
    payload = order.entry_payload()

    assert payload["symbol"] == "BTCUSDT"
    assert payload["side"] == "BUY"
    assert payload["type"] == "LIMIT"
    assert payload["timeInForce"] == "GTC"
    assert float(payload["price"]) == pytest.approx(plan.entry)
    assert float(payload["quantity"]) == pytest.approx(plan.quantity)


def test_short_entry_payload_is_a_resting_limit_sell() -> None:
    order, _ = make_order(Direction.SHORT)
    assert order.entry_payload()["side"] == "SELL"
    assert order.exit_side == "BUY"


def test_prices_and_quantity_are_strings() -> None:
    """Float repr is how an over-precise value slips past a venue filter."""
    order, _ = make_order()
    payload = order.entry_payload()
    assert isinstance(payload["price"], str)
    assert isinstance(payload["quantity"], str)


def test_venue_precision_is_applied_when_supplied() -> None:
    block = make_block(Direction.LONG)
    plan = build_trade_plan(block)
    assert plan is not None
    order = build_execution_order(
        "BTC/USDT",
        plan,
        block,
        price_to_precision=lambda _s, v: f"{v:.2f}",
        amount_to_precision=lambda _s, v: f"{v:.3f}",
    )
    assert order.entry == "100.00"
    assert order.quantity.count(".") == 1
    assert len(order.quantity.split(".")[1]) == 3


def test_oco_brackets_the_position_on_the_correct_sides() -> None:
    order, plan = make_order(Direction.LONG)
    oco = order.oco_payload()

    assert oco["side"] == "SELL"  # closes a long
    # For a long the target is above and the stop below.
    assert float(oco["abovePrice"]) == pytest.approx(plan.take_profit)
    assert float(oco["belowPrice"]) == pytest.approx(plan.stop_loss)
    assert float(oco["belowStopPrice"]) == pytest.approx(plan.stop_loss)


def test_short_oco_inverts_the_brackets() -> None:
    order, plan = make_order(Direction.SHORT)
    oco = order.oco_payload()

    assert oco["side"] == "BUY"  # closes a short
    assert float(oco["abovePrice"]) == pytest.approx(plan.stop_loss)
    assert float(oco["belowPrice"]) == pytest.approx(plan.take_profit)


def test_serialised_order_round_trips_as_json() -> None:
    order, plan = make_order()
    parsed = json.loads(order.to_json())

    assert parsed["symbol"] == "BTCUSDT"
    assert parsed["entry_order"]["type"] == "LIMIT"
    assert parsed["exit_oco"]["side"] == "SELL"
    assert parsed["risk"]["reward_ratio"] == pytest.approx(4.0)
    assert parsed["risk"]["risk_pct_of_equity"] == pytest.approx(1.0)
    assert parsed["risk"]["risk_amount"] == pytest.approx(plan.risk_amount)


def test_entry_and_exit_quantities_match() -> None:
    """A mismatch would leave a residual position after the bracket fires."""
    order, _ = make_order()
    assert order.entry_payload()["quantity"] == order.oco_payload()["quantity"]
