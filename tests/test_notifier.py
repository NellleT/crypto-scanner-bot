"""Unit tests for alert formatting.

The message is the product the user sees, so its content and escaping are
pinned here. No network is involved.
"""

from __future__ import annotations

import pytest

from scanner.execution import build_execution_order
from scanner.notifier import (
    build_message,
    format_money,
    format_price,
    format_price_group,
    format_quantity,
    quote_prefix,
)
from scanner.risk import build_trade_plan
from scanner.smc import Direction
from scanner.strategy import TradeSignal
from tests.test_risk import make_block


def make_signal(
    direction: Direction = Direction.LONG,
    *,
    symbol: str = "BTC/USDT",
    equity: float = 10_000.0,
) -> TradeSignal:
    block = make_block(direction)
    plan = build_trade_plan(block, equity=equity, risk_pct=1.0, reward_ratio=4.0)
    assert plan is not None
    return TradeSignal(symbol=symbol, timeframe="1h", block=block, plan=plan)


# ---------------------------------------------------------------------------
# Required content
# ---------------------------------------------------------------------------
def test_message_contains_every_required_field() -> None:
    message = build_message(make_signal())
    for expected in (
        "BTC/USDT",
        "1h",
        "ORDER BLOCK",
        "OB zone",
        "Fair Value Gap",
        "Pending Limit Order",
        "Entry:",
        "Stop-Loss:",
        "Take-Profit",
        "Position Sizing",
        "Quantity:",
        "Risk:",
    ):
        assert expected in message, f"missing {expected!r}"


def test_long_and_short_are_visually_distinct() -> None:
    long_message = build_message(make_signal(Direction.LONG))
    short_message = build_message(make_signal(Direction.SHORT))
    assert "🟢" in long_message and "LONG" in long_message
    assert "🔴" in short_message and "SHORT" in short_message


def test_reward_ratio_is_labelled() -> None:
    assert "1:4" in build_message(make_signal())


def test_risk_budget_is_reported_in_currency() -> None:
    """1% of 10,000 is 100 — the number that decides position size."""
    message = build_message(make_signal(equity=10_000.0))
    assert "1% of" in message
    assert "$10,000.00" in message
    assert "$100.00" in message
    assert "$400.00" in message  # 4R reward


def test_money_amounts_use_two_decimals_not_price_precision() -> None:
    """Cash is money; "$100.0000" reads like a bug.

    Scoped to the sizing lines: an instrument *price* of 100 legitimately
    renders as 100.0000 under the magnitude fallback, and the two must not be
    conflated.
    """
    assert format_money(100.0) == "100.00"
    assert format_money(1_234_567.891) == "1,234,567.89"

    message = build_message(make_signal(equity=10_000.0))
    sizing = [line for line in message.splitlines() if "Risk:" in line or "Reward" in line]
    assert sizing, "sizing lines missing from the alert"
    for line in sizing:
        assert "0000</code>" not in line, f"price precision leaked into money: {line}"


def test_short_target_shows_a_negative_price_move() -> None:
    """A short's target is below entry; "+" would read as a rally."""
    message = build_message(make_signal(Direction.SHORT))
    assert "-" in message
    plan = make_signal(Direction.SHORT).plan
    assert plan.take_profit_move_pct < 0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def test_all_plan_prices_share_a_decimal_count() -> None:
    message = build_message(make_signal(), to_precision=lambda v: f"{v:.10g}")
    # entry 100, stop 94.81, target 120.76 -> all rendered with two decimals.
    assert "100.00" in message
    assert "94.81" in message
    assert "120.76" in message


def test_format_price_group_uses_the_widest_precision() -> None:
    assert format_price_group([1925.91, 1898.1], ["1925.91", "1898.1"]) == [
        "1,925.91",
        "1,898.10",
    ]


def test_format_quantity_spans_btc_fractions_to_meme_coin_millions() -> None:
    assert format_quantity(0.00123456) == "0.00123456"
    assert format_quantity(19.2657) == "19.2657"
    assert format_quantity(1_250_000.0) == "1,250,000.00"


def test_usd_quotes_render_a_dollar_sign() -> None:
    assert quote_prefix("BTC/USDT") == "$"
    assert quote_prefix("ETH/BTC") == "BTC "
    assert "$" in build_message(make_signal())


def test_format_price_falls_back_by_magnitude() -> None:
    assert format_price(64_000.0) == "64,000.00"
    assert format_price(0.00004821) == "0.00004821"


# ---------------------------------------------------------------------------
# Routing block
# ---------------------------------------------------------------------------
def test_execution_order_is_shown_when_supplied() -> None:
    signal = make_signal()
    order = build_execution_order(signal.symbol, signal.plan, signal.block)
    message = build_message(signal, order=order)
    assert "Route:" in message
    assert "BTCUSDT" in message
    assert "BUY" in message


def test_message_renders_without_an_execution_order() -> None:
    """Order formatting failures must not suppress the alert."""
    message = build_message(make_signal(), order=None)
    assert "Route:" not in message
    assert "Pending Limit Order" in message


# ---------------------------------------------------------------------------
# Escaping and limits
# ---------------------------------------------------------------------------
def test_html_metacharacters_are_escaped() -> None:
    """Unescaped '<' would make Telegram reject the message with HTTP 400."""
    message = build_message(make_signal(symbol="A<B>/USDT"))
    assert "A&lt;B&gt;/USDT" in message
    assert "<b>" in message  # our own markup survives


def test_message_is_within_the_telegram_length_limit() -> None:
    assert len(build_message(make_signal())) <= 4096
