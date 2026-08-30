"""Unit tests for alert formatting.

The message is the product the user sees, so its content and escaping are
pinned here. No network is involved.
"""

from __future__ import annotations

import pytest

from scanner.bot import build_execution_order_from_zone
from scanner.mtf import LtfTrigger
from scanner.notifier import (
    build_message,
    format_money,
    format_price,
    format_price_group,
    format_quantity,
    quote_prefix,
)
from scanner.smc import Direction
from tests.test_watchlist import make_zone

_BASE_MS = 1_700_000_000_000


def make_trigger(direction: Direction = Direction.LONG) -> LtfTrigger:
    return LtfTrigger(
        direction=direction,
        timeframe="15m",
        choch_timestamp=_BASE_MS + 3_600_000,
        fvg_timestamp=_BASE_MS + 4_500_000,
        fvg_pct=0.42,
        price=100.0,
    )


# ---------------------------------------------------------------------------
# Required content
# ---------------------------------------------------------------------------
def test_message_contains_every_required_field() -> None:
    message = build_message(make_zone(), trigger=make_trigger())
    for expected in (
        "BTC/USDT",
        "1h",
        "ENTRY",
        "OB zone",
        "Displacement",
        "Location",
        "Confirmation",
        "CHoCH",
        "Entry:",
        "Stop-Loss:",
        "Take-Profit",
        "Quantity:",
    ):
        assert expected in message, f"missing {expected!r}"


def test_long_and_short_are_visually_distinct() -> None:
    long_message = build_message(make_zone(Direction.LONG), trigger=make_trigger())
    short_message = build_message(
        make_zone(Direction.SHORT), trigger=make_trigger(Direction.SHORT)
    )
    assert "🟢" in long_message and "LONG" in long_message
    assert "🔴" in short_message and "SHORT" in short_message


def test_displacement_and_location_are_reported() -> None:
    """The two v3.1 filters must be visible in the alert that they let through."""
    message = build_message(make_zone(), trigger=make_trigger())
    assert "1.50% fair value gap" in message
    assert "fib 0.25" in message
    assert "discount half" in message


def test_short_reports_the_premium_half() -> None:
    message = build_message(make_zone(Direction.SHORT), trigger=make_trigger(Direction.SHORT))
    assert "premium half" in message


def test_ltf_confirmation_block_is_shown() -> None:
    message = build_message(make_zone(), trigger=make_trigger())
    assert "15m Confirmation" in message
    assert "0.42%" in message


def test_message_renders_without_a_trigger() -> None:
    """A zone can be rendered before confirmation without crashing."""
    message = build_message(make_zone(), trigger=None)
    assert "Confirmation" not in message
    assert "OB zone" in message


def test_reward_ratio_is_derived_from_the_levels() -> None:
    """Entry 100, stop 94, target 124 is exactly 1:4."""
    assert "1:4" in build_message(make_zone(), trigger=make_trigger())


def test_short_target_shows_a_negative_price_move() -> None:
    """A short's target is below entry; "+" would read as a rally."""
    message = build_message(
        make_zone(Direction.SHORT), trigger=make_trigger(Direction.SHORT)
    )
    assert "-24.00%" in message  # 100 -> 76


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def test_all_prices_share_a_decimal_count() -> None:
    """Venues strip trailing zeros per value, which renders a ragged set."""
    zone = make_zone()
    zone.stop_loss = 94.25          # 2 decimals
    zone.entry = 100.5              # 1 decimal on its own
    message = build_message(
        zone, trigger=make_trigger(), to_precision=lambda v: f"{v:.10g}"
    )
    assert "100.50" in message      # padded up to match the stop
    assert "94.25" in message
    assert "100.5<" not in message  # never the ragged one-decimal form


def test_format_price_group_uses_the_widest_precision() -> None:
    assert format_price_group([1925.91, 1898.1], ["1925.91", "1898.1"]) == [
        "1,925.91",
        "1,898.10",
    ]


def test_money_amounts_use_two_decimals() -> None:
    assert format_money(100.0) == "100.00"
    assert format_money(1_234_567.891) == "1,234,567.89"


def test_format_quantity_spans_btc_fractions_to_millions() -> None:
    assert format_quantity(0.00123456) == "0.00123456"
    assert format_quantity(19.2657) == "19.2657"
    assert format_quantity(1_250_000.0) == "1,250,000.00"


def test_usd_quotes_render_a_dollar_sign() -> None:
    assert quote_prefix("BTC/USDT") == "$"
    assert quote_prefix("ETH/BTC") == "BTC "
    assert "$" in build_message(make_zone(), trigger=make_trigger())


def test_format_price_falls_back_by_magnitude() -> None:
    assert format_price(64_000.0) == "64,000.00"
    assert format_price(0.00004821) == "0.00004821"


# ---------------------------------------------------------------------------
# Routing block
# ---------------------------------------------------------------------------
def test_execution_order_is_shown_when_supplied() -> None:
    zone = make_zone()
    order = build_execution_order_from_zone(zone)
    message = build_message(zone, trigger=make_trigger(), order=order)
    assert "Route:" in message
    assert "BTCUSDT" in message
    assert "BUY" in message


def test_order_built_from_a_zone_preserves_the_validated_levels() -> None:
    """Entry must use the structure that was validated, not a fresh price."""
    zone = make_zone()
    order = build_execution_order_from_zone(zone)
    assert float(order.entry) == pytest.approx(zone.entry)
    assert float(order.stop_loss) == pytest.approx(zone.stop_loss)
    assert float(order.take_profit) == pytest.approx(zone.take_profit)
    assert float(order.quantity) == pytest.approx(zone.quantity)
    assert order.reward_ratio == pytest.approx(4.0)


def test_short_zone_routes_as_a_sell() -> None:
    order = build_execution_order_from_zone(make_zone(Direction.SHORT))
    assert order.side == "SELL"
    assert order.exit_side == "BUY"


def test_message_renders_without_an_execution_order() -> None:
    """Order formatting failures must not suppress the alert."""
    message = build_message(make_zone(), trigger=make_trigger(), order=None)
    assert "Route:" not in message
    assert "Entry:" in message


# ---------------------------------------------------------------------------
# Escaping and limits
# ---------------------------------------------------------------------------
def test_html_metacharacters_are_escaped() -> None:
    """Unescaped '<' would make Telegram reject the message with HTTP 400."""
    message = build_message(make_zone(symbol="A<B>/USDT"), trigger=make_trigger())
    assert "A&lt;B&gt;/USDT" in message
    assert "<b>" in message  # our own markup survives


def test_message_is_within_the_telegram_length_limit() -> None:
    assert len(build_message(make_zone(), trigger=make_trigger())) <= 4096
