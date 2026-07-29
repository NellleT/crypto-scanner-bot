"""Unit tests for alert formatting.

The message is the product the user actually sees, so its content and its
escaping are pinned here. No network is involved.
"""

from __future__ import annotations

import pytest

from scanner.notifier import (
    align_decimals,
    build_message,
    format_price,
    format_price_group,
    format_volume,
    humanize_price,
    quote_prefix,
)
from scanner.patterns import Candle, PatternType
from scanner.risk import RiskPlan, TakeProfit
from scanner.strategy import SignalDirection, TradeSignal

PREV = Candle(1_753_480_800_000, 118_500.0, 118_900.0, 117_800.0, 117_950.0, 1_200.0)
CURR = Candle(1_753_481_700_000, 117_900.0, 119_400.0, 117_850.0, 118_700.0, 2_500.0)


def make_plan(
    *,
    is_long: bool = True,
    entry: float = 118_700.0,
    stop_loss: float = 117_000.0,
    ratios: tuple[float, ...] = (2.0, 3.0),
) -> RiskPlan:
    risk = abs(entry - stop_loss)
    prices = [entry + risk * r if is_long else entry - risk * r for r in ratios]
    return RiskPlan(
        is_long=is_long,
        entry=entry,
        stop_loss=stop_loss,
        structural_level=stop_loss / 0.999 if is_long else stop_loss / 1.001,
        lookback=10,
        buffer_pct=0.1,
        take_profits=tuple(
            TakeProfit(ratio=r, price=p) for r, p in zip(ratios, prices)
        ),
    )


def make_signal(
    direction: SignalDirection = SignalDirection.LONG,
    *,
    price: float = 118_700.0,
    trend_sma: float = 112_000.0,
    volume: float = 2_500.0,
    volume_sma: float = 1_000.0,
    previous_volume: float = 1_200.0,
    risk: RiskPlan | None = None,
) -> TradeSignal:
    pattern = (
        PatternType.BULLISH_ENGULFING
        if direction is SignalDirection.LONG
        else PatternType.BEARISH_ENGULFING
    )
    is_long = direction is SignalDirection.LONG
    plan = risk or make_plan(
        is_long=is_long,
        entry=price,
        stop_loss=price * (0.98 if is_long else 1.02),
    )
    return TradeSignal(
        symbol="BTC/USDT",
        timeframe="4h",
        direction=direction,
        pattern=pattern,
        price=price,
        trend_sma=trend_sma,
        trend_sma_period=200,
        volume=volume,
        volume_sma=volume_sma,
        volume_sma_period=20,
        previous_volume=previous_volume,
        engulf_ratio=1.45,
        candle=CURR,
        risk=plan,
    )


# ---------------------------------------------------------------------------
# Required content
# ---------------------------------------------------------------------------
def test_message_contains_every_required_field() -> None:
    message = build_message(make_signal())
    for expected in (
        "BTC/USDT",
        "4h",
        "LONG",
        "SMA 200",
        "Volume",
        "Engulf ratio",
        "VSA",
        "Risk Management",
        "Entry:",
        "Stop-Loss",
        "Take-Profit 1",
        "Take-Profit 2",
    ):
        assert expected in message, f"missing {expected!r}"


# ---------------------------------------------------------------------------
# Risk block
# ---------------------------------------------------------------------------
def test_risk_block_reports_the_structural_anchor_and_ratios() -> None:
    message = build_message(make_signal())
    assert "10-bar low" in message
    assert "(1:2)" in message
    assert "(1:3)" in message
    assert "Risk:" in message


def test_short_risk_block_anchors_to_the_high() -> None:
    message = build_message(make_signal(SignalDirection.SHORT, price=100.0))
    assert "10-bar high" in message


def test_risk_prices_all_share_one_decimal_count() -> None:
    """Entry, stop and both targets are compared side by side."""
    plan = make_plan(entry=1925.91, stop_loss=1898.10, ratios=(2.0, 3.0))
    message = build_message(
        make_signal(price=1925.91, risk=plan),
        to_precision=lambda value: f"{value:.10g}",  # strips trailing zeros
    )
    assert "1,925.91" in message
    assert "1,898.10" in message  # not "1,898.1"
    assert "1,981.53" in message  # 1925.91 + 2 * 27.81
    assert "2,009.34" in message  # 1925.91 + 3 * 27.81


def test_risk_percentages_are_reported() -> None:
    plan = make_plan(entry=100.0, stop_loss=98.0, ratios=(2.0,))
    message = build_message(make_signal(price=100.0, risk=plan))
    assert "Risk: 2.00%" in message
    assert "+4.00%" in message  # 2R on 2% risk, price moves up


def test_short_target_shows_a_negative_price_move() -> None:
    """A short's target is below entry; "+5.74%" would read as a rally."""
    plan = make_plan(is_long=False, entry=100.0, stop_loss=102.0, ratios=(2.0,))
    message = build_message(make_signal(SignalDirection.SHORT, price=100.0, risk=plan))
    assert "-4.00%" in message
    assert "+4.00%" not in message
    # The gain itself is still a positive magnitude for sizing purposes.
    assert plan.reward_pct(plan.take_profits[0]) == pytest.approx(4.0)
    assert plan.price_move_pct(plan.take_profits[0]) == pytest.approx(-4.0)


def test_usd_quotes_render_a_dollar_sign() -> None:
    assert quote_prefix("BTC/USDT") == "$"
    assert quote_prefix("BTC/USDC") == "$"
    assert "$" in build_message(make_signal())


def test_non_usd_quote_does_not_claim_dollars() -> None:
    """An ETH/BTC price is not denominated in dollars."""
    assert quote_prefix("ETH/BTC") == "BTC "
    assert quote_prefix("SOL/EUR") == "EUR "


def test_format_price_group_uses_the_widest_precision() -> None:
    assert format_price_group([1925.91, 1898.1], ["1925.91", "1898.1"]) == [
        "1,925.91",
        "1,898.10",
    ]
    assert format_price_group([1.5, 2.0], None) == ["1.5000", "2.0000"]


def test_long_and_short_are_visually_distinct() -> None:
    long_message = build_message(make_signal(SignalDirection.LONG))
    short_message = build_message(
        make_signal(SignalDirection.SHORT, price=100_000.0, trend_sma=112_000.0)
    )
    assert "🟢" in long_message and "LONG SIGNAL" in long_message
    assert "🔴" in short_message and "SHORT SIGNAL" in short_message


def test_volume_is_reported_relative_to_its_average() -> None:
    message = build_message(make_signal(volume=2_500.0, volume_sma=1_000.0))
    assert "2.50x" in message


def test_vsa_expansion_is_reported() -> None:
    """The alert shows the follow-through it was accepted for."""
    message = build_message(
        make_signal(volume=2_500.0, volume_sma=1_000.0, previous_volume=1_000.0)
    )
    assert "VSA:" in message
    assert "2.50x the engulfed candle" in message


def test_distance_from_sma_is_reported_with_direction() -> None:
    above = build_message(make_signal(price=110.0, trend_sma=100.0))
    assert "10.00% above" in above

    below = build_message(
        make_signal(SignalDirection.SHORT, price=90.0, trend_sma=100.0)
    )
    assert "10.00% below" in below


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------
def test_price_and_sma_use_the_same_decimal_count() -> None:
    """price_to_precision strips trailing zeros; a ragged pair looks like a bug."""
    message = build_message(
        make_signal(price=58.11, trend_sma=56.0),
        price_text="58.11",
        sma_text="56",  # what Binance returns for 56.00
    )
    assert "58.11" in message
    assert "56.00" in message


def test_align_decimals_keeps_extra_precision_when_needed() -> None:
    assert align_decimals(0.5, "0.50000000", "1.23") == "0.50000000"
    assert align_decimals(56.0, "56", "58.11") == "56.00"
    assert align_decimals(1234.0, "1234", "1.5") == "1,234.0"


def test_humanize_price_adds_grouping_at_venue_precision() -> None:
    assert humanize_price(118_700.0, "118700") == "118,700"
    assert humanize_price(0.00004821, "0.00004821") == "0.00004821"
    assert humanize_price(118_700.0, None) == format_price(118_700.0)


def test_format_volume_is_compact() -> None:
    assert format_volume(68_390.0) == "68.39K"
    assert format_volume(2_450_000.0) == "2.45M"
    assert format_volume(3_100_000_000.0) == "3.10B"
    assert format_volume(942.5) == "942.50"


def test_format_volume_handles_meme_coin_scale() -> None:
    """Sub-cent pairs trade in trillions of base units, not billions."""
    assert format_volume(9.1e12) == "9.10T"


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------
def test_html_metacharacters_are_escaped() -> None:
    """Unescaped '<' would make Telegram reject the message with HTTP 400."""
    signal = make_signal()
    hostile = TradeSignal(**{**{f: getattr(signal, f) for f in signal.__slots__}, "symbol": "A<B>/USDT"})
    message = build_message(hostile)
    assert "A&lt;B&gt;/USDT" in message
    assert "<b>" in message  # our own markup survives


def test_message_is_within_the_telegram_length_limit() -> None:
    assert len(build_message(make_signal())) <= 4096
