"""Unit tests for the HTF filter chain.

Each stage is exercised in isolation by holding the others satisfied, so a
failure points at exactly one condition.
"""

from __future__ import annotations

import pytest

from scanner.smc import Direction, RangeZone
from scanner.strategy import FilterStage, OrderBlockStrategy
from tests.test_smc import (
    BEAR_CONFIRM,
    BEAR_IMPULSE,
    BEAR_OB,
    BULL_CONFIRM,
    BULL_CONFIRM_NO_FVG,
    BULL_CONFIRM_TINY_FVG,
    BULL_IMPULSE,
    BULL_OB,
    build_spatial_frame,
    make_df,
)


@pytest.fixture
def strategy() -> OrderBlockStrategy:
    return OrderBlockStrategy(
        min_body_ratio=0.05,
        min_fvg_pct=0.30,
        range_lookback=50,
        require_extreme=True,
        stop_buffer_pct=0.2,
        max_stop_pct=0.0,   # off: these fixtures use a deliberately thick zone
        reward_ratio=4.0,
        account_equity=10_000.0,
        risk_per_trade_pct=1.0,
    )


def long_frame(**kwargs):
    """A bullish structure sitting in the discount half of a 0-400 range."""
    return build_spatial_frame(
        [BULL_OB, BULL_IMPULSE, kwargs.get("confirm", BULL_CONFIRM)],
        range_low=0.0,
        range_high=400.0,
    )


# ---------------------------------------------------------------------------
# Accepted setups
# ---------------------------------------------------------------------------
def test_long_reaching_the_watchlist(strategy: OrderBlockStrategy) -> None:
    result = strategy.evaluate(long_frame(), "BTC/USDT", "1h")

    assert result.matched, result.reason
    assert result.stage is FilterStage.WATCHLIST
    signal = result.signal
    assert signal is not None
    assert signal.direction is Direction.LONG
    assert signal.block.zone is RangeZone.DISCOUNT
    assert signal.plan.entry == pytest.approx(signal.block.proximal)
    assert signal.plan.stop_loss < signal.block.distal
    assert signal.plan.quantity > 0


def test_short_reaching_the_watchlist(strategy: OrderBlockStrategy) -> None:
    frame = build_spatial_frame(
        [BEAR_OB, BEAR_IMPULSE, BEAR_CONFIRM], range_low=50.0, range_high=101.0
    )
    result = strategy.evaluate(frame, "ETH/USDT", "1h")

    assert result.matched, result.reason
    signal = result.signal
    assert signal is not None
    assert signal.direction is Direction.SHORT
    assert signal.block.zone is RangeZone.PREMIUM
    assert signal.plan.stop_loss > signal.block.distal


def test_the_terminal_stage_is_the_watchlist_not_an_order(
    strategy: OrderBlockStrategy,
) -> None:
    """v3.1 does not enter blind — acceptance means "watch", not "buy"."""
    result = strategy.evaluate(long_frame(), "BTC/USDT", "1h")
    assert result.stage is FilterStage.WATCHLIST
    assert result.stage is max(FilterStage, key=lambda s: s.depth)


# ---------------------------------------------------------------------------
# Stage-by-stage rejection
# ---------------------------------------------------------------------------
def test_no_structure_stops_at_order_block(strategy: OrderBlockStrategy) -> None:
    continuation = (95.0, 100.0, 94.0, 99.0)  # bullish before a bullish impulse
    frame = build_spatial_frame(
        [continuation, BULL_IMPULSE, BULL_CONFIRM], range_low=0.0, range_high=400.0
    )
    result = strategy.evaluate(frame, "BTC/USDT", "1h")
    assert not result.matched
    assert result.stage is FilterStage.ORDER_BLOCK
    assert not result.stage.reached_order_block


def test_missing_gap_stops_at_fvg(strategy: OrderBlockStrategy) -> None:
    result = strategy.evaluate(long_frame(confirm=BULL_CONFIRM_NO_FVG), "BTC/USDT", "1h")
    assert not result.matched
    assert result.stage is FilterStage.FVG
    assert result.stage.reached_order_block
    assert not result.stage.reached_fvg


def test_narrow_gap_stops_at_displacement(strategy: OrderBlockStrategy) -> None:
    """A real gap below 0.30% is noise, and is reported separately from no gap."""
    result = strategy.evaluate(
        long_frame(confirm=BULL_CONFIRM_TINY_FVG), "BTC/USDT", "1h"
    )
    assert not result.matched
    assert result.stage is FilterStage.DISPLACEMENT
    assert result.stage.reached_fvg          # a gap did exist
    assert not result.stage.reached_displacement


def test_wrong_half_of_the_range_stops_at_premium_discount(
    strategy: OrderBlockStrategy,
) -> None:
    frame = build_spatial_frame(
        [BULL_OB, BULL_IMPULSE, BULL_CONFIRM], range_low=80.0, range_high=108.0
    )
    result = strategy.evaluate(frame, "BTC/USDT", "1h")
    assert not result.matched
    assert result.stage is FilterStage.PREMIUM_DISCOUNT
    assert result.stage.reached_displacement  # displacement was fine
    assert not result.stage.reached_spatial


def test_unsizeable_structure_stops_at_risk() -> None:
    """A short whose stop exceeds a quarter of entry makes a 1:4 target negative.

    The block spans 100-130, so the stop is 30% of entry — but it still sits in
    the premium half, so the rejection can only come from sizing.
    """
    ob = (105.0, 130.0, 100.0, 125.0)   # bullish, body 20 of range 30
    impulse = (125.0, 126.0, 95.0, 96.0)  # bearish displacement
    confirm = (96.0, 98.0, 60.0, 62.0)  # high 98 < ob low 100 -> 2% FVG
    strategy = OrderBlockStrategy(reward_ratio=4.0, max_stop_pct=0.0)
    result = strategy.evaluate(make_df([ob, impulse, confirm]), "X/USDT", "1h")

    assert not result.matched
    assert result.stage is FilterStage.RISK
    assert result.stage.reached_spatial


def test_warmup_when_history_is_too_short(strategy: OrderBlockStrategy) -> None:
    result = strategy.evaluate(make_df([BULL_IMPULSE, BULL_CONFIRM]), "BTC/USDT", "1h")
    assert not result.matched
    assert result.stage is FilterStage.WARMUP


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_required_candles_covers_the_dealing_range() -> None:
    """Premium/discount is meaningless until the full range window is available."""
    assert OrderBlockStrategy(range_lookback=50).required_candles == 50
    assert OrderBlockStrategy(range_lookback=2).required_candles == 3  # structure floor


def test_spatial_filter_can_be_relaxed() -> None:
    frame = build_spatial_frame(
        [BULL_OB, BULL_IMPULSE, BULL_CONFIRM], range_low=80.0, range_high=108.0
    )
    assert not OrderBlockStrategy(max_stop_pct=0.0).evaluate(frame, "X/USDT", "1h").matched
    relaxed = OrderBlockStrategy(require_extreme=False, max_stop_pct=0.0)
    assert relaxed.evaluate(frame, "X/USDT", "1h").matched


def test_displacement_threshold_flows_through_from_config() -> None:
    frame = long_frame(confirm=BULL_CONFIRM_TINY_FVG)
    assert OrderBlockStrategy(max_stop_pct=0.0).evaluate(frame, "X/USDT", "1h").stage is (
        FilterStage.DISPLACEMENT
    )
    lenient = OrderBlockStrategy(min_fvg_pct=0.05, max_stop_pct=0.0)
    assert lenient.evaluate(frame, "X/USDT", "1h").matched


# ---------------------------------------------------------------------------
# Stage ordering
# ---------------------------------------------------------------------------
def test_stage_depth_follows_declaration_order() -> None:
    order = [
        FilterStage.WARMUP,
        FilterStage.ORDER_BLOCK,
        FilterStage.FVG,
        FilterStage.DISPLACEMENT,
        FilterStage.PREMIUM_DISCOUNT,
        FilterStage.STOP_WIDTH,
        FilterStage.RISK,
        FilterStage.WATCHLIST,
    ]
    assert list(FilterStage) == order
    assert [s.depth for s in order] == sorted(s.depth for s in order)


def test_passed_is_strict_about_the_stage_it_stopped_at() -> None:
    for stage in FilterStage:
        assert not stage.passed(stage)
    assert FilterStage.RISK.passed(FilterStage.PREMIUM_DISCOUNT)
    assert not FilterStage.FVG.passed(FilterStage.DISPLACEMENT)


def test_dedup_key_identifies_the_block(strategy: OrderBlockStrategy) -> None:
    frame = long_frame()
    signal = strategy.evaluate(frame, "BTC/USDT", "1h").signal
    assert signal is not None
    assert signal.dedup_key == (
        "BTC/USDT",
        "1h",
        int(frame["timestamp"].iloc[-3]),
        "LONG",
    )


# ---------------------------------------------------------------------------
# Stop-width filter
#
# Fixed-fraction sizing holds the LOSS constant however wide the stop is, so
# this filter is not about account risk. It rejects zones so thick they have
# not located anything precise enough to trade.
# ---------------------------------------------------------------------------
def test_stop_wider_than_the_limit_is_rejected() -> None:
    """The canonical zone is 95-100 on a ~100 price: a 5.19% stop."""
    strategy = OrderBlockStrategy(max_stop_pct=3.5)
    result = strategy.evaluate(long_frame(), "BTC/USDT", "1h")

    assert not result.matched
    assert result.stage is FilterStage.STOP_WIDTH
    assert "wider than the 3.5% limit" in result.reason
    # Everything structural passed; only the width refused it.
    assert result.stage.reached_spatial
    assert not result.stage.reached_stop_width


def test_a_tight_zone_passes_the_width_filter() -> None:
    """Same structure and displacement, but a zone only ~0.6% thick."""
    tight_ob = (99.7, 100.0, 99.4, 99.5)      # bearish, 0.6 wide
    tight_impulse = (99.5, 106.0, 99.45, 105.0)
    tight_confirm = (105.0, 108.0, 104.0, 107.0)
    frame = build_spatial_frame(
        [tight_ob, tight_impulse, tight_confirm], range_low=0.0, range_high=400.0
    )
    result = OrderBlockStrategy(max_stop_pct=3.5).evaluate(frame, "BTC/USDT", "1h")

    assert result.matched, result.reason
    assert result.signal is not None
    assert result.signal.plan.risk_pct_of_entry < 3.5


def test_the_width_limit_is_configurable() -> None:
    frame = long_frame()
    assert OrderBlockStrategy(max_stop_pct=3.5).evaluate(frame, "X/USDT", "1h").stage is (
        FilterStage.STOP_WIDTH
    )
    assert OrderBlockStrategy(max_stop_pct=6.0).evaluate(frame, "X/USDT", "1h").matched


def test_zero_disables_the_width_filter() -> None:
    """0.0 means "no limit", not "reject everything"."""
    assert OrderBlockStrategy(max_stop_pct=0.0).evaluate(long_frame(), "X/USDT", "1h").matched


def test_the_width_filter_holds_the_loss_constant_either_way() -> None:
    """Sizing already caps the loss; the filter is about zone precision.

    A 5.19% stop still risks exactly 1% of equity — it just implies a position
    small enough to be pointless.
    """
    permissive = OrderBlockStrategy(max_stop_pct=0.0, account_equity=10_000.0)
    signal = permissive.evaluate(long_frame(), "X/USDT", "1h").signal
    assert signal is not None
    assert signal.plan.quantity * signal.plan.risk_per_unit == pytest.approx(100.0)
    assert signal.plan.risk_pct_of_entry > 3.5
