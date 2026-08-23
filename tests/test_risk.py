"""Unit tests for order-block trade construction and position sizing.

Values are chosen so entry, stop, target and quantity can be checked by hand.
"""

from __future__ import annotations

import pytest

from scanner.candles import Candle
from scanner.risk import (
    DEFAULT_REWARD_RATIO,
    DEFAULT_RISK_PER_TRADE_PCT,
    DEFAULT_STOP_BUFFER_PCT,
    build_trade_plan,
    position_size,
)
from scanner.smc import Direction, FairValueGap, OrderBlock  # noqa: F401

_HOUR_MS = 3_600_000


def candle(open_: float, high: float, low: float, close: float, index: int = 0) -> Candle:
    return Candle(
        timestamp=1_700_000_000_000 + index * _HOUR_MS,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100.0,
    )


def make_block(direction: Direction = Direction.LONG) -> OrderBlock:
    """A block spanning 95-100, so the zone is exactly 5 wide."""
    if direction is Direction.LONG:
        ob = candle(99.0, 100.0, 95.0, 96.0)          # bearish
        impulse = candle(96.0, 106.0, 95.5, 105.0, 1)  # bullish
        confirm = candle(105.0, 108.0, 104.0, 107.0, 2)
        fvg = FairValueGap(is_bullish=True, bottom=100.0, top=104.0)
    else:
        ob = candle(96.0, 100.0, 95.0, 99.0, 0)        # bullish
        impulse = candle(99.0, 99.5, 89.0, 90.0, 1)    # bearish
        confirm = candle(90.0, 91.0, 87.0, 88.0, 2)
        fvg = FairValueGap(is_bullish=False, bottom=91.0, top=95.0)
    return OrderBlock(
        direction=direction, candle=ob, impulse=impulse, confirmation=confirm, fvg=fvg
    )


# ---------------------------------------------------------------------------
# Long plans
# ---------------------------------------------------------------------------
def test_long_plan_is_computed_by_hand() -> None:
    plan = build_trade_plan(
        make_block(Direction.LONG),
        buffer_pct=0.2,
        reward_ratio=4.0,
        equity=10_000.0,
        risk_pct=1.0,
    )
    assert plan is not None

    assert plan.entry == pytest.approx(100.0)          # proximal = OB high
    assert plan.stop_loss == pytest.approx(94.81)      # 95 * (1 - 0.002)
    assert plan.risk_per_unit == pytest.approx(5.19)   # 100 - 94.81
    assert plan.take_profit == pytest.approx(120.76)   # 100 + 4 * 5.19
    assert plan.risk_amount == pytest.approx(100.0)    # 1% of 10k
    assert plan.quantity == pytest.approx(100.0 / 5.19)
    assert plan.reward_amount == pytest.approx(400.0)  # 4R


def test_long_stop_sits_below_the_block_not_inside_it() -> None:
    """"Distal + buffer" must push away from the zone, not up into it.

    A long stop moved up toward the entry would sit closer to the sweep it is
    meant to survive.
    """
    block = make_block(Direction.LONG)
    plan = build_trade_plan(block, buffer_pct=0.2)
    assert plan is not None
    assert plan.stop_loss < block.distal
    assert plan.stop_loss < block.proximal
    assert plan.entry == block.proximal


# ---------------------------------------------------------------------------
# Short plans
# ---------------------------------------------------------------------------
def test_short_plan_is_computed_by_hand() -> None:
    plan = build_trade_plan(
        make_block(Direction.SHORT),
        buffer_pct=0.2,
        reward_ratio=4.0,
        equity=10_000.0,
        risk_pct=1.0,
    )
    assert plan is not None

    assert plan.entry == pytest.approx(95.0)           # proximal = OB low
    assert plan.stop_loss == pytest.approx(100.2)      # 100 * (1 + 0.002)
    assert plan.risk_per_unit == pytest.approx(5.2)
    assert plan.take_profit == pytest.approx(74.2)     # 95 - 4 * 5.2
    assert plan.take_profit_move_pct < 0               # price must fall


def test_short_stop_sits_above_the_block() -> None:
    block = make_block(Direction.SHORT)
    plan = build_trade_plan(block, buffer_pct=0.2)
    assert plan is not None
    assert plan.stop_loss > block.distal
    assert plan.stop_loss > block.proximal


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------
def test_position_size_risks_exactly_the_configured_fraction() -> None:
    quantity = position_size(equity=10_000.0, risk_pct=1.0, risk_per_unit=5.0)
    assert quantity == pytest.approx(20.0)
    # Being stopped out loses exactly 1% of equity.
    assert quantity * 5.0 == pytest.approx(100.0)


def test_position_size_scales_inversely_with_stop_distance() -> None:
    """A wider stop must buy less, so the loss stays constant."""
    tight = position_size(equity=10_000.0, risk_pct=1.0, risk_per_unit=1.0)
    wide = position_size(equity=10_000.0, risk_pct=1.0, risk_per_unit=10.0)
    assert tight == pytest.approx(wide * 10)
    assert tight * 1.0 == pytest.approx(wide * 10.0)


def test_position_size_rejects_degenerate_inputs() -> None:
    assert position_size(equity=0.0, risk_pct=1.0, risk_per_unit=5.0) == 0.0
    assert position_size(equity=10_000.0, risk_pct=0.0, risk_per_unit=5.0) == 0.0
    assert position_size(equity=10_000.0, risk_pct=1.0, risk_per_unit=0.0) == 0.0


def test_loss_at_stop_equals_the_risk_budget_for_both_directions() -> None:
    for direction in (Direction.LONG, Direction.SHORT):
        plan = build_trade_plan(make_block(direction), equity=25_000.0, risk_pct=1.0)
        assert plan is not None
        loss = plan.quantity * plan.risk_per_unit
        assert loss == pytest.approx(plan.risk_amount)
        assert loss == pytest.approx(250.0)


def test_notional_can_exceed_equity_and_is_reported() -> None:
    """A tight stop implies leverage; sizing ignores it but must not hide it.

    Notional exceeds equity exactly when the stop is nearer than the equity risk
    budget: a 0.1%-wide stop funding a 1% risk needs 10x the account.
    """
    tight = OrderBlock(
        direction=Direction.LONG,
        candle=candle(99.95, 100.0, 99.9, 99.92),   # zone only 0.1% wide
        impulse=candle(99.92, 106.0, 99.9, 105.0, 1),
        confirmation=candle(105.0, 108.0, 104.0, 107.0, 2),
        fvg=FairValueGap(is_bullish=True, bottom=100.0, top=104.0),
    )
    plan = build_trade_plan(tight, buffer_pct=0.0, equity=10_000.0, risk_pct=1.0)
    assert plan is not None

    assert plan.risk_pct_of_entry == pytest.approx(0.1)
    assert plan.notional > plan.equity
    assert plan.leverage_required == pytest.approx(10.0, rel=1e-3)
    # The budgeted loss is still exactly 1% — only the notional is large.
    assert plan.quantity * plan.risk_per_unit == pytest.approx(100.0)


def test_wide_stop_needs_less_than_the_account() -> None:
    """The mirror case: a 5%-wide stop funds 1% risk with a fifth of equity."""
    plan = build_trade_plan(
        make_block(Direction.LONG), buffer_pct=0.0, equity=10_000.0, risk_pct=1.0
    )
    assert plan is not None
    assert plan.risk_pct_of_entry == pytest.approx(5.0)
    assert plan.leverage_required == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Reward geometry
# ---------------------------------------------------------------------------
def test_target_is_exactly_the_reward_multiple() -> None:
    for direction in (Direction.LONG, Direction.SHORT):
        for ratio in (1.0, 2.5, 4.0, 10.0):
            plan = build_trade_plan(make_block(direction), reward_ratio=ratio)
            assert plan is not None
            reward = abs(plan.take_profit - plan.entry)
            assert reward / plan.risk_per_unit == pytest.approx(ratio)


def test_buffer_widens_the_stop_and_shrinks_the_position() -> None:
    tight = build_trade_plan(make_block(Direction.LONG), buffer_pct=0.0)
    wide = build_trade_plan(make_block(Direction.LONG), buffer_pct=2.0)
    assert tight is not None and wide is not None
    assert wide.risk_per_unit > tight.risk_per_unit
    assert wide.quantity < tight.quantity
    # The budgeted loss is unchanged; only the size absorbs the wider stop.
    assert wide.risk_amount == pytest.approx(tight.risk_amount)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------
def test_short_target_below_zero_is_rejected() -> None:
    """Risk above 1/ratio of entry makes a short target negative."""
    ob = candle(96.0, 100.0, 20.0, 99.0)
    impulse = candle(99.0, 99.5, 15.0, 16.0, 1)
    confirm = candle(16.0, 18.0, 14.0, 15.0, 2)
    block = OrderBlock(
        direction=Direction.SHORT,
        candle=ob,
        impulse=impulse,
        confirmation=confirm,
        fvg=FairValueGap(is_bullish=False, bottom=18.0, top=20.0),
    )
    assert build_trade_plan(block, reward_ratio=4.0) is None


def test_zero_height_block_is_rejected() -> None:
    """A block whose edges coincide gives no stop distance."""
    flat = candle(50.0, 50.0, 50.0, 50.0)
    block = OrderBlock(
        direction=Direction.LONG,
        candle=flat,
        impulse=candle(50.0, 60.0, 50.0, 59.0, 1),
        confirmation=candle(59.0, 62.0, 55.0, 61.0, 2),
        fvg=FairValueGap(is_bullish=True, bottom=50.0, top=55.0),
    )
    assert build_trade_plan(block, buffer_pct=0.0) is None


def test_defaults_match_the_documented_values() -> None:
    assert DEFAULT_STOP_BUFFER_PCT == 0.2
    assert DEFAULT_REWARD_RATIO == 4.0
    assert DEFAULT_RISK_PER_TRADE_PCT == 1.0
