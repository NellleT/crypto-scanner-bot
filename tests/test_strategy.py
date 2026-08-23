"""Unit tests for the v3.0 filter chain.

Each stage is exercised in isolation by holding the others satisfied, so a
failure points at exactly one condition.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scanner.smc import Direction
from scanner.strategy import FilterStage, OrderBlockStrategy
from tests.test_smc import (
    BEAR_CONFIRM,
    BEAR_CONFIRM_NO_FVG,
    BEAR_IMPULSE,
    BEAR_OB,
    BULL_CONFIRM,
    BULL_CONFIRM_NO_FVG,
    BULL_IMPULSE,
    BULL_OB,
    make_df,
)


@pytest.fixture
def default_strategy() -> OrderBlockStrategy:
    return OrderBlockStrategy(
        min_body_ratio=0.05,
        stop_buffer_pct=0.2,
        reward_ratio=4.0,
        account_equity=10_000.0,
        risk_per_trade_pct=1.0,
    )


# ---------------------------------------------------------------------------
# Confirmed signals
# ---------------------------------------------------------------------------
def test_long_signal_when_block_and_fvg_agree(
    default_strategy: OrderBlockStrategy,
) -> None:
    df = make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM])
    result = default_strategy.evaluate(df, "BTC/USDT", "1h")

    assert result.matched, result.reason
    signal = result.signal
    assert signal is not None
    assert signal.direction is Direction.LONG
    assert signal.plan.entry == pytest.approx(signal.block.proximal)
    assert signal.plan.stop_loss < signal.block.distal
    assert signal.plan.take_profit > signal.plan.entry
    assert signal.plan.quantity > 0
    assert result.stage is FilterStage.CONFIRMED


def test_short_signal_when_block_and_fvg_agree(
    default_strategy: OrderBlockStrategy,
) -> None:
    df = make_df([BEAR_OB, BEAR_IMPULSE, BEAR_CONFIRM])
    result = default_strategy.evaluate(df, "ETH/USDT", "1h")

    assert result.matched, result.reason
    signal = result.signal
    assert signal is not None
    assert signal.direction is Direction.SHORT
    assert signal.plan.stop_loss > signal.block.distal
    assert signal.plan.take_profit < signal.plan.entry


# ---------------------------------------------------------------------------
# FVG rejection — the rule this version exists for
# ---------------------------------------------------------------------------
def test_long_rejected_without_a_validating_fvg(
    default_strategy: OrderBlockStrategy,
) -> None:
    """A structurally valid order block with no gap must produce nothing."""
    df = make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM_NO_FVG])
    result = default_strategy.evaluate(df, "BTC/USDT", "1h")

    assert not result.matched
    assert result.stage is FilterStage.FVG
    assert "no fair value gap" in result.reason
    # It got past the structural check — only the gap was missing.
    assert result.stage.reached_order_block
    assert not result.stage.reached_fvg


def test_short_rejected_without_a_validating_fvg(
    default_strategy: OrderBlockStrategy,
) -> None:
    df = make_df([BEAR_OB, BEAR_IMPULSE, BEAR_CONFIRM_NO_FVG])
    result = default_strategy.evaluate(df, "ETH/USDT", "1h")

    assert not result.matched
    assert result.stage is FilterStage.FVG
    assert result.stage.reached_order_block


def test_fvg_is_the_only_difference_between_accept_and_reject(
    default_strategy: OrderBlockStrategy,
) -> None:
    """Same block, same impulse — only the confirmation candle's low moves."""
    accepted = default_strategy.evaluate(
        make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM]), "BTC/USDT", "1h"
    )
    rejected = default_strategy.evaluate(
        make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM_NO_FVG]), "BTC/USDT", "1h"
    )
    assert accepted.matched
    assert not rejected.matched
    assert rejected.stage is FilterStage.FVG


# ---------------------------------------------------------------------------
# Structural rejection
# ---------------------------------------------------------------------------
def test_rejected_without_an_order_block(default_strategy: OrderBlockStrategy) -> None:
    continuation = (95.0, 100.0, 94.0, 99.0)  # bullish before a bullish impulse
    df = make_df([continuation, BULL_IMPULSE, BULL_CONFIRM])
    result = default_strategy.evaluate(df, "BTC/USDT", "1h")

    assert not result.matched
    assert result.stage is FilterStage.ORDER_BLOCK
    assert not result.stage.reached_order_block


def test_warmup_when_history_is_too_short(default_strategy: OrderBlockStrategy) -> None:
    df = make_df([BULL_IMPULSE, BULL_CONFIRM])
    result = default_strategy.evaluate(df, "BTC/USDT", "1h")
    assert not result.matched
    assert result.stage is FilterStage.WARMUP


def test_required_candles_is_the_structure_length(
    default_strategy: OrderBlockStrategy,
) -> None:
    assert default_strategy.required_candles == 3


# ---------------------------------------------------------------------------
# Risk-stage rejection
# ---------------------------------------------------------------------------
def test_rejected_when_the_structure_cannot_be_sized() -> None:
    """A short whose stop exceeds 1/4 of entry makes a 1:4 target negative.

    Bodies are kept well above the doji threshold so the rejection can only
    come from the sizing stage.
    """
    ob = (30.0, 100.0, 20.0, 99.0)      # bullish, body 69 of range 80
    impulse = (99.0, 99.5, 15.0, 16.0)  # bearish displacement
    confirm = (16.0, 18.0, 14.0, 15.0)  # high 18 < ob low 20 -> valid FVG
    strategy = OrderBlockStrategy(reward_ratio=4.0)
    result = strategy.evaluate(make_df([ob, impulse, confirm]), "X/USDT", "1h")

    assert not result.matched
    assert result.stage is FilterStage.RISK
    # Everything structural passed; only sizing refused it.
    assert result.stage.reached_fvg


# ---------------------------------------------------------------------------
# Stage ordering
# ---------------------------------------------------------------------------
def test_stage_depth_follows_declaration_order() -> None:
    order = [
        FilterStage.WARMUP,
        FilterStage.ORDER_BLOCK,
        FilterStage.FVG,
        FilterStage.RISK,
        FilterStage.CONFIRMED,
    ]
    assert list(FilterStage) == order
    assert [s.depth for s in order] == sorted(s.depth for s in order)


def test_passed_is_strict_about_the_stage_it_stopped_at() -> None:
    for stage in FilterStage:
        assert not stage.passed(stage)
    assert FilterStage.RISK.passed(FilterStage.FVG)
    assert not FilterStage.FVG.passed(FilterStage.FVG)


def test_dedup_key_identifies_the_block_not_the_confirmation(
    default_strategy: OrderBlockStrategy,
) -> None:
    df = make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM])
    signal = default_strategy.evaluate(df, "BTC/USDT", "1h").signal
    assert signal is not None
    assert signal.dedup_key == (
        "BTC/USDT",
        "1h",
        int(df["timestamp"].iloc[-3]),
        "LONG",
    )
