"""Unit tests for structural stop placement and R:R targets.

Pure arithmetic — no network, no indicators. Values are chosen so the expected
stop and targets can be checked by hand.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scanner.risk import (
    DEFAULT_STOP_BUFFER_PCT,
    DEFAULT_STRUCTURAL_LOOKBACK,
    build_risk_plan,
    structural_extreme,
)

_MINUTE_MS = 60_000


def make_df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """bars are (open, high, low, close)."""
    return pd.DataFrame(
        {
            "timestamp": [1_700_000_000_000 + i * _MINUTE_MS for i in range(len(bars))],
            "open": [b[0] for b in bars],
            "high": [b[1] for b in bars],
            "low": [b[2] for b in bars],
            "close": [b[3] for b in bars],
            "volume": [100.0] * len(bars),
        }
    )


def ramp(count: int, *, low: float, high: float) -> list[tuple[float, float, float, float]]:
    mid = (low + high) / 2
    return [(mid, high, low, mid)] * count


# ---------------------------------------------------------------------------
# Structural extreme
# ---------------------------------------------------------------------------
def test_structural_low_scans_the_lookback_window() -> None:
    df = make_df(ramp(5, low=100.0, high=110.0) + [(105.0, 112.0, 95.0, 108.0)])
    assert structural_extreme(df, is_long=True, lookback=10) == pytest.approx(95.0)


def test_structural_high_scans_the_lookback_window() -> None:
    df = make_df(ramp(5, low=100.0, high=110.0) + [(105.0, 125.0, 103.0, 108.0)])
    assert structural_extreme(df, is_long=False, lookback=10) == pytest.approx(125.0)


def test_lookback_window_excludes_older_bars() -> None:
    """A deeper low outside the window must not be picked up."""
    df = make_df(
        [(100.0, 105.0, 50.0, 100.0)]          # ancient spike low, outside the window
        + ramp(5, low=90.0, high=110.0)
    )
    assert structural_extreme(df, is_long=True, lookback=3) == pytest.approx(90.0)
    assert structural_extreme(df, is_long=True, lookback=10) == pytest.approx(50.0)


def test_lookback_clamps_when_history_is_short() -> None:
    df = make_df(ramp(3, low=99.0, high=101.0))
    assert structural_extreme(df, is_long=True, lookback=50) == pytest.approx(99.0)


def test_signal_candle_is_included_in_the_window() -> None:
    """The newest bar sets the extreme when it is the most extreme."""
    df = make_df(ramp(5, low=100.0, high=110.0) + [(105.0, 111.0, 88.0, 106.0)])
    assert structural_extreme(df, is_long=True, lookback=10) == pytest.approx(88.0)


def test_structural_extreme_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="no candles"):
        structural_extreme(make_df([]), is_long=True, lookback=10)
    with pytest.raises(ValueError, match="missing 'low'"):
        structural_extreme(pd.DataFrame({"high": [1.0]}), is_long=True, lookback=10)


# ---------------------------------------------------------------------------
# Long plans
# ---------------------------------------------------------------------------
def test_long_plan_is_computed_by_hand() -> None:
    df = make_df(ramp(9, low=100.0, high=110.0) + [(101.0, 112.0, 100.0, 110.0)])
    plan = build_risk_plan(df, is_long=True, entry=110.0, lookback=10, buffer_pct=0.1)
    assert plan is not None

    assert plan.structural_level == pytest.approx(100.0)
    assert plan.stop_loss == pytest.approx(99.9)          # 100 * (1 - 0.001)
    assert plan.risk_per_unit == pytest.approx(10.1)      # 110 - 99.9
    assert plan.risk_pct == pytest.approx(10.1 / 110 * 100)
    assert plan.take_profits[0].price == pytest.approx(130.2)  # 110 + 2 * 10.1
    assert plan.take_profits[1].price == pytest.approx(140.3)  # 110 + 3 * 10.1


def test_long_stop_sits_below_the_structural_low() -> None:
    df = make_df(ramp(10, low=100.0, high=110.0))
    plan = build_risk_plan(df, is_long=True, entry=108.0)
    assert plan is not None
    assert plan.stop_loss < plan.structural_level
    assert plan.stop_loss < 100.0


# ---------------------------------------------------------------------------
# Short plans
# ---------------------------------------------------------------------------
def test_short_plan_is_computed_by_hand() -> None:
    df = make_df(ramp(9, low=90.0, high=100.0) + [(99.0, 100.0, 88.0, 90.0)])
    plan = build_risk_plan(df, is_long=False, entry=90.0, lookback=10, buffer_pct=0.1)
    assert plan is not None

    assert plan.structural_level == pytest.approx(100.0)
    assert plan.stop_loss == pytest.approx(100.1)         # 100 * (1 + 0.001)
    assert plan.risk_per_unit == pytest.approx(10.1)      # 100.1 - 90
    assert plan.take_profits[0].price == pytest.approx(69.8)   # 90 - 2 * 10.1
    assert plan.take_profits[1].price == pytest.approx(59.7)   # 90 - 3 * 10.1


def test_short_stop_sits_above_the_structural_high() -> None:
    df = make_df(ramp(10, low=90.0, high=100.0))
    plan = build_risk_plan(df, is_long=False, entry=92.0)
    assert plan is not None
    assert plan.stop_loss > plan.structural_level
    assert plan.stop_loss > 100.0


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------
def test_short_with_risk_above_one_third_of_entry_is_rejected() -> None:
    """A 1:3 target would be negative — not a price any venue can express."""
    df = make_df(ramp(10, low=10.0, high=100.0))
    plan = build_risk_plan(df, is_long=False, entry=20.0, rr_targets=(2.0, 3.0))
    assert plan is None


def test_short_is_accepted_when_only_a_reachable_target_is_configured() -> None:
    """The same setup is fine if the ladder stays above zero."""
    df = make_df(ramp(10, low=10.0, high=100.0))
    plan = build_risk_plan(df, is_long=False, entry=95.0, rr_targets=(1.0,))
    assert plan is not None
    assert plan.take_profits[0].price > 0


def test_non_positive_entry_is_rejected() -> None:
    df = make_df(ramp(10, low=100.0, high=110.0))
    assert build_risk_plan(df, is_long=True, entry=0.0) is None
    assert build_risk_plan(df, is_long=True, entry=-5.0) is None


def test_stop_on_the_wrong_side_of_entry_is_rejected() -> None:
    """Entry below the structural low cannot produce positive long risk."""
    df = make_df(ramp(10, low=100.0, high=110.0))
    assert build_risk_plan(df, is_long=True, entry=99.0) is None


def test_empty_reward_ladder_is_rejected() -> None:
    df = make_df(ramp(10, low=100.0, high=110.0))
    assert build_risk_plan(df, is_long=True, entry=108.0, rr_targets=()) is None


def test_non_positive_ratios_are_skipped_not_fatal() -> None:
    df = make_df(ramp(10, low=100.0, high=110.0))
    plan = build_risk_plan(df, is_long=True, entry=108.0, rr_targets=(-1.0, 2.0))
    assert plan is not None
    assert [tp.ratio for tp in plan.take_profits] == [2.0]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def test_plan_describes_its_own_anchor() -> None:
    df = make_df(ramp(10, low=100.0, high=110.0))
    long_plan = build_risk_plan(df, is_long=True, entry=108.0, lookback=10)
    short_plan = build_risk_plan(df, is_long=False, entry=101.0, lookback=10)
    assert long_plan is not None and short_plan is not None

    assert long_plan.stop_description == "10-bar low"
    assert short_plan.stop_description == "10-bar high"
    assert long_plan.take_profits[0].label == "1:2"


def test_reward_pct_and_prices_are_consistent() -> None:
    df = make_df(ramp(10, low=100.0, high=110.0))
    plan = build_risk_plan(df, is_long=True, entry=110.0)
    assert plan is not None

    first = plan.take_profits[0]
    expected = (first.price - plan.entry) / plan.entry * 100.0
    assert plan.reward_pct(first) == pytest.approx(expected)

    # prices() feeds group formatting: entry, stop, then each target in order.
    assert plan.prices() == (
        plan.entry,
        plan.stop_loss,
        plan.take_profits[0].price,
        plan.take_profits[1].price,
    )


def test_reward_is_the_configured_multiple_of_risk() -> None:
    df = make_df(ramp(10, low=100.0, high=110.0))
    plan = build_risk_plan(df, is_long=True, entry=110.0, rr_targets=(2.0, 3.0))
    assert plan is not None
    for take_profit in plan.take_profits:
        reward = abs(take_profit.price - plan.entry)
        assert reward / plan.risk_per_unit == pytest.approx(take_profit.ratio)


def test_defaults_match_the_documented_values() -> None:
    assert DEFAULT_STRUCTURAL_LOOKBACK == 10
    assert DEFAULT_STOP_BUFFER_PCT == 0.1
