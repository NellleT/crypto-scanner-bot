"""Unit tests for the V4.0 evaluation: bracket walk, costs, position rule,
controls. Uses the hand-traced range from test_deviation."""

from __future__ import annotations

import numpy as np
import pytest

from scanner.deviation import DeviationParams
from scanner.deviation_backtest import (
    FUTURES_FEES,
    Fees,
    backtest,
    flipped_r,
    random_r,
    summarize,
    walk_bracket,
)
from tests.test_deviation import RANGE, WICK_ABOVE, frame

P = DeviationParams(lookback=20, strength=1)

# After the 109 short (stop 111.111, target 105): bar 8 itself becomes a 111
# swing high, a second wick over it re-sweeps while the first trade is still
# open, then price falls through the 105 target.
FOLLOW = [
    (109.0, 110.5, 107.0, 108.0),   # 9   confirms bar 8 as a 111 pivot
    (108.0, 111.05, 107.5, 109.0),  # 10  sweeps 111, closes inside: 2nd deviation
    (109.0, 109.5, 104.0, 104.5),   # 11  first trade's target
]


def test_the_stop_wins_a_candle_that_spans_both_levels() -> None:
    highs, lows = np.array([112.0]), np.array([104.0])
    assert walk_bracket(highs, lows, 0, long=False, stop=111.0, target=105.0) == ("sl", 0)


def test_the_walk_reports_the_first_level_reached() -> None:
    highs = np.array([110.0, 109.0, 108.0])
    lows = np.array([107.0, 106.0, 104.0])
    assert walk_bracket(highs, lows, 0, long=False, stop=111.0, target=105.0) == ("tp", 2)
    assert walk_bracket(highs, lows, 0, long=False, stop=120.0, target=90.0) == ("open", -1)


def test_one_position_per_symbol_and_costs_in_r() -> None:
    report = backtest({"TEST/USDT": frame(RANGE + [WICK_ABOVE] + FOLLOW)}, P)
    assert report.deviations == 2
    assert report.skipped_busy == 1                      # the re-sweep at bar 10
    (trade,) = report.trades
    assert trade.status == "tp" and trade.exit_index == 11
    assert trade.r == pytest.approx(4.0 / 2.111)
    # Taker in at 109, maker out at the 105 target, over 2.111 of risk.
    expected_fee = (0.0005 * 109.0 + 0.0002 * 105.0) / 2.111
    assert trade.fee_r(FUTURES_FEES) == pytest.approx(expected_fee)
    assert trade.net_r(FUTURES_FEES) == pytest.approx(trade.r - expected_fee)


def test_the_flipped_control_takes_the_breakout_side_at_the_same_distances() -> None:
    df = frame(RANGE + [WICK_ABOVE] + FOLLOW)
    (trade,) = backtest({"TEST/USDT": df}, P).trades
    # Long from 109, stop 106.889, target 113: bar 11's low of 104 stops it.
    assert flipped_r(trade, df) == -1.0


def test_random_entries_use_the_same_bracket_geometry() -> None:
    df = frame(RANGE + [WICK_ABOVE] + FOLLOW)
    (trade,) = backtest({"TEST/USDT": df}, P).trades
    draws = random_r(trade, df, draws=50, rng=np.random.default_rng(0), tail_margin=1)
    assert draws
    assert all(d == -1.0 or d == pytest.approx(trade.plan.reward_ratio) for d in draws)


def test_summary_counts_net_of_fees() -> None:
    report = backtest({"TEST/USDT": frame(RANGE + [WICK_ABOVE] + FOLLOW)}, P)
    free = summarize(report.trades, Fees(0.0, 0.0), draws=200)
    assert free.trades == 1 and free.wins == 1 and free.win_rate == 100.0
    assert free.net_r == pytest.approx(4.0 / 2.111)
    costly = summarize(report.trades, FUTURES_FEES, draws=200)
    assert costly.net_r < free.net_r


def test_no_confidence_interval_is_claimed_from_under_three_months() -> None:
    """Resampling a single month only reshuffles it; the interval would collapse
    to a point and read as certainty."""
    report = backtest({"TEST/USDT": frame(RANGE + [WICK_ABOVE] + FOLLOW)}, P)
    summary = summarize(report.trades, FUTURES_FEES, draws=200)
    assert np.isnan(summary.ci_low) and np.isnan(summary.ci_high)
