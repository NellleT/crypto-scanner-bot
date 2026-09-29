"""Unit tests for V4.0 liquidity-sweep detection and trade planning.

The fixtures are small enough to trace by hand at pivot strength 1: a range
from a swing low at 100 (bar 3) to a swing high at 110 (bar 1), with an
internal high at 108 (bar 5), then one candle that decides the case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scanner.deviation import (
    DeviationParams,
    PoolSide,
    TargetMode,
    latest_deviation,
    plan_deviation,
    scan_deviations,
)
from scanner.smc import Direction

_HOUR_MS = 3_600_000
_BASE_MS = 1_700_000_000_000
P = DeviationParams(lookback=20, strength=1)

RANGE = [
    (105.0, 106.0, 104.0, 105.0),   # 0
    (105.0, 110.0, 104.5, 109.0),   # 1  swing high 110 -> BSL
    (109.0, 109.5, 103.0, 104.0),   # 2
    (104.0, 105.0, 100.0, 101.0),   # 3  swing low 100 -> SSL
    (101.0, 106.0, 100.5, 105.0),   # 4
    (105.0, 108.0, 104.0, 107.0),   # 5  internal high 108
    (107.0, 107.5, 102.0, 103.0),   # 6
    (103.0, 104.0, 101.0, 102.0),   # 7
]
WICK_ABOVE = (102.0, 111.0, 101.5, 109.0)     # pierces 110, closes back inside
CLOSE_ABOVE = (102.0, 111.0, 101.5, 110.5)    # pierces 110 and closes outside
RECLAIM = (110.5, 112.0, 109.0, 109.5)        # next candle closes back inside
HOLD_1 = (110.5, 112.0, 110.2, 111.5)
HOLD_2 = (111.5, 113.0, 111.0, 112.5)
LATE_RETURN = (112.5, 112.6, 108.0, 109.0)    # back inside, but too late


def frame(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [_BASE_MS + i * _HOUR_MS for i in range(len(bars))],
            "open": [b[0] for b in bars],
            "high": [b[1] for b in bars],
            "low": [b[2] for b in bars],
            "close": [b[3] for b in bars],
            "volume": [1.0] * len(bars),
        }
    )


def mirror(bars):
    """Reflect through 105 — the range's equilibrium — turning highs into lows."""
    return [(210.0 - o, 210.0 - l, 210.0 - h, 210.0 - c) for o, h, l, c in bars]


def random_walk(n: int = 1500, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    return pd.DataFrame(
        {
            "timestamp": _BASE_MS + np.arange(n) * _HOUR_MS,
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": 1.0,
        }
    )


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def test_a_wick_through_the_high_that_closes_inside_is_a_short_deviation() -> None:
    found = scan_deviations(frame(RANGE + [WICK_ABOVE]), P)
    assert len(found) == 1
    dev = found[0]
    assert dev.swept is PoolSide.BUY_SIDE and dev.direction is Direction.SHORT
    assert dev.pool.price == 110.0 and dev.range.low.price == 100.0
    assert dev.range.equilibrium == 105.0
    assert (dev.extreme, dev.close, dev.reclaim_bars) == (111.0, 109.0, 0)
    assert dev.sweep_ms == dev.confirm_ms == _BASE_MS + 8 * _HOUR_MS


def test_a_close_outside_reclaimed_next_candle_is_a_deviation() -> None:
    found = scan_deviations(frame(RANGE + [CLOSE_ABOVE, RECLAIM]), P)
    assert len(found) == 1
    dev = found[0]
    assert dev.reclaim_bars == 1
    assert dev.extreme == 112.0                 # the furthest point of the whole sweep
    assert dev.close == 109.5
    assert dev.sweep_ms == _BASE_MS + 8 * _HOUR_MS
    assert dev.confirm_ms == _BASE_MS + 9 * _HOUR_MS


def test_a_break_that_holds_beyond_the_reclaim_window_is_not_traded() -> None:
    """Two closes outside means the break was accepted, not a grab — even if
    price wanders back in afterwards."""
    bars = RANGE + [CLOSE_ABOVE, HOLD_1, HOLD_2, LATE_RETURN]
    assert scan_deviations(frame(bars), P) == []


def test_the_reclaim_window_is_configurable() -> None:
    bars = RANGE + [CLOSE_ABOVE, HOLD_1, HOLD_2, LATE_RETURN]
    wide = DeviationParams(lookback=20, strength=1, reclaim_bars=3)
    found = scan_deviations(frame(bars), wide)
    assert len(found) == 1 and found[0].reclaim_bars == 3
    strict = DeviationParams(lookback=20, strength=1, reclaim_bars=0)
    assert scan_deviations(frame(RANGE + [CLOSE_ABOVE, RECLAIM]), strict) == []


def test_a_candle_through_both_ends_of_the_range_is_not_traded() -> None:
    engulfing = (102.0, 111.0, 99.0, 105.0)
    assert scan_deviations(frame(RANGE + [engulfing]), P) == []


def test_sell_side_sweeps_mirror_into_longs() -> None:
    found = scan_deviations(frame(mirror(RANGE + [WICK_ABOVE])), P)
    assert len(found) == 1
    dev = found[0]
    assert dev.swept is PoolSide.SELL_SIDE and dev.direction is Direction.LONG
    assert (dev.pool.price, dev.extreme, dev.close) == (100.0, 99.0, 101.0)


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------
def test_pools_older_than_the_lookback_do_not_define_the_range() -> None:
    """With a 6-candle lookback the 110 high (bar 1) has aged out by bar 8; the
    internal 108 high is the range top, so the same candle closes OUTSIDE it and
    only the next candle's close back under 108 confirms."""
    short = DeviationParams(lookback=6, strength=1)
    bars = RANGE + [WICK_ABOVE, (109.0, 109.2, 106.0, 107.0)]
    found = scan_deviations(frame(bars), short)
    assert len(found) == 1
    assert found[0].pool.price == 108.0 and found[0].reclaim_bars == 1


def test_a_pool_already_traded_through_cannot_be_swept_again() -> None:
    """After an accepted break above 110, a later wick over 110 sweeps nothing:
    those stops were filled on the way up."""
    bars = RANGE + [CLOSE_ABOVE, HOLD_1, HOLD_2, LATE_RETURN,
                    (109.0, 109.5, 107.0, 108.0), (108.0, 111.0, 107.5, 108.5)]
    for dev in scan_deviations(frame(bars), P):
        assert dev.pool.price != 110.0


def test_nothing_is_traded_without_both_ends_of_a_range() -> None:
    assert scan_deviations(frame(RANGE[:4] + [WICK_ABOVE]), P) == []


# ---------------------------------------------------------------------------
# Causality and live/backtest parity
# ---------------------------------------------------------------------------
def test_appending_candles_never_changes_an_earlier_verdict() -> None:
    df = random_walk()
    params = DeviationParams(lookback=30, strength=2)
    full = scan_deviations(df, params)
    assert len(full) > 20, "fixture should produce plenty of deviations"
    stamps = df["timestamp"].to_numpy()
    for m in (200, 457, 800, 1111, 1499):
        cutoff = int(stamps[m - 1])
        assert scan_deviations(df.iloc[:m], params) == [d for d in full if d.confirm_ms <= cutoff]


def test_the_live_window_sees_exactly_what_full_history_sees() -> None:
    """The live scanner only fetches a window of candles; its verdict on the
    newest candle must be the backtest's verdict on that same candle."""
    df = random_walk()
    params = DeviationParams(lookback=30, strength=2)
    by_confirm = {d.confirm_ms: d for d in scan_deviations(df, params)}
    stamps = df["timestamp"].to_numpy()
    need = params.required_candles
    checked = 0
    for m in range(need, len(df)):
        live = latest_deviation(df.iloc[m - need : m], params)
        assert live == by_confirm.get(int(stamps[m - 1])), m
        checked += live is not None
    assert checked > 20


def test_a_frame_shorter_than_the_live_window_returns_nothing() -> None:
    assert latest_deviation(frame(RANGE + [WICK_ABOVE]), P) is None


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def deviation(bars=None, params=P):
    return scan_deviations(frame(bars or RANGE + [WICK_ABOVE]), params)[0]


def test_short_plan_stops_beyond_the_wick_and_targets_equilibrium() -> None:
    plan, rejection = plan_deviation(deviation(), P, equity=10_000.0, risk_pct=1.0)
    assert rejection is None and plan is not None
    assert plan.direction is Direction.SHORT
    assert plan.entry == 109.0
    assert plan.stop_loss == pytest.approx(111.111)           # 111 + 0.1%
    assert plan.take_profit == 105.0
    assert plan.reward_ratio == pytest.approx(4.0 / 2.111)
    assert plan.quantity == pytest.approx(100.0 / 2.111)      # 1% of 10k at risk


def test_opposite_target_reaches_for_the_far_pool() -> None:
    params = DeviationParams(lookback=20, strength=1, target=TargetMode.OPPOSITE)
    plan, _ = plan_deviation(deviation(params=params), params)
    assert plan is not None and plan.take_profit == 100.0
    assert plan.reward_ratio == pytest.approx(9.0 / 2.111)


def test_long_plan_mirrors() -> None:
    plan, _ = plan_deviation(deviation(mirror(RANGE + [WICK_ABOVE])), P)
    assert plan is not None and plan.direction is Direction.LONG
    assert plan.stop_loss == pytest.approx(99.0 * 0.999)
    assert plan.take_profit == 105.0


def test_a_close_already_past_equilibrium_leaves_nothing_to_take() -> None:
    deep = (102.0, 111.0, 101.5, 104.0)        # wick to 111, close under 105
    plan, rejection = plan_deviation(deviation(RANGE + [deep]), P)
    assert plan is None and rejection is not None and rejection.stage == "no_reward"


def test_stop_width_and_reward_floors_reject() -> None:
    narrow = DeviationParams(lookback=20, strength=1, max_stop_pct=1.0)
    _, rejection = plan_deviation(deviation(), narrow)          # 2.111 / 109 = 1.94%
    assert rejection is not None and rejection.stage == "stop_width"

    picky = DeviationParams(lookback=20, strength=1, min_reward_ratio=2.0)
    _, rejection = plan_deviation(deviation(), picky)           # 1:1.89
    assert rejection is not None and rejection.stage == "reward_ratio"


def test_params_are_validated() -> None:
    with pytest.raises(ValueError):
        DeviationParams(lookback=4, strength=2)
    with pytest.raises(ValueError):
        DeviationParams(strength=0)
    with pytest.raises(ValueError):
        DeviationParams(reclaim_bars=-1)
