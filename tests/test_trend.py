"""Unit tests for the V5.0 daily trend rules.

A hand-traceable 10-day series at miniature settings (SMA 5, 3-day entry
channel, 2-day exit channel):

* day 5 closes 11.5 above the prior 3-day high of 11.0, with the SMA at 10.7 —
  an entry; the initial stop is the 2-day low ending that day, 10.2;
* day 8 closes 11.2 under the prior 2-day low of 11.4 — an exit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scanner.trend import (
    SignalKind,
    TrendParams,
    latest_signal,
    plan_trend_entry,
    trend_frame,
)

_DAY_MS = 86_400_000
_BASE_MS = 1_600_000_000_000 - (1_600_000_000_000 % _DAY_MS)
P = TrendParams(sma=5, entry_channel=3, exit_channel=2)

BARS = [
    (10.0, 10.5, 9.5, 10.0),    # 0
    (10.0, 10.6, 9.6, 10.2),    # 1
    (10.2, 10.8, 9.8, 10.4),    # 2
    (10.4, 10.9, 10.0, 10.6),   # 3
    (10.6, 11.0, 10.2, 10.8),   # 4  above the SMA, but not above 10.9
    (10.8, 11.6, 10.7, 11.5),   # 5  ENTRY: 11.5 > 11.0, SMA 10.7
    (11.6, 12.2, 11.4, 12.0),   # 6  filled at the 11.6 open
    (12.0, 12.5, 11.8, 12.3),   # 7
    (12.3, 12.4, 11.0, 11.2),   # 8  EXIT: 11.2 < 11.4
    (11.1, 11.3, 10.8, 11.0),   # 9  sold at the 11.1 open
]


def frame(bars=None) -> pd.DataFrame:
    bars = BARS if bars is None else bars
    return pd.DataFrame(
        {
            "timestamp": [_BASE_MS + i * _DAY_MS for i in range(len(bars))],
            "open": [b[0] for b in bars],
            "high": [b[1] for b in bars],
            "low": [b[2] for b in bars],
            "close": [b[3] for b in bars],
            "volume": [1.0] * len(bars),
        }
    )


def random_walk(n: int = 900, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.03, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0, 0.015, n)) * close
    return pd.DataFrame(
        {
            "timestamp": _BASE_MS + np.arange(n) * _DAY_MS,
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": 1.0,
        }
    )


def test_channels_exclude_the_current_candle() -> None:
    f = trend_frame(frame(), P)
    assert f.at[5, "entry_level"] == 11.0      # highs of days 2-4, not day 5's 11.6
    assert f.at[8, "exit_level"] == 11.4       # lows of days 6-7, not day 8's 11.0
    assert f.at[5, "next_stop"] == 10.2        # lows of days 4-5: tomorrow's exit level
    assert f.at[5, "sma"] == pytest.approx(10.7)


def test_signals_fire_exactly_where_the_rules_say() -> None:
    f = trend_frame(frame(), P)
    assert f.index[f["entry_signal"]].tolist()[0] == 5
    assert not f.at[4, "entry_signal"]         # above the SMA, below the channel
    assert f.at[8, "exit_signal"]
    assert not f["entry_signal"].iloc[:5].any()   # nothing inside the warm-up


def test_a_breakout_below_the_moving_average_is_not_an_entry() -> None:
    """The regime filter: a channel break in a downtrend is ignored."""
    # A gap from 100 down to 30: day 4's close of 100 is still inside the 5-day
    # average on day 8, but already outside its 3-day channel.
    bars = [(100.0, 101.0, 99.0, 100.0)] * 5 + [(30.0, 31.0, 29.0, 30.0)] * 3
    bars.append((30.0, 32.5, 29.5, 32.0))      # 32 > prior 3-day high of 31
    f = trend_frame(frame(bars), P)
    last = f.iloc[-1]
    assert last["close"] > last["entry_level"]          # the channel IS broken...
    assert last["sma"] == pytest.approx(44.4)           # ...but below the average
    assert not last["entry_signal"]


def test_indicators_are_causal() -> None:
    """Every row must be what a live scanner would have computed that day."""
    df = random_walk()
    full = trend_frame(df, TrendParams())
    cols = ["sma", "entry_level", "exit_level", "next_stop", "entry_signal", "exit_signal"]
    for m in (250, 401, 777, 900):
        part = trend_frame(df.iloc[:m], TrendParams())
        pd.testing.assert_frame_equal(part[cols], full[cols].iloc[:m])


def test_latest_signal_answers_the_live_question() -> None:
    entry = latest_signal(frame(BARS[:6]), P, in_position=False)
    assert entry is not None and entry.kind is SignalKind.ENTRY
    assert (entry.close, entry.level, entry.stop) == (11.5, 11.0, 10.2)
    assert latest_signal(frame(BARS[:6]), P, in_position=True) is None

    exit_ = latest_signal(frame(BARS[:9]), P, in_position=True)
    assert exit_ is not None and exit_.kind is SignalKind.EXIT and exit_.level == 11.4
    assert latest_signal(frame(BARS[:9]), P, in_position=False) is None
    assert latest_signal(frame(BARS[:4]), P, in_position=False) is None   # warm-up


def test_entry_is_sized_on_the_initial_stop() -> None:
    signal = latest_signal(frame(BARS[:6]), P, in_position=False)
    plan = plan_trend_entry(signal, equity=10_000.0, risk_pct=1.0)
    assert plan is not None
    assert plan.risk_per_unit == pytest.approx(1.3)
    assert plan.quantity == pytest.approx(100.0 / 1.3)
    assert plan.stop_distance_pct == pytest.approx(1.3 / 11.5 * 100)


def test_params_are_validated() -> None:
    with pytest.raises(ValueError):
        TrendParams(exit_channel=0)
    assert TrendParams().warmup == 200
