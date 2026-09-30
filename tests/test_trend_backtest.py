"""Unit tests for the V5.0 backtest: fills, costs, R, portfolio accounting,
delistings, the no-leverage cap, and the random-entry control."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scanner.trend_backtest import (
    SPOT_COSTS,
    Costs,
    buy_and_hold,
    curve_stats,
    random_entries,
    simulate_portfolio,
    symbol_trades,
    trade_stats,
)
from tests.test_trend import BARS, P, _BASE_MS, _DAY_MS, frame

# Buy 11.6 * 1.001, sell 11.1 * 0.999, 0.1% fee on each, over a 1.3 planned risk.
NET_R = (11.1 * 0.999 - 11.6 * 1.001 - 0.001 * (11.6 * 1.001 + 11.1 * 0.999)) / 1.3


def test_the_trade_fills_at_next_opens_and_prices_costs_in_r() -> None:
    (trade,) = symbol_trades("TEST/USDT", frame(), P)
    assert trade.signal_ms == _BASE_MS + 5 * _DAY_MS
    assert (trade.entry_ms, trade.entry_open) == (_BASE_MS + 6 * _DAY_MS, 11.6)
    assert (trade.exit_ms, trade.exit_price) == (_BASE_MS + 9 * _DAY_MS, 11.1)
    assert trade.planned_risk == pytest.approx(1.3)
    assert trade.gross_r == pytest.approx(-0.5 / 1.3)
    assert trade.r(SPOT_COSTS) == pytest.approx(NET_R)
    assert trade.r(Costs(0.0, 0.0)) == pytest.approx(trade.gross_r)
    assert trade.exit_kind == "channel" and trade.days_held == 3


def test_a_position_at_the_end_is_open_or_delisted() -> None:
    held = frame(BARS[:8])                       # entered, never exited
    (open_,) = symbol_trades("TEST/USDT", held, P)
    assert open_.exit_kind == "open" and open_.exit_price == 12.3   # marked at last close
    (gone,) = symbol_trades("TEST/USDT", held, P, data_end_ms=_BASE_MS + 30 * _DAY_MS)
    assert gone.exit_kind == "delisted"


def test_portfolio_pnl_is_the_trade_r_times_the_risk_amount() -> None:
    result = simulate_portfolio({"TEST/USDT": frame()}, P, SPOT_COSTS, equity=10_000.0)
    assert len(result.fills) == 1 and result.capped == 0
    assert result.equity.iloc[-1] == pytest.approx(10_000.0 + NET_R * 100.0)
    # On the fill day the account holds 100 / 1.3 units, marked at the close.
    qty = 100.0 / 1.3
    cash = 10_000.0 - qty * 11.6 * 1.001 * 1.001
    assert result.equity.iloc[6] == pytest.approx(cash + qty * 12.0)


def test_no_leverage_an_oversized_entry_is_shrunk_to_the_cash() -> None:
    result = simulate_portfolio({"TEST/USDT": frame()}, P, SPOT_COSTS,
                                equity=10_000.0, risk_pct=50.0)   # wants ~3,846 units
    assert result.capped == 1
    assert (result.exposure <= 1.0 + 1e-9).all()
    assert result.fills[0]["cost"] == pytest.approx(10_000.0)


def test_a_delisted_holding_is_sold_at_its_last_close() -> None:
    frames = {
        "GONE/USDT": frame(BARS[:8]),                                  # ends while held
        "HOLD/USDT": frame([(5.0, 5.1, 4.9, 5.0)] * 12),               # keeps the clock
    }
    result = simulate_portfolio(frames, P, SPOT_COSTS, equity=10_000.0)
    (fill,) = result.fills
    assert fill["kind"] == "delisted"
    qty = 100.0 / 1.3
    assert fill["proceeds"] == pytest.approx(qty * 12.3 * 0.999 * 0.999)


def test_random_entries_use_the_same_stop_and_exit_rule() -> None:
    from tests.test_trend import random_walk
    from scanner.trend import TrendParams, trend_frame

    df, params = random_walk(), TrendParams()
    f = trend_frame(df, params).set_index("timestamp")
    for trade in random_entries("RW", df, params, count=40, rng=np.random.default_rng(1)):
        signal = f.loc[trade.signal_ms]
        assert bool(signal["regime"])
        assert trade.initial_stop == signal["next_stop"]
        if trade.exit_kind == "channel":
            exit_day = f.index[f.index.get_loc(trade.exit_ms) - 1]
            assert bool(f.loc[exit_day, "exit_signal"])


def test_trade_stats_claim_no_interval_from_under_three_months() -> None:
    stats = trade_stats(symbol_trades("TEST/USDT", frame(), P), SPOT_COSTS, draws=100)
    assert stats.trades == 1 and stats.win_rate == 0.0
    assert stats.total == pytest.approx(NET_R)
    assert np.isnan(stats.ci_low)


def test_curve_stats_drawdown_and_time_under_water() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    stats = curve_stats(pd.Series([100.0, 120.0, 90.0, 110.0], index=index))
    assert stats.max_drawdown_pct == pytest.approx(-25.0)
    assert stats.longest_underwater_days == 2
    assert stats.total_return_pct == pytest.approx(10.0)


def test_buy_and_hold_pays_costs_on_the_way_in() -> None:
    curve = buy_and_hold(frame(), SPOT_COSTS, equity=1_000.0)
    qty = 1_000.0 / (10.0 * 1.001 * 1.001)
    assert curve.iloc[0] == pytest.approx(qty * 10.0)
    assert curve.iloc[-1] == pytest.approx(qty * 11.0)
