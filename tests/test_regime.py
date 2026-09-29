"""Unit tests for regime classification and the regime kill switch.

The properties that matter most are pinned hardest: ADX must match an
independent implementation, both measurements must be causal (the value at a bar
may not depend on later bars), and a filter must block only what it claims to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scanner.analytics import _confirm_from_tag
from scanner.config import ConfigError, Settings
from scanner.regime import (
    Regime,
    RegimeFilter,
    RegimeGate,
    RegimeMethod,
    adx_frame,
    structure_frame,
)
from scanner.strategy import FilterStage, OrderBlockStrategy
from tests.test_mtf import CONFIRMING_LONG, ltf
from tests.test_smc import BULL_CONFIRM, BULL_IMPULSE, BULL_OB, build_spatial_frame
from tests.test_watchlist import make_zone

_HOUR_MS = 3_600_000


def random_walk(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.5, n)
    high = np.maximum.reduce([close + rng.uniform(0.1, 1.5, n), open_, close])
    low = np.minimum.reduce([close - rng.uniform(0.1, 1.5, n), open_, close])
    return pd.DataFrame(
        {
            "timestamp": np.arange(n) * _HOUR_MS,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1.0,
        }
    )


def bars(rows: list[tuple[float, float, float, float]], start_ms: int = 0) -> pd.DataFrame:
    """rows are (open, high, low, close), one hour apart."""
    return pd.DataFrame(
        {
            "timestamp": [start_ms + i * _HOUR_MS for i in range(len(rows))],
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": 1.0,
        }
    )


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------
def test_adx_matches_an_independent_implementation() -> None:
    pta = pytest.importorskip("pandas_ta_classic")
    df = random_walk()
    ours = adx_frame(df, period=14)["adx"]
    reference = pta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    converged = slice(200, None)
    assert (ours[converged] - reference[converged]).abs().max() < 1e-3


def test_adx_is_causal() -> None:
    """The value at bar t must not change when later bars are appended."""
    df = random_walk()
    full = adx_frame(df)["adx"]
    for t in range(60, len(df), 41):
        assert adx_frame(df.iloc[: t + 1])["adx"].iloc[-1] == pytest.approx(full.iloc[t])


def test_adx_on_the_live_window_matches_full_history() -> None:
    """Live reads the last 200 bars; the backtest has years. They must agree."""
    df = random_walk()
    assert adx_frame(df.tail(200))["adx"].iloc[-1] == pytest.approx(
        adx_frame(df)["adx"].iloc[-1], abs=0.01
    )


def test_adx_warmup_is_masked() -> None:
    adx = adx_frame(random_walk(100), period=14)["adx"]
    assert adx.iloc[:28].isna().all()
    assert adx.iloc[28:].notna().all()


def test_steady_trend_reads_high_and_tight_chop_reads_low() -> None:
    x = np.arange(300)
    trend = bars([(100 + i * 0.5, 100 + i * 0.5 + 0.6, 100 + i * 0.5 - 0.2, 100 + i * 0.5 + 0.4)
                  for i in x])
    # Alternates up and down every bar: movement with no persistent direction.
    zig = [100 + (1.0 if i % 2 else -1.0) for i in x]
    chop = bars([(z, z + 1.2, z - 1.2, 200 - z) for z in zig])
    assert adx_frame(trend)["adx"].iloc[-1] > 40
    assert adx_frame(chop)["adx"].iloc[-1] < 20


def test_adx_rejects_a_degenerate_period() -> None:
    with pytest.raises(ValueError, match="period"):
        adx_frame(random_walk(50), period=1)


# ---------------------------------------------------------------------------
# Structural containment
# ---------------------------------------------------------------------------
# One swing high (110, bar 1) and one swing low (90, bar 3), then a steady grind
# upward inside them. Rising highs and rising lows form no further pivots at
# strength 1, so the range stays exactly [90, 110] — any local extreme after the
# two pivots would redraw it, which is what an earlier version of this fixture
# got wrong.
RANGE = [
    (100, 101, 99.5, 100),
    (100, 110, 99, 109),   # 1: swing high 110
    (109, 108, 95, 96),
    (96, 97, 90, 91),      # 3: swing low 90
    (91, 98, 92, 97),
    (97, 100, 93, 99),
    (99, 102, 94, 101),
    (101, 104, 95, 103),
    (103, 105, 96, 104),
]


def test_price_inside_confirmed_bounds_is_contained() -> None:
    frame = structure_frame(bars(RANGE), strength=1)
    last = frame.iloc[-1]
    assert bool(last["valid"])
    assert last["swing_high"] == pytest.approx(110)
    assert last["swing_low"] == pytest.approx(90)
    assert bool(last["contained"])


def test_a_full_body_close_beyond_the_bound_escapes() -> None:
    frame = structure_frame(bars(RANGE + [(111, 116, 111, 115)]), strength=1)
    assert not bool(frame.iloc[-1]["contained"])


def test_a_wick_beyond_the_bound_is_a_sweep_not_a_breakout() -> None:
    """Body inside, wick through the high: liquidity taken, range intact."""
    frame = structure_frame(bars(RANGE + [(104, 114, 103, 108)]), strength=1)
    assert bool(frame.iloc[-1]["contained"])


def test_a_pivot_is_only_known_strength_bars_after_it_prints() -> None:
    """Until `strength` bars have printed after it, no scanner could confirm it."""
    peak = bars([
        (100, 101, 99, 100),
        (100, 102, 99, 101),
        (101, 110, 100, 109),   # 2: pivot high, needs bars 0-1 and 3-4
        (109, 108, 101, 102),
        (102, 105, 100, 103),
        (103, 104, 101, 102),
    ])
    frame = structure_frame(peak, strength=2)
    assert pd.isna(frame.iloc[3]["swing_high"])            # one bar short
    assert frame.iloc[4]["swing_high"] == pytest.approx(110)  # confirmed now


def test_structure_is_causal() -> None:
    df = random_walk(400)
    full = structure_frame(df, strength=5)
    for t in range(40, len(df), 29):
        prefix = structure_frame(df.iloc[: t + 1], strength=5).iloc[-1]
        assert bool(prefix["contained"]) == bool(full.iloc[t]["contained"])


# ---------------------------------------------------------------------------
# RegimeFilter
# ---------------------------------------------------------------------------
def always(method: RegimeMethod = RegimeMethod.ADX, gate: RegimeGate = RegimeGate.ENTRY,
           ranging: bool = True) -> RegimeFilter:
    """ADX is 0..100, so a threshold above 100 always reads ranging, 0 never."""
    return RegimeFilter(method, gate, adx_period=2, adx_threshold=101.0 if ranging else 0.0,
                        lookback=200)


def test_off_never_blocks() -> None:
    f = RegimeFilter()
    reading = f.read(random_walk(100))
    assert reading.regime is Regime.TRENDING
    assert not f.blocks(reading)


def test_ranging_reading_blocks_and_trending_does_not() -> None:
    df = random_walk(100)
    assert always(ranging=True).blocks(always(ranging=True).read(df))
    assert not always(ranging=False).blocks(always(ranging=False).read(df))


def test_unknown_never_blocks() -> None:
    """Short history must not silently halt all trading."""
    f = RegimeFilter(RegimeMethod.ADX)
    reading = f.read(random_walk(10))
    assert reading.regime is Regime.UNKNOWN
    assert not f.blocks(reading)


def test_read_at_excludes_the_candle_still_forming() -> None:
    df = random_walk(100)
    f = RegimeFilter(RegimeMethod.ADX)
    mid_bar = int(df["timestamp"].iloc[-1]) + _HOUR_MS // 2
    at_close = int(df["timestamp"].iloc[-1]) + _HOUR_MS
    assert f.read_at(df, mid_bar).adx == pytest.approx(f.read(df.iloc[:-1]).adx)
    assert f.read_at(df, at_close).adx == pytest.approx(f.read(df).adx)


def test_confluence_needs_both_measurements_to_agree() -> None:
    df = bars(RANGE * 6)
    contained = structure_frame(df.tail(200), strength=5).iloc[-1]["contained"]
    both = RegimeFilter(RegimeMethod.CONFLUENCE, adx_period=2, adx_threshold=101.0)
    reading = both.read(df)
    assert reading.is_ranging == bool(contained)


def test_gate_placement_flags() -> None:
    assert RegimeGate.ADMISSION.at_admission and not RegimeGate.ADMISSION.at_entry
    assert RegimeGate.ENTRY.at_entry and not RegimeGate.ENTRY.at_admission
    assert RegimeGate.BOTH.at_admission and RegimeGate.BOTH.at_entry


# ---------------------------------------------------------------------------
# Admission gate — inside the strategy
# ---------------------------------------------------------------------------
def extreme_long() -> pd.DataFrame:
    return build_spatial_frame([BULL_OB, BULL_IMPULSE, BULL_CONFIRM], range_low=0.0,
                               range_high=400.0)


def test_admission_gate_rejects_a_valid_block_while_ranging() -> None:
    strategy = OrderBlockStrategy(max_stop_pct=0.0,
                                  regime=always(gate=RegimeGate.ADMISSION, ranging=True))
    result = strategy.evaluate(extreme_long(), "BTC/USDT", "1h")
    assert result.stage is FilterStage.REGIME
    assert "ranging" in result.reason
    assert result.stage.reached_spatial  # it was a valid extreme block


def test_admission_gate_passes_the_same_block_when_trending() -> None:
    strategy = OrderBlockStrategy(max_stop_pct=0.0,
                                  regime=always(gate=RegimeGate.ADMISSION, ranging=False))
    assert strategy.evaluate(extreme_long(), "BTC/USDT", "1h").matched


def test_an_entry_only_gate_does_not_touch_admission() -> None:
    strategy = OrderBlockStrategy(max_stop_pct=0.0,
                                  regime=always(gate=RegimeGate.ENTRY, ranging=True))
    assert strategy.evaluate(extreme_long(), "BTC/USDT", "1h").matched


# ---------------------------------------------------------------------------
# Entry gate — inside the historical replay
# ---------------------------------------------------------------------------
def confirm(regime: RegimeFilter | None):
    frame = ltf(CONFIRMING_LONG)
    first = int(frame["timestamp"].iloc[0])
    htf = random_walk(80)
    htf["timestamp"] = first - (80 - np.arange(80)) * _HOUR_MS   # all closed before
    return _confirm_from_tag(
        frame, make_zone(), tagged_ms=first, strategy_timeframe="15m",
        confirm_window=30, horizon=40, min_fvg_pct=0.0, swing_strength=1,
        regime=regime, htf=htf, ltf_bar_ms=15 * 60_000,
    )


def test_entry_gate_suppresses_a_trigger_while_ranging() -> None:
    trigger, reason, suppressed, _note = confirm(always(ranging=True))
    assert trigger is None
    assert reason == "regime"
    assert suppressed >= 1


def test_entry_gate_lets_the_same_trigger_through_when_trending() -> None:
    trigger, reason, suppressed, note = confirm(always(ranging=False))
    assert trigger is not None
    assert reason == "confirmed"
    assert suppressed == 0
    assert note.startswith("trending")


def test_no_filter_confirms_exactly_as_before() -> None:
    trigger, reason, suppressed, note = confirm(None)
    assert trigger is not None and reason == "confirmed"
    assert suppressed == 0 and note == ""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_regime_settings_parse_and_default_off(monkeypatch) -> None:
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.delenv("REGIME_FILTER", raising=False)
    assert Settings.from_env().regime_method is RegimeMethod.OFF
    monkeypatch.setenv("REGIME_FILTER", "ADX")
    monkeypatch.setenv("REGIME_GATE", "both")
    s = Settings.from_env()
    assert s.regime_method is RegimeMethod.ADX
    assert s.regime_gate is RegimeGate.BOTH


def test_unknown_regime_method_is_a_config_error(monkeypatch) -> None:
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("REGIME_FILTER", "vibes")
    with pytest.raises(ConfigError, match="REGIME_FILTER"):
        Settings.from_env()


def test_regime_window_longer_than_a_live_fetch_is_rejected(monkeypatch) -> None:
    """The backtest must never read more history than the scanner fetches."""
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("REGIME_FILTER", "adx")
    monkeypatch.setenv("CANDLE_LIMIT", "200")
    monkeypatch.setenv("REGIME_LOOKBACK", "500")
    with pytest.raises(ConfigError, match="REGIME_LOOKBACK"):
        Settings.from_env()


# ---------------------------------------------------------------------------
# Entry gate — inside the live scanner
# ---------------------------------------------------------------------------
def live_bot(tmp_path, monkeypatch, *, ranging: bool):
    """A dry-run bot with an isolated watchlist file and an always/never filter."""
    import dataclasses

    from scanner.bot import ScannerBot

    monkeypatch.setenv("DRY_RUN", "true")
    settings = dataclasses.replace(
        Settings.from_env(),
        watchlist_file=tmp_path / "watchlist.json",
        regime_method=RegimeMethod.ADX,
        regime_gate=RegimeGate.ENTRY,
        adx_period=2,
        adx_threshold=101.0 if ranging else 0.0,
        swing_strength=1,   # CONFIRMING_LONG is built around strength-1 pivots
    )
    return ScannerBot(settings)


def tagged_setup():
    frame = ltf(CONFIRMING_LONG)
    first = int(frame["timestamp"].iloc[0])
    htf = random_walk(80)
    htf["timestamp"] = first - (80 - np.arange(80)) * _HOUR_MS
    zone = make_zone()
    zone.state = zone.state.TAGGED
    zone.tagged_ms = first
    return zone, frame, htf


def test_live_gate_suppresses_and_keeps_the_zone_waiting(tmp_path, monkeypatch) -> None:
    bot = live_bot(tmp_path, monkeypatch, ranging=True)
    zone, frame, htf = tagged_setup()
    trigger, suppressed = bot._try_confirm(zone, frame, htf)
    assert trigger is None and suppressed
    assert zone.state.value == "tagged"   # not consumed; a later trigger may fire


def test_live_gate_dispatches_when_trending(tmp_path, monkeypatch) -> None:
    bot = live_bot(tmp_path, monkeypatch, ranging=False)
    zone, frame, htf = tagged_setup()
    trigger, suppressed = bot._try_confirm(zone, frame, htf)
    assert trigger is not None and not suppressed
    assert zone.state.value == "triggered"
