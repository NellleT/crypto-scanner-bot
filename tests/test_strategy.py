"""Unit tests for the multi-factor signal logic.

Each filter is exercised in isolation by holding the other two satisfied, so a
failure points at exactly one condition.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scanner.indicators import add_indicators
from scanner.patterns import PatternType
from scanner.strategy import EngulfingTrendStrategy, FilterStage, SignalDirection

_MINUTE_MS = 60_000

# Small periods keep the fixtures readable; the arithmetic is identical at 200.
SMA_PERIOD = 5
VOL_PERIOD = 3
FILLER_BARS = 10

# Two-candle fixtures. Bodies are wide relative to range, so the doji filter
# never interferes with what these tests are actually checking.
BULLISH_PREV = (100.0, 101.0, 97.0, 98.0)   # bearish body 100 -> 98
BULLISH_CURR = (97.5, 101.5, 97.0, 100.5)   # bullish body 97.5 -> 100.5, engulfs
BEARISH_PREV = (98.0, 101.0, 97.0, 100.0)   # bullish body 98 -> 100
BEARISH_CURR = (100.5, 101.0, 97.0, 97.5)   # bearish body 100.5 -> 97.5, engulfs

NO_PATTERN_PREV = (98.0, 101.0, 97.0, 100.0)  # both bullish — not engulfing
NO_PATTERN_CURR = (99.0, 102.0, 98.0, 101.0)

# Filler levels that put the SMA on the required side of the signal candle while
# staying within a plausible distance of it. A filler far from the signal price
# (say 200 against a close of 97.5) implies a >100% gap between adjacent bars,
# which no real market prints and which makes the structural stop wider than the
# entry itself — see test_absurd_structural_range_is_rejected.
LONG_FILLER = 90.0
SHORT_FILLER = 110.0


def build_frame(
    previous: tuple[float, float, float, float],
    current: tuple[float, float, float, float],
    *,
    filler_close: float,
    filler_volume: float = 100.0,
    previous_volume: float = 100.0,
    current_volume: float = 500.0,
) -> pd.DataFrame:
    """Assemble a closed-candle frame ending in the two supplied bars.

    ``filler_close`` sets where the moving average sits relative to the signal
    candle, which is how the trend filter is steered on and off.

    The default volumes satisfy both volume filters: 500 clears the 3-period
    average of (100, 100, 500) = 233.3, and also expands over the engulfed
    candle's 100. ``current_volume`` and ``previous_volume`` are what the VSA
    tests below vary independently.
    """
    rows: list[dict[str, float]] = []
    for i in range(FILLER_BARS):
        rows.append(
            {
                "timestamp": 1_700_000_000_000 + i * _MINUTE_MS,
                "open": filler_close,
                "high": filler_close + 1.0,
                "low": filler_close - 1.0,
                "close": filler_close,
                "volume": filler_volume,
            }
        )
    for offset, (bar, volume) in enumerate(
        ((previous, previous_volume), (current, current_volume))
    ):
        open_, high, low, close = bar
        rows.append(
            {
                "timestamp": 1_700_000_000_000 + (FILLER_BARS + offset) * _MINUTE_MS,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    df = pd.DataFrame.from_records(rows)
    return add_indicators(df, sma_period=SMA_PERIOD, volume_sma_period=VOL_PERIOD)


@pytest.fixture
def strategy() -> EngulfingTrendStrategy:
    return EngulfingTrendStrategy(sma_period=SMA_PERIOD, volume_sma_period=VOL_PERIOD)


# ---------------------------------------------------------------------------
# Confirmed signals
# ---------------------------------------------------------------------------
def test_long_signal_when_all_three_filters_agree(strategy: EngulfingTrendStrategy) -> None:
    df = build_frame(BULLISH_PREV, BULLISH_CURR, filler_close=90.0)
    result = strategy.evaluate(df, "BTC/USDT", "4h")

    assert result.matched, result.reason
    signal = result.signal
    assert signal is not None
    assert signal.direction is SignalDirection.LONG
    assert signal.pattern is PatternType.BULLISH_ENGULFING
    assert signal.price == pytest.approx(100.5)
    assert signal.price > signal.trend_sma
    assert signal.volume > signal.volume_sma
    assert signal.engulf_ratio == pytest.approx(1.5)  # body 3.0 vs 2.0


def test_short_signal_when_all_three_filters_agree(strategy: EngulfingTrendStrategy) -> None:
    df = build_frame(BEARISH_PREV, BEARISH_CURR, filler_close=SHORT_FILLER)
    result = strategy.evaluate(df, "ETH/USDT", "4h")

    assert result.matched, result.reason
    signal = result.signal
    assert signal is not None
    assert signal.direction is SignalDirection.SHORT
    assert signal.pattern is PatternType.BEARISH_ENGULFING
    assert signal.price < signal.trend_sma
    assert signal.volume > signal.volume_sma


# ---------------------------------------------------------------------------
# Trend filter
# ---------------------------------------------------------------------------
def test_long_rejected_below_the_trend_average(strategy: EngulfingTrendStrategy) -> None:
    """Bullish engulfing in a downtrend is counter-trend — not a signal."""
    df = build_frame(BULLISH_PREV, BULLISH_CURR, filler_close=200.0)
    result = strategy.evaluate(df, "BTC/USDT", "4h")
    assert not result.matched
    assert "not above" in result.reason


def test_short_rejected_above_the_trend_average(strategy: EngulfingTrendStrategy) -> None:
    df = build_frame(BEARISH_PREV, BEARISH_CURR, filler_close=10.0)
    result = strategy.evaluate(df, "ETH/USDT", "4h")
    assert not result.matched
    assert "not below" in result.reason


def test_trend_comparison_is_strict(strategy: EngulfingTrendStrategy) -> None:
    """Close exactly equal to the SMA must not qualify."""
    df = build_frame(BULLISH_PREV, BULLISH_CURR, filler_close=90.0)
    df.loc[df.index[-1], "SMA_5"] = df["close"].iloc[-1]
    result = strategy.evaluate(df, "BTC/USDT", "4h")
    assert not result.matched
    assert "not above" in result.reason


# ---------------------------------------------------------------------------
# Volume filter
# ---------------------------------------------------------------------------
def test_rejected_when_volume_is_below_average(strategy: EngulfingTrendStrategy) -> None:
    df = build_frame(BULLISH_PREV, BULLISH_CURR, filler_close=90.0, current_volume=10.0)
    result = strategy.evaluate(df, "BTC/USDT", "4h")
    assert not result.matched
    assert "volume" in result.reason


def test_volume_comparison_is_strict(strategy: EngulfingTrendStrategy) -> None:
    """Volume exactly equal to its average must not qualify."""
    df = build_frame(
        BULLISH_PREV,
        BULLISH_CURR,
        filler_close=90.0,
        filler_volume=100.0,
        previous_volume=100.0,
        current_volume=100.0,
    )
    assert df["VOL_SMA_3"].iloc[-1] == pytest.approx(100.0)
    result = strategy.evaluate(df, "BTC/USDT", "4h")
    assert not result.matched
    assert "volume" in result.reason


# ---------------------------------------------------------------------------
# VSA follow-through filter
#
# The anomaly this rule exists for: a candle can sit comfortably above its
# 20-period volume average and still print LESS volume than the bar it engulfed.
# The two volume filters are independent and neither implies the other.
# ---------------------------------------------------------------------------
def test_rejected_when_volume_does_not_expand_over_the_engulfed_candle(
    strategy: EngulfingTrendStrategy,
) -> None:
    """Volume above the average but below the prior bar — the live-testing case."""
    df = build_frame(
        BULLISH_PREV,
        BULLISH_CURR,
        filler_close=LONG_FILLER,
        filler_volume=10.0,      # keeps the 3-period average low
        previous_volume=900.0,   # the engulfed bar traded heavily
        current_volume=500.0,    # signal bar: above average, but FADING
    )
    # Precondition: the liquidity filter genuinely passes, so the rejection can
    # only come from the VSA rule.
    assert df["volume"].iloc[-1] > df["VOL_SMA_3"].iloc[-1]

    result = strategy.evaluate(df, "BTC/USDT", "4h")
    assert not result.matched
    assert result.stage is FilterStage.VSA
    assert "did not expand" in result.reason
    assert result.stage.reached_volume  # it cleared the average first


def test_vsa_comparison_is_strict(strategy: EngulfingTrendStrategy) -> None:
    """Equal volume on both bars is not expansion."""
    df = build_frame(
        BULLISH_PREV,
        BULLISH_CURR,
        filler_close=LONG_FILLER,
        filler_volume=10.0,
        previous_volume=500.0,
        current_volume=500.0,
    )
    result = strategy.evaluate(df, "BTC/USDT", "4h")
    assert not result.matched
    assert result.stage is FilterStage.VSA


def test_vsa_accepts_genuine_expansion(strategy: EngulfingTrendStrategy) -> None:
    df = build_frame(
        BULLISH_PREV,
        BULLISH_CURR,
        filler_close=LONG_FILLER,
        previous_volume=200.0,
        current_volume=600.0,
    )
    result = strategy.evaluate(df, "BTC/USDT", "4h")
    assert result.matched, result.reason
    assert result.signal is not None
    assert result.signal.volume_expansion_ratio == pytest.approx(3.0)


def test_vsa_applies_to_shorts_too(strategy: EngulfingTrendStrategy) -> None:
    df = build_frame(
        BEARISH_PREV,
        BEARISH_CURR,
        filler_close=SHORT_FILLER,
        filler_volume=10.0,
        previous_volume=900.0,
        current_volume=500.0,
    )
    result = strategy.evaluate(df, "ETH/USDT", "4h")
    assert not result.matched
    assert result.stage is FilterStage.VSA


def test_vsa_rule_can_be_disabled() -> None:
    """Same frame, opposite outcomes — proves the flag is what decides."""
    df = build_frame(
        BULLISH_PREV,
        BULLISH_CURR,
        filler_close=LONG_FILLER,
        filler_volume=10.0,
        previous_volume=900.0,
        current_volume=500.0,
    )
    strict = EngulfingTrendStrategy(sma_period=SMA_PERIOD, volume_sma_period=VOL_PERIOD)
    lenient = EngulfingTrendStrategy(
        sma_period=SMA_PERIOD,
        volume_sma_period=VOL_PERIOD,
        require_volume_expansion=False,
    )
    assert strict.evaluate(df, "BTC/USDT", "4h").stage is FilterStage.VSA

    permitted = lenient.evaluate(df, "BTC/USDT", "4h")
    assert permitted.matched, permitted.reason
    # The signal still reports the contraction honestly.
    assert permitted.signal is not None
    assert permitted.signal.volume_expansion_ratio < 1.0


def test_zero_volume_on_the_engulfed_candle_is_rejected(
    strategy: EngulfingTrendStrategy,
) -> None:
    """Guards the ratio in the rejection message against divide-by-zero."""
    df = build_frame(
        BULLISH_PREV,
        BULLISH_CURR,
        filler_close=LONG_FILLER,
        filler_volume=10.0,
        previous_volume=0.0,
        current_volume=0.0,
    )
    result = strategy.evaluate(df, "BTC/USDT", "4h")
    assert not result.matched
    # Rejected earlier, by the average, but must not raise either way.
    assert result.stage in (FilterStage.VOLUME, FilterStage.VSA)


# ---------------------------------------------------------------------------
# Pattern filter
# ---------------------------------------------------------------------------
def test_rejected_without_a_pattern(strategy: EngulfingTrendStrategy) -> None:
    df = build_frame(NO_PATTERN_PREV, NO_PATTERN_CURR, filler_close=90.0)
    result = strategy.evaluate(df, "BTC/USDT", "4h")
    assert not result.matched
    assert result.reason == "no engulfing pattern"


def test_doji_filter_still_applies(strategy: EngulfingTrendStrategy) -> None:
    """A one-tick body engulfed in a valid trend is still not a signal."""
    thin_prev = (100.0, 105.0, 95.0, 99.9)   # body 0.1 of range 10
    thin_curr = (99.5, 101.0, 99.0, 100.5)
    df = build_frame(thin_prev, thin_curr, filler_close=90.0)
    assert not strategy.evaluate(df, "BTC/USDT", "4h").matched

    permissive = EngulfingTrendStrategy(
        sma_period=SMA_PERIOD, volume_sma_period=VOL_PERIOD, min_body_ratio=0.0
    )
    assert permissive.evaluate(df, "BTC/USDT", "4h").matched


# ---------------------------------------------------------------------------
# Warm-up and input contract
# ---------------------------------------------------------------------------
def test_rejected_while_indicators_are_warming_up() -> None:
    """A newly listed market has no SMA-200 yet, so it must produce nothing."""
    strategy = EngulfingTrendStrategy(sma_period=200, volume_sma_period=20)
    df = build_frame(BULLISH_PREV, BULLISH_CURR, filler_close=90.0)
    df = add_indicators(df, sma_period=200, volume_sma_period=20)
    result = strategy.evaluate(df, "NEW/USDT", "4h")
    assert not result.matched
    assert "not warmed up" in result.reason


def test_missing_indicator_columns_is_a_programming_error(
    strategy: EngulfingTrendStrategy,
) -> None:
    df = build_frame(BULLISH_PREV, BULLISH_CURR, filler_close=90.0)
    with pytest.raises(ValueError, match="add_indicators"):
        strategy.evaluate(df.drop(columns=["SMA_5"]), "BTC/USDT", "4h")


def test_too_few_candles(strategy: EngulfingTrendStrategy) -> None:
    df = build_frame(BULLISH_PREV, BULLISH_CURR, filler_close=90.0).iloc[:1]
    result = strategy.evaluate(df, "BTC/USDT", "4h")
    assert not result.matched
    assert "two-bar" in result.reason


def test_required_candles_reflects_configuration() -> None:
    assert EngulfingTrendStrategy(sma_period=200, volume_sma_period=20).required_candles == 201
    assert EngulfingTrendStrategy(sma_period=50, volume_sma_period=20).required_candles == 51


# ---------------------------------------------------------------------------
# Reporting fields used in the alert message
# ---------------------------------------------------------------------------
def test_signal_reports_volume_and_distance(strategy: EngulfingTrendStrategy) -> None:
    df = build_frame(
        BULLISH_PREV,
        BULLISH_CURR,
        filler_close=90.0,
        filler_volume=100.0,
        previous_volume=100.0,
        current_volume=400.0,
    )
    signal = strategy.evaluate(df, "BTC/USDT", "4h").signal
    assert signal is not None

    # VOL_SMA_3 over (100, 100, 400) = 200 -> 400 is exactly 2x its average.
    assert signal.volume_sma == pytest.approx(200.0)
    assert signal.volume_ratio == pytest.approx(2.0)

    expected = (signal.price - signal.trend_sma) / signal.trend_sma * 100.0
    assert signal.sma_distance_pct == pytest.approx(expected)
    assert signal.sma_distance_pct > 0  # long signals sit above the average


# ---------------------------------------------------------------------------
# Risk structure
# ---------------------------------------------------------------------------
def test_confirmed_signal_carries_a_risk_plan(strategy: EngulfingTrendStrategy) -> None:
    df = build_frame(BULLISH_PREV, BULLISH_CURR, filler_close=LONG_FILLER)
    signal = strategy.evaluate(df, "BTC/USDT", "4h").signal
    assert signal is not None

    plan = signal.risk
    assert plan.entry == pytest.approx(signal.price)
    assert plan.stop_loss < plan.entry          # long stop sits below entry
    assert plan.risk_per_unit > 0
    assert [tp.ratio for tp in plan.take_profits] == [2.0, 3.0]
    assert plan.take_profits[0].price < plan.take_profits[1].price


def test_stop_anchors_to_the_structural_low_not_the_signal_candle(
    strategy: EngulfingTrendStrategy,
) -> None:
    """The whole point of a structural stop: it must sit beyond the recent swing."""
    df = build_frame(BULLISH_PREV, BULLISH_CURR, filler_close=LONG_FILLER)
    plan = strategy.evaluate(df, "BTC/USDT", "4h").signal.risk  # type: ignore[union-attr]

    signal_candle_low = BULLISH_CURR[2]
    structural_low = float(df["low"].tail(strategy.structural_lookback).min())

    assert plan.structural_level == pytest.approx(structural_low)
    assert plan.stop_loss < structural_low        # buffered beyond the extreme
    assert plan.stop_loss < signal_candle_low     # and well beyond the signal bar


def test_absurd_structural_range_is_rejected_at_the_risk_stage() -> None:
    """A short whose stop exceeds entry makes a 1:2 target negative.

    Risk above 1/max(rr) of entry cannot produce a real target price. Alerting a
    negative take-profit would be worse than staying silent.
    """
    strategy = EngulfingTrendStrategy(sma_period=SMA_PERIOD, volume_sma_period=VOL_PERIOD)
    df = build_frame(BEARISH_PREV, BEARISH_CURR, filler_close=200.0)
    result = strategy.evaluate(df, "ETH/USDT", "4h")

    assert not result.matched
    assert result.stage is FilterStage.RISK
    # It got all the way past the entry filters before risk sizing refused it.
    assert result.stage.reached_volume


def test_risk_plan_respects_custom_lookback_and_targets() -> None:
    strategy = EngulfingTrendStrategy(
        sma_period=SMA_PERIOD,
        volume_sma_period=VOL_PERIOD,
        structural_lookback=3,
        stop_buffer_pct=0.5,
        rr_targets=(1.5,),
    )
    df = build_frame(BULLISH_PREV, BULLISH_CURR, filler_close=LONG_FILLER)
    plan = strategy.evaluate(df, "BTC/USDT", "4h").signal.risk  # type: ignore[union-attr]

    # Only the last 3 bars are considered, so the deeper filler lows are ignored.
    assert plan.structural_level == pytest.approx(float(df["low"].tail(3).min()))
    assert plan.lookback == 3
    assert plan.stop_loss == pytest.approx(plan.structural_level * 0.995)
    assert [tp.ratio for tp in plan.take_profits] == [1.5]


def test_required_candles_covers_a_long_structural_lookback() -> None:
    strategy = EngulfingTrendStrategy(
        sma_period=20, volume_sma_period=20, structural_lookback=500
    )
    assert strategy.required_candles == 500


# ---------------------------------------------------------------------------
# Stage attribution
#
# The trend and volume rejection messages BOTH contain "is not above", so a
# funnel built by substring-matching `reason` silently counts volume rejections
# as trend rejections. These tests pin the typed stage that callers must use.
# ---------------------------------------------------------------------------
def test_each_filter_reports_its_own_stage(strategy: EngulfingTrendStrategy) -> None:
    confirmed = build_frame(BULLISH_PREV, BULLISH_CURR, filler_close=90.0)
    no_pattern = build_frame(NO_PATTERN_PREV, NO_PATTERN_CURR, filler_close=90.0)
    bad_trend = build_frame(BULLISH_PREV, BULLISH_CURR, filler_close=200.0)
    thin_volume = build_frame(
        BULLISH_PREV, BULLISH_CURR, filler_close=90.0, current_volume=10.0
    )

    fading_volume = build_frame(
        BULLISH_PREV,
        BULLISH_CURR,
        filler_close=LONG_FILLER,
        filler_volume=10.0,
        previous_volume=900.0,
        current_volume=500.0,
    )

    assert strategy.evaluate(confirmed, "X/USDT", "4h").stage is FilterStage.CONFIRMED
    assert strategy.evaluate(no_pattern, "X/USDT", "4h").stage is FilterStage.PATTERN
    assert strategy.evaluate(bad_trend, "X/USDT", "4h").stage is FilterStage.TREND
    assert strategy.evaluate(thin_volume, "X/USDT", "4h").stage is FilterStage.VOLUME
    assert strategy.evaluate(fading_volume, "X/USDT", "4h").stage is FilterStage.VSA


def test_trend_and_volume_rejections_are_not_distinguishable_by_text(
    strategy: EngulfingTrendStrategy,
) -> None:
    """Regression guard for the funnel-counting bug this enum exists to prevent."""
    bad_trend = strategy.evaluate(
        build_frame(BULLISH_PREV, BULLISH_CURR, filler_close=200.0), "X/USDT", "4h"
    )
    thin_volume = strategy.evaluate(
        build_frame(BULLISH_PREV, BULLISH_CURR, filler_close=90.0, current_volume=10.0),
        "X/USDT",
        "4h",
    )
    # Both messages share this substring — which is exactly why `stage` exists.
    assert "is not above" in bad_trend.reason
    assert "is not above" in thin_volume.reason
    assert bad_trend.stage is not thin_volume.stage


def test_warmup_stage_for_cold_indicators() -> None:
    strategy = EngulfingTrendStrategy(sma_period=200, volume_sma_period=20)
    df = add_indicators(
        build_frame(BULLISH_PREV, BULLISH_CURR, filler_close=90.0),
        sma_period=200,
        volume_sma_period=20,
    )
    assert strategy.evaluate(df, "NEW/USDT", "4h").stage is FilterStage.WARMUP


def test_stage_helpers_describe_funnel_depth() -> None:
    assert FilterStage.CONFIRMED.reached_pattern and FilterStage.CONFIRMED.reached_trend
    assert FilterStage.CONFIRMED.reached_volume and FilterStage.CONFIRMED.reached_vsa
    assert FilterStage.VOLUME.reached_pattern and FilterStage.VOLUME.reached_trend
    assert not FilterStage.VOLUME.reached_volume  # it stopped AT the volume filter
    assert FilterStage.VSA.reached_volume and not FilterStage.VSA.reached_vsa
    assert FilterStage.TREND.reached_pattern and not FilterStage.TREND.reached_trend
    assert not FilterStage.PATTERN.reached_pattern
    assert not FilterStage.WARMUP.reached_pattern


def test_stage_depth_follows_declaration_order() -> None:
    """Depth is derived, so inserting a stage cannot desync the funnel helpers."""
    order = [
        FilterStage.WARMUP,
        FilterStage.PATTERN,
        FilterStage.TREND,
        FilterStage.VOLUME,
        FilterStage.VSA,
        FilterStage.RISK,
        FilterStage.CONFIRMED,
    ]
    assert list(FilterStage) == order
    assert [s.depth for s in order] == sorted(s.depth for s in order)
    assert FilterStage.CONFIRMED.depth == max(s.depth for s in FilterStage)


def test_passed_is_strict_about_the_stage_it_stopped_at() -> None:
    for stage in FilterStage:
        assert not stage.passed(stage), f"{stage} should not claim to pass itself"
    assert FilterStage.RISK.passed(FilterStage.VSA)
    assert not FilterStage.VSA.passed(FilterStage.VSA)


def test_dedup_key_is_stable_per_candle(strategy: EngulfingTrendStrategy) -> None:
    df = build_frame(BULLISH_PREV, BULLISH_CURR, filler_close=90.0)
    first = strategy.evaluate(df, "BTC/USDT", "4h").signal
    second = strategy.evaluate(df, "BTC/USDT", "4h").signal
    assert first is not None and second is not None
    assert first.dedup_key == second.dedup_key == ("BTC/USDT", "4h", first.candle.timestamp)
