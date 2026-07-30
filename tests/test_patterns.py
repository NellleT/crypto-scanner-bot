"""Unit tests for the engulfing detection rules.

Run with:  python -m pytest -q
The pattern module is pure, so these tests need no network access.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scanner.patterns import Candle, PatternType, classify_engulfing, validate_ohlcv

_MINUTE_MS = 60_000


def make_candle(open_: float, high: float, low: float, close: float, index: int = 0) -> Candle:
    return Candle(
        timestamp=1_700_000_000_000 + index * _MINUTE_MS,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100.0,
    )


def make_df(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    records = [
        {
            "timestamp": 1_700_000_000_000 + i * _MINUTE_MS,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": 100.0,
        }
        for i, (o, h, l, c) in enumerate(rows)
    ]
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Bullish engulfing
# ---------------------------------------------------------------------------
def test_bullish_engulfing_is_detected() -> None:
    previous = make_candle(100.0, 101.0, 97.0, 98.0)          # bearish body 100 -> 98
    current = make_candle(97.5, 101.5, 97.0, 100.5, index=1)  # bullish body 97.5 -> 100.5
    assert classify_engulfing(previous, current) is PatternType.BULLISH_ENGULFING


def test_bullish_accepts_an_open_exactly_at_the_previous_close() -> None:
    """No gap is required — Bybit stitches every candle open to the prior close.

    Demanding ``open < prev.close`` makes the pattern undetectable on venues
    that publish a continuous series (100% of Bybit candles), which is a
    property of the data feed, not of price action.
    """
    previous = make_candle(100.0, 101.0, 97.0, 98.0)
    current = make_candle(98.0, 101.5, 98.0, 100.5, index=1)  # opens AT previous close
    assert classify_engulfing(previous, current) is PatternType.BULLISH_ENGULFING


def test_bullish_rejects_an_open_above_the_previous_close() -> None:
    """Opening inside the previous body means it was never fully engulfed."""
    previous = make_candle(100.0, 101.0, 97.0, 98.0)
    current = make_candle(99.0, 101.5, 98.5, 100.5, index=1)
    assert classify_engulfing(previous, current) is None


def test_bullish_requires_close_above_previous_open() -> None:
    previous = make_candle(100.0, 101.0, 97.0, 98.0)
    current = make_candle(97.5, 101.5, 97.0, 100.0, index=1)  # closes AT previous open
    assert classify_engulfing(previous, current) is None


# ---------------------------------------------------------------------------
# Bearish engulfing
# ---------------------------------------------------------------------------
def test_bearish_engulfing_is_detected() -> None:
    previous = make_candle(98.0, 101.0, 97.0, 100.0)          # bullish body 98 -> 100
    current = make_candle(100.5, 101.0, 97.0, 97.5, index=1)  # bearish body 100.5 -> 97.5
    assert classify_engulfing(previous, current) is PatternType.BEARISH_ENGULFING


def test_bearish_requires_strictly_engulfing_body() -> None:
    previous = make_candle(98.0, 101.0, 97.0, 100.0)
    current = make_candle(100.5, 101.0, 97.0, 98.5, index=1)  # closes inside previous body
    assert classify_engulfing(previous, current) is None


def test_bearish_accepts_an_open_exactly_at_the_previous_close() -> None:
    previous = make_candle(98.0, 101.0, 97.0, 100.0)
    current = make_candle(100.0, 101.0, 97.0, 97.5, index=1)  # opens AT previous close
    assert classify_engulfing(previous, current) is PatternType.BEARISH_ENGULFING


def test_close_matching_the_previous_open_is_not_engulfing() -> None:
    """The close side stays strict: matching is not covering."""
    previous = make_candle(98.0, 101.0, 97.0, 100.0)
    current = make_candle(100.0, 101.0, 97.0, 98.0, index=1)  # closes AT previous open
    assert classify_engulfing(previous, current) is None


def test_stitched_series_still_produces_patterns() -> None:
    """Simulates a Bybit-style feed where every open equals the prior close."""
    bearish = make_candle(100.0, 101.0, 97.0, 98.0)
    bullish = make_candle(98.0, 102.0, 97.5, 101.0, index=1)
    assert bullish.open == bearish.close
    assert classify_engulfing(bearish, bullish) is PatternType.BULLISH_ENGULFING


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------
def test_same_direction_candles_are_not_a_pattern() -> None:
    previous = make_candle(98.0, 101.0, 97.0, 100.0)
    current = make_candle(97.0, 102.0, 96.0, 101.0, index=1)  # both bullish
    assert classify_engulfing(previous, current) is None


def test_doji_previous_candle_is_rejected() -> None:
    previous = make_candle(100.0, 101.0, 99.0, 100.0)         # zero-width body
    current = make_candle(100.5, 101.0, 98.0, 99.0, index=1)
    assert classify_engulfing(previous, current) is None


def test_zero_body_rejected_even_with_filter_disabled() -> None:
    """A flat bar has no direction, so it can never form either pattern."""
    previous = make_candle(100.0, 101.0, 99.0, 100.0)
    current = make_candle(100.5, 101.0, 98.0, 99.0, index=1)
    assert classify_engulfing(previous, current, min_body_ratio=0.0) is None


def test_thin_previous_body_is_filtered_by_default() -> None:
    """Body is 1% of range — mathematically engulfed, but meaningless."""
    previous = make_candle(100.0, 105.0, 95.0, 99.9)          # body 0.1 of range 10
    current = make_candle(99.5, 101.0, 99.0, 100.5, index=1)
    assert classify_engulfing(previous, current) is None
    # ...but it does satisfy the raw inequalities when the filter is off.
    assert (
        classify_engulfing(previous, current, min_body_ratio=0.0)
        is PatternType.BULLISH_ENGULFING
    )


def test_raising_the_threshold_rejects_marginal_bars() -> None:
    previous = make_candle(100.0, 101.0, 97.0, 98.0)          # body 2.0 / range 4.0 = 0.50
    current = make_candle(97.5, 101.5, 97.0, 100.5, index=1)  # body 3.0 / range 4.5 = 0.67
    assert classify_engulfing(previous, current, min_body_ratio=0.4) is PatternType.BULLISH_ENGULFING
    assert classify_engulfing(previous, current, min_body_ratio=0.6) is None


def test_inside_bar_is_not_engulfing() -> None:
    previous = make_candle(100.0, 105.0, 95.0, 96.0)
    current = make_candle(97.0, 99.0, 96.5, 98.0, index=1)    # body inside previous body
    assert classify_engulfing(previous, current) is None


# ---------------------------------------------------------------------------
# Bar geometry
# ---------------------------------------------------------------------------
def test_body_and_range_geometry() -> None:
    candle = make_candle(100.0, 105.0, 95.0, 103.0)
    assert candle.body == pytest.approx(3.0)
    assert candle.range == pytest.approx(10.0)
    assert candle.body_ratio == pytest.approx(0.3)
    assert candle.is_bullish and not candle.is_bearish


def test_flat_bar_has_no_direction() -> None:
    candle = make_candle(100.0, 100.0, 100.0, 100.0)
    assert not candle.is_bullish
    assert not candle.is_bearish
    assert candle.body_ratio == 0.0  # guards against divide-by-zero


def test_validate_ohlcv_rejects_frames_missing_columns() -> None:
    df = pd.DataFrame({"timestamp": [1], "open": [1.0]})
    with pytest.raises(ValueError, match="missing columns"):
        validate_ohlcv(df)


def test_validate_ohlcv_accepts_a_well_formed_frame() -> None:
    validate_ohlcv(make_df([(100.0, 101.0, 97.0, 98.0)]))
