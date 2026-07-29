"""Unit tests for indicator calculation.

These assert the properties the strategy actually relies on: exact SMA values,
correct warm-up (no partial averages leaking through), and equivalence between
the pandas-ta backend and the pandas fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scanner.indicators import (
    PANDAS_TA_AVAILABLE,
    add_indicators,
    required_candles,
    sma,
    trend_column,
    volume_column,
)

_MINUTE_MS = 60_000


def make_df(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    volumes = volumes if volumes is not None else [100.0] * len(closes)
    return pd.DataFrame(
        {
            "timestamp": [1_700_000_000_000 + i * _MINUTE_MS for i in range(len(closes))],
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": volumes,
        }
    )


# ---------------------------------------------------------------------------
# SMA correctness
# ---------------------------------------------------------------------------
def test_sma_matches_hand_computed_average() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(series, 3)
    assert result.iloc[2] == pytest.approx(2.0)  # (1+2+3)/3
    assert result.iloc[3] == pytest.approx(3.0)  # (2+3+4)/3
    assert result.iloc[4] == pytest.approx(4.0)  # (3+4+5)/3


def test_sma_is_nan_until_fully_warmed_up() -> None:
    """A partial average must never leak through — it would misstate the trend."""
    result = sma(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert result.iloc[:2].isna().all()
    assert result.notna().sum() == 3
    assert int(result.first_valid_index()) == 2


def test_sma_all_nan_when_series_shorter_than_period() -> None:
    result = sma(pd.Series([1.0, 2.0, 3.0]), 200)
    assert result.isna().all()


def test_sma_rejects_invalid_length() -> None:
    with pytest.raises(ValueError, match="length must be"):
        sma(pd.Series([1.0, 2.0]), 0)


def test_sma_equals_pandas_rolling_mean() -> None:
    """The two backends must agree exactly, whichever one is active."""
    rng = np.random.default_rng(7)
    series = pd.Series(100 + rng.normal(0, 2, 400).cumsum())
    reference = series.rolling(window=200, min_periods=200).mean()
    result = sma(series, 200)

    both = result.notna() & reference.notna()
    assert both.sum() == 201
    assert float((result[both] - reference[both]).abs().max()) < 1e-9


@pytest.mark.skipif(not PANDAS_TA_AVAILABLE, reason="pandas-ta is not installed")
def test_pandas_ta_backend_matches_reference() -> None:
    """Runs only where pandas-ta is importable; skipped otherwise."""
    import scanner.indicators as indicators

    rng = np.random.default_rng(11)
    series = pd.Series(50 + rng.normal(0, 1, 300).cumsum())

    from_library = indicators._PTA.sma(series, length=50)  # type: ignore[union-attr]
    reference = series.rolling(window=50, min_periods=50).mean()

    both = from_library.notna() & reference.notna()
    assert both.sum() > 0
    assert float((from_library[both] - reference[both]).abs().max()) < 1e-9


# ---------------------------------------------------------------------------
# Frame enrichment
# ---------------------------------------------------------------------------
def test_add_indicators_attaches_named_columns() -> None:
    df = make_df([float(i) for i in range(250)])
    enriched = add_indicators(df, sma_period=200, volume_sma_period=20)

    assert "SMA_200" in enriched.columns
    assert "VOL_SMA_20" in enriched.columns
    assert enriched["SMA_200"].notna().sum() == 51
    assert enriched["VOL_SMA_20"].notna().sum() == 231


def test_add_indicators_does_not_mutate_the_input() -> None:
    df = make_df([float(i) for i in range(60)])
    original_columns = list(df.columns)
    add_indicators(df, sma_period=50, volume_sma_period=20)
    assert list(df.columns) == original_columns


def test_add_indicators_honours_custom_periods() -> None:
    df = make_df([float(i) for i in range(60)])
    enriched = add_indicators(df, sma_period=50, volume_sma_period=10)
    assert "SMA_50" in enriched.columns
    assert "VOL_SMA_10" in enriched.columns


def test_add_indicators_requires_close_and_volume() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing 'volume'"):
        add_indicators(df)


def test_volume_sma_averages_volume_not_price() -> None:
    df = make_df(closes=[10.0] * 30, volumes=[float(i) for i in range(30)])
    enriched = add_indicators(df, sma_period=5, volume_sma_period=10)
    # Mean of volumes 20..29 inclusive.
    assert enriched["VOL_SMA_10"].iloc[-1] == pytest.approx(24.5)


# ---------------------------------------------------------------------------
# Sizing helpers
# ---------------------------------------------------------------------------
def test_required_candles_accounts_for_the_pattern_bar() -> None:
    assert required_candles(200, 20) == 201
    assert required_candles(20, 200) == 201  # driven by the longer period
    assert required_candles(50, 20) == 51


def test_column_name_helpers() -> None:
    assert trend_column(200) == "SMA_200"
    assert trend_column(50) == "SMA_50"
    assert volume_column(20) == "VOL_SMA_20"
