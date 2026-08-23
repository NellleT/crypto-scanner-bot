"""Unit tests for order block detection and fair value gap validation.

The fixtures are built so that the FVG is the *only* thing varying between the
accept and reject cases — the order block structure is identical in both.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scanner.smc import (
    BEAR_GAP_COLUMN,
    BULL_GAP_COLUMN,
    STRUCTURE_LENGTH,
    Direction,
    detect_order_block,
    fvg_frame,
    order_block_mask,
)

_HOUR_MS = 3_600_000


def make_df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """bars are (open, high, low, close)."""
    return pd.DataFrame(
        {
            "timestamp": [1_700_000_000_000 + i * _HOUR_MS for i in range(len(bars))],
            "open": [b[0] for b in bars],
            "high": [b[1] for b in bars],
            "low": [b[2] for b in bars],
            "close": [b[3] for b in bars],
            "volume": [100.0] * len(bars),
        }
    )


# --- Canonical structures -------------------------------------------------
# [-3] order block, [-2] impulse, [-1] confirmation.

# Bullish: bearish OB, bullish impulse, confirmation low (104) > OB high (100).
BULL_OB = (99.0, 100.0, 95.0, 96.0)      # bearish, body 3 of range 5
BULL_IMPULSE = (96.0, 106.0, 95.5, 105.0)  # bullish displacement
BULL_CONFIRM = (105.0, 108.0, 104.0, 107.0)  # low 104 > 100 -> FVG

# Same structure, confirmation low dropped to 99 -> 99 <= 100, no gap.
BULL_CONFIRM_NO_FVG = (105.0, 108.0, 99.0, 107.0)

# Bearish: bullish OB, bearish impulse, confirmation high (91) < OB low (95).
BEAR_OB = (96.0, 100.0, 95.0, 99.0)       # bullish, body 3 of range 5
BEAR_IMPULSE = (99.0, 99.5, 89.0, 90.0)   # bearish displacement
BEAR_CONFIRM = (90.0, 91.0, 87.0, 88.0)   # high 91 < 95 -> FVG
BEAR_CONFIRM_NO_FVG = (90.0, 96.0, 87.0, 88.0)  # high 96 >= 95, no gap

FILLER = (50.0, 51.0, 49.0, 50.5)


# ---------------------------------------------------------------------------
# Vectorised gap computation
# ---------------------------------------------------------------------------
def test_fvg_frame_computes_both_gap_series() -> None:
    df = make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM])
    frame = fvg_frame(df)

    assert BULL_GAP_COLUMN in frame.columns
    assert BEAR_GAP_COLUMN in frame.columns
    # low[-1] - high[-3] = 104 - 100
    assert frame[BULL_GAP_COLUMN].iloc[-1] == pytest.approx(4.0)
    # low[-3] - high[-1] = 95 - 108
    assert frame[BEAR_GAP_COLUMN].iloc[-1] == pytest.approx(-13.0)


def test_fvg_frame_leaves_the_warmup_rows_nan() -> None:
    frame = fvg_frame(make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM]))
    assert frame[BULL_GAP_COLUMN].iloc[:2].isna().all()
    assert frame[BULL_GAP_COLUMN].notna().sum() == 1


def test_fvg_frame_does_not_mutate_the_input() -> None:
    df = make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM])
    original = list(df.columns)
    fvg_frame(df)
    assert list(df.columns) == original


def test_vectorised_mask_agrees_with_the_scalar_detector() -> None:
    """The fast path and the live path must classify identically."""
    bars = [FILLER, BULL_OB, BULL_IMPULSE, BULL_CONFIRM, FILLER,
            BEAR_OB, BEAR_IMPULSE, BEAR_CONFIRM]
    df = make_df(bars)
    long_mask, short_mask = order_block_mask(df)

    for i in range(STRUCTURE_LENGTH, len(df) + 1):
        block, _ = detect_order_block(df.iloc[:i])
        row = i - 1
        expected_long = block is not None and block.direction is Direction.LONG
        expected_short = block is not None and block.direction is Direction.SHORT
        assert bool(long_mask.iloc[row]) == expected_long, f"long mismatch at {row}"
        assert bool(short_mask.iloc[row]) == expected_short, f"short mismatch at {row}"


# ---------------------------------------------------------------------------
# Accepted structures
# ---------------------------------------------------------------------------
def test_bullish_order_block_with_fvg_is_accepted() -> None:
    block, rejection = detect_order_block(make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM]))
    assert rejection is None
    assert block is not None
    assert block.direction is Direction.LONG
    assert block.candle.is_bearish
    assert block.impulse.is_bullish
    assert block.fvg.is_bullish
    assert block.fvg.size == pytest.approx(4.0)  # 104 - 100


def test_bearish_order_block_with_fvg_is_accepted() -> None:
    block, rejection = detect_order_block(make_df([BEAR_OB, BEAR_IMPULSE, BEAR_CONFIRM]))
    assert rejection is None
    assert block is not None
    assert block.direction is Direction.SHORT
    assert block.candle.is_bullish
    assert block.impulse.is_bearish
    assert not block.fvg.is_bullish
    assert block.fvg.size == pytest.approx(4.0)  # 95 - 91


def test_proximal_and_distal_edges_follow_direction() -> None:
    long_block, _ = detect_order_block(make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM]))
    short_block, _ = detect_order_block(make_df([BEAR_OB, BEAR_IMPULSE, BEAR_CONFIRM]))
    assert long_block is not None and short_block is not None

    # Long: price falls back into the block from above, so proximal is the top.
    assert long_block.proximal == pytest.approx(BULL_OB[1])   # high
    assert long_block.distal == pytest.approx(BULL_OB[2])     # low
    # Short: price rallies back into it from below.
    assert short_block.proximal == pytest.approx(BEAR_OB[2])  # low
    assert short_block.distal == pytest.approx(BEAR_OB[1])    # high


def test_only_the_newest_three_candles_are_considered() -> None:
    """Older structures must not leak into the current evaluation."""
    df = make_df([BEAR_OB, BEAR_IMPULSE, BEAR_CONFIRM, BULL_OB, BULL_IMPULSE, BULL_CONFIRM])
    block, _ = detect_order_block(df)
    assert block is not None
    assert block.direction is Direction.LONG
    assert block.candle.timestamp == int(df["timestamp"].iloc[-3])


# ---------------------------------------------------------------------------
# FVG rejection — the critical validation rule
# ---------------------------------------------------------------------------
def test_bullish_order_block_without_fvg_is_rejected() -> None:
    """Identical structure to the accepted case; only the gap is missing."""
    block, rejection = detect_order_block(
        make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM_NO_FVG])
    )
    assert block is None
    assert rejection is not None
    assert rejection.stage == "fvg"
    assert "no fair value gap" in rejection.reason


def test_bearish_order_block_without_fvg_is_rejected() -> None:
    block, rejection = detect_order_block(
        make_df([BEAR_OB, BEAR_IMPULSE, BEAR_CONFIRM_NO_FVG])
    )
    assert block is None
    assert rejection is not None
    assert rejection.stage == "fvg"


def test_fvg_comparison_is_strict_for_longs() -> None:
    """low[-1] exactly equal to high[-3] is not a gap — nothing was left behind."""
    touching = (105.0, 108.0, BULL_OB[1], 107.0)  # low == OB high == 100
    block, rejection = detect_order_block(make_df([BULL_OB, BULL_IMPULSE, touching]))
    assert block is None
    assert rejection is not None and rejection.stage == "fvg"


def test_fvg_comparison_is_strict_for_shorts() -> None:
    touching = (90.0, BEAR_OB[2], 87.0, 88.0)  # high == OB low == 95
    block, rejection = detect_order_block(make_df([BEAR_OB, BEAR_IMPULSE, touching]))
    assert block is None
    assert rejection is not None and rejection.stage == "fvg"


def test_one_tick_gap_is_enough() -> None:
    """The rule is strict inequality, not a minimum size."""
    barely = (105.0, 108.0, BULL_OB[1] + 0.01, 107.0)
    block, _ = detect_order_block(make_df([BULL_OB, BULL_IMPULSE, barely]))
    assert block is not None
    assert block.fvg.size == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Structural rejections
# ---------------------------------------------------------------------------
def test_block_must_oppose_the_impulse() -> None:
    """A bullish candle before a bullish impulse is continuation, not a block."""
    bullish_before = (95.0, 100.0, 94.0, 99.0)
    block, rejection = detect_order_block(
        make_df([bullish_before, BULL_IMPULSE, BULL_CONFIRM])
    )
    assert block is None
    assert rejection is not None and rejection.stage == "order_block"


def test_doji_block_or_impulse_is_rejected() -> None:
    doji = (99.0, 104.0, 94.0, 99.02)  # body 0.02 of range 10
    block, rejection = detect_order_block(make_df([doji, BULL_IMPULSE, BULL_CONFIRM]))
    assert block is None
    assert rejection is not None and rejection.stage == "order_block"


def test_doji_filter_can_be_relaxed() -> None:
    doji = (99.0, 100.0, 94.0, 98.9)
    block, _ = detect_order_block(
        make_df([doji, BULL_IMPULSE, BULL_CONFIRM]), min_body_ratio=0.0
    )
    assert block is not None


def test_too_few_candles_reports_warmup() -> None:
    block, rejection = detect_order_block(make_df([BULL_IMPULSE, BULL_CONFIRM]))
    assert block is None
    assert rejection is not None and rejection.stage == "warmup"


def test_missing_columns_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        detect_order_block(pd.DataFrame({"open": [1.0], "close": [2.0]}))
