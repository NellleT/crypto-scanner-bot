"""Unit tests for order blocks, displacement, premium/discount and CHoCH.

Fixtures are built so that exactly one rule varies between an accept case and
its matching reject case, so a failure names the rule that broke.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scanner.smc import (
    BEAR_GAP_COLUMN,
    BULL_GAP_COLUMN,
    STRUCTURE_LENGTH,
    Direction,
    RangeZone,
    SwingRange,
    detect_choch,
    detect_order_block,
    fvg_frame,
    order_block_mask,
    premium_discount_frame,
    swing_points,
    swing_range,
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


# --- Canonical structures: [-3] order block, [-2] impulse, [-1] confirmation.
# Gaps are ~4% of price, comfortably above the 0.30% displacement threshold.
BULL_OB = (99.0, 100.0, 95.0, 96.0)          # bearish, body 3 of range 5
BULL_IMPULSE = (96.0, 106.0, 95.5, 105.0)    # bullish displacement
BULL_CONFIRM = (105.0, 108.0, 104.0, 107.0)  # low 104 > 100 -> 4% FVG
BULL_CONFIRM_NO_FVG = (105.0, 108.0, 99.0, 107.0)   # low 99 <= 100, no gap
BULL_CONFIRM_TINY_FVG = (105.0, 108.0, 100.1, 107.0)  # gap 0.1 -> 0.10% only

BEAR_OB = (96.0, 100.0, 95.0, 99.0)          # bullish, body 3 of range 5
BEAR_IMPULSE = (99.0, 99.5, 89.0, 90.0)      # bearish displacement
BEAR_CONFIRM = (90.0, 91.0, 87.0, 88.0)      # high 91 < 95 -> 4.2% FVG
BEAR_CONFIRM_NO_FVG = (90.0, 96.0, 87.0, 88.0)      # high 96 >= 95, no gap
BEAR_CONFIRM_TINY_FVG = (90.0, 94.9, 87.0, 88.0)    # gap 0.1 -> 0.11% only


# ---------------------------------------------------------------------------
# Vectorised gap computation
# ---------------------------------------------------------------------------
def test_fvg_frame_computes_both_gap_series() -> None:
    frame = fvg_frame(make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM]))
    assert frame[BULL_GAP_COLUMN].iloc[-1] == pytest.approx(4.0)   # 104 - 100
    assert frame[BEAR_GAP_COLUMN].iloc[-1] == pytest.approx(-13.0)  # 95 - 108


def test_fvg_frame_leaves_the_warmup_rows_nan() -> None:
    frame = fvg_frame(make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM]))
    assert frame[BULL_GAP_COLUMN].iloc[:2].isna().all()


def test_vectorised_mask_agrees_with_the_scalar_detector() -> None:
    """The fast path and the live path must classify identically.

    Spatial filtering is off on both sides: the mask deliberately does not do
    premium/discount, so comparing with it enabled would compare two rules.
    """
    bars = [BULL_OB, BULL_IMPULSE, BULL_CONFIRM, BEAR_OB, BEAR_IMPULSE, BEAR_CONFIRM]
    df = make_df(bars)
    long_mask, short_mask = order_block_mask(df)

    for i in range(STRUCTURE_LENGTH, len(df) + 1):
        block, _ = detect_order_block(df.iloc[:i], require_extreme=False)
        row = i - 1
        expected_long = block is not None and block.direction is Direction.LONG
        expected_short = block is not None and block.direction is Direction.SHORT
        assert bool(long_mask.iloc[row]) == expected_long, f"long mismatch at {row}"
        assert bool(short_mask.iloc[row]) == expected_short, f"short mismatch at {row}"


# ---------------------------------------------------------------------------
# Displacement threshold (min_fvg_pct)
# ---------------------------------------------------------------------------
def test_wide_gap_passes_the_displacement_threshold() -> None:
    block, rejection = detect_order_block(make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM]))
    assert rejection is None
    assert block is not None
    assert block.fvg.pct == pytest.approx(4.0)  # 4 / 100


def test_narrow_gap_is_rejected_as_noise() -> None:
    """A real gap that is only 0.10% of price is spread, not displacement."""
    block, rejection = detect_order_block(
        make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM_TINY_FVG])
    )
    assert block is None
    assert rejection is not None
    assert rejection.stage == "displacement"
    assert "below the 0.3% displacement threshold" in rejection.reason


def test_narrow_short_gap_is_rejected_as_noise() -> None:
    block, rejection = detect_order_block(
        make_df([BEAR_OB, BEAR_IMPULSE, BEAR_CONFIRM_TINY_FVG])
    )
    assert block is None
    assert rejection is not None and rejection.stage == "displacement"


def test_displacement_threshold_is_configurable() -> None:
    """The same narrow gap passes once the threshold is lowered below it."""
    df = make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM_TINY_FVG])
    assert detect_order_block(df)[0] is None
    block, _ = detect_order_block(df, min_fvg_pct=0.05)
    assert block is not None
    assert block.fvg.pct == pytest.approx(0.1)


def test_missing_gap_is_reported_before_the_threshold() -> None:
    """No gap at all is a different failure from a gap that is too small."""
    _, rejection = detect_order_block(
        make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM_NO_FVG])
    )
    assert rejection is not None
    assert rejection.stage == "fvg"


def test_fvg_pct_is_measured_against_the_pre_displacement_extreme() -> None:
    block, _ = detect_order_block(make_df([BEAR_OB, BEAR_IMPULSE, BEAR_CONFIRM]))
    assert block is not None
    # gap 4 over the order block low of 95
    assert block.fvg.pct == pytest.approx(4.0 / 95.0 * 100.0)


# ---------------------------------------------------------------------------
# Swing range and premium/discount
# ---------------------------------------------------------------------------
def test_swing_range_spans_the_window_extremes() -> None:
    df = make_df([(10, 20, 5, 15), (15, 30, 12, 25), (25, 28, 8, 20)])
    found = swing_range(df, lookback=10)
    assert found is not None
    assert found.low == pytest.approx(5.0)
    assert found.high == pytest.approx(30.0)
    assert found.equilibrium == pytest.approx(17.5)


def test_swing_range_honours_the_lookback() -> None:
    """An old extreme outside the window must not define the current range."""
    df = make_df([(10, 100, 1, 50)] + [(20, 22, 18, 21)] * 5)
    recent = swing_range(df, lookback=3)
    assert recent is not None
    assert recent.high == pytest.approx(22.0)
    assert recent.low == pytest.approx(18.0)


def test_swing_range_is_none_when_flat() -> None:
    assert swing_range(make_df([(10, 10, 10, 10)] * 5)) is None


def test_fib_levels_split_the_range_at_the_midpoint() -> None:
    rng = SwingRange(low=100.0, high=200.0, lookback=50)
    assert rng.fib_level(100.0) == pytest.approx(0.0)
    assert rng.fib_level(150.0) == pytest.approx(0.5)
    assert rng.fib_level(200.0) == pytest.approx(1.0)
    assert rng.zone_of(120.0) is RangeZone.DISCOUNT
    assert rng.zone_of(180.0) is RangeZone.PREMIUM
    assert rng.zone_of(150.0) is RangeZone.EQUILIBRIUM


def test_zone_favours_the_matching_direction() -> None:
    assert RangeZone.DISCOUNT.favours(Direction.LONG)
    assert not RangeZone.DISCOUNT.favours(Direction.SHORT)
    assert RangeZone.PREMIUM.favours(Direction.SHORT)
    assert not RangeZone.PREMIUM.favours(Direction.LONG)
    # Exactly at equilibrium is neither half, so it favours nothing.
    assert not RangeZone.EQUILIBRIUM.favours(Direction.LONG)
    assert not RangeZone.EQUILIBRIUM.favours(Direction.SHORT)


def test_premium_discount_frame_is_vectorised_and_causal() -> None:
    df = make_df([(10, 20, 5, 15), (15, 30, 12, 25), (25, 28, 8, 20), (20, 24, 19, 23)])
    frame = premium_discount_frame(df, lookback=50)

    for column in ("range_high", "range_low", "equilibrium", "fib_level"):
        assert column in frame.columns
    # Row 1 sees only rows 0-1, so its range cannot include row 2's low of 8.
    assert frame["range_low"].iloc[1] == pytest.approx(5.0)
    assert frame["range_high"].iloc[1] == pytest.approx(30.0)
    assert frame["equilibrium"].iloc[1] == pytest.approx(17.5)
    assert frame["fib_level"].iloc[1] == pytest.approx((25 - 5) / 25)


# ---------------------------------------------------------------------------
# Spatial filtering of order blocks
# ---------------------------------------------------------------------------
def build_spatial_frame(
    structure: list[tuple[float, float, float, float]],
    *,
    range_low: float,
    range_high: float,
) -> pd.DataFrame:
    """Prepend two candles that pin the dealing range to a known band."""
    anchor = [
        (range_low, range_low + 0.1, range_low, range_low + 0.05),
        (range_high, range_high, range_high - 0.1, range_high - 0.05),
    ]
    return make_df(anchor + structure)


def test_long_in_discount_is_accepted() -> None:
    """OB high 100 against a 0-400 range sits at fib 0.25 — discount."""
    df = build_spatial_frame(
        [BULL_OB, BULL_IMPULSE, BULL_CONFIRM], range_low=0.0, range_high=400.0
    )
    block, rejection = detect_order_block(df, range_lookback=50)
    assert rejection is None, rejection
    assert block is not None
    assert block.zone is RangeZone.DISCOUNT


def test_long_in_premium_is_rejected() -> None:
    """Identical structure, range shifted so the same block is now expensive."""
    df = build_spatial_frame(
        [BULL_OB, BULL_IMPULSE, BULL_CONFIRM], range_low=80.0, range_high=108.0
    )
    block, rejection = detect_order_block(df, range_lookback=50)
    assert block is None
    assert rejection is not None
    assert rejection.stage == "premium_discount"
    assert "discount" in rejection.reason


def test_short_in_premium_is_accepted() -> None:
    df = build_spatial_frame(
        [BEAR_OB, BEAR_IMPULSE, BEAR_CONFIRM], range_low=50.0, range_high=101.0
    )
    block, rejection = detect_order_block(df, range_lookback=50)
    assert rejection is None, rejection
    assert block is not None
    assert block.zone is RangeZone.PREMIUM


def test_short_in_discount_is_rejected() -> None:
    df = build_spatial_frame(
        [BEAR_OB, BEAR_IMPULSE, BEAR_CONFIRM], range_low=87.0, range_high=400.0
    )
    block, rejection = detect_order_block(df, range_lookback=50)
    assert block is None
    assert rejection is not None and rejection.stage == "premium_discount"
    assert "premium" in rejection.reason


def test_spatial_filter_can_be_disabled() -> None:
    """The same mid-range block passes once extremes are not required."""
    df = build_spatial_frame(
        [BULL_OB, BULL_IMPULSE, BULL_CONFIRM], range_low=80.0, range_high=108.0
    )
    assert detect_order_block(df)[0] is None
    block, _ = detect_order_block(df, require_extreme=False)
    assert block is not None
    assert block.zone is RangeZone.PREMIUM  # still reported, just not enforced


def test_spatial_filter_tests_the_edge_nearest_equilibrium() -> None:
    """The whole block must be on the correct side, not merely overlap it."""
    # Range 0-200 puts equilibrium at 100, exactly the bullish block high.
    df = build_spatial_frame(
        [BULL_OB, BULL_IMPULSE, BULL_CONFIRM], range_low=0.0, range_high=200.0
    )
    block, rejection = detect_order_block(df, range_lookback=50)
    assert block is None
    assert rejection is not None and rejection.stage == "premium_discount"


# ---------------------------------------------------------------------------
# Swing pivots
# ---------------------------------------------------------------------------
def test_swing_points_find_centred_extremes() -> None:
    bars = [
        (10, 11, 9, 10),
        (10, 12, 9, 11),
        (11, 20, 10, 19),   # index 2: swing high
        (19, 20, 12, 13),
        (13, 14, 11, 12),
        (12, 13, 2, 3),     # index 5: swing low
        (3, 8, 2, 7),
        (7, 9, 6, 8),
    ]
    is_high, is_low = swing_points(make_df(bars), strength=2)
    assert bool(is_high.iloc[2])
    assert bool(is_low.iloc[5])


def test_swing_points_leave_the_newest_bars_unconfirmed() -> None:
    """A pivot needs bars on both sides; the tail cannot be confirmed yet.

    This is what makes look-ahead bias impossible by construction.
    """
    df = make_df([(10, 11, 9, 10)] * 10)
    is_high, is_low = swing_points(df, strength=2)
    assert not bool(is_high.iloc[-1])
    assert not bool(is_low.iloc[-1])


def test_swing_strength_must_be_positive() -> None:
    with pytest.raises(ValueError, match="strength"):
        swing_points(make_df([(1, 2, 0, 1)] * 5), strength=0)


# ---------------------------------------------------------------------------
# Change of Character
# ---------------------------------------------------------------------------
def downtrend_then_break() -> pd.DataFrame:
    """Lower highs, then a close above the most recent swing high."""
    return make_df(
        [
            (100, 102, 98, 99),
            (99, 105, 97, 100),   # swing high 105
            (100, 101, 95, 96),
            (96, 98, 92, 93),
            (93, 99, 91, 95),     # swing high 99 (lower than 105)
            (95, 96, 90, 91),
            (91, 93, 88, 92),
            (92, 101, 91, 100),   # closes above 99 -> bullish CHoCH
        ]
    )


def uptrend_then_break() -> pd.DataFrame:
    """Higher lows, then a close below the most recent swing low."""
    return make_df(
        [
            (100, 102, 98, 101),
            (101, 103, 95, 102),  # swing low 95
            (102, 106, 100, 105),
            (105, 108, 103, 107),
            (107, 109, 99, 108),  # swing low 99 (higher than 95)
            (108, 110, 105, 109),
            (109, 111, 107, 110),
            (110, 111, 96, 97),   # closes below 99 -> bearish CHoCH
        ]
    )


def test_bullish_choch_after_a_lower_high_sequence() -> None:
    assert detect_choch(downtrend_then_break(), strength=1) is Direction.LONG


def test_bearish_choch_after_a_higher_low_sequence() -> None:
    assert detect_choch(uptrend_then_break(), strength=1) is Direction.SHORT


def test_no_choch_without_a_break() -> None:
    """The same descending structure, but price never reclaims the swing high."""
    df = downtrend_then_break()
    df.loc[df.index[-1], ["high", "close"]] = [97.0, 96.0]
    assert detect_choch(df, strength=1) is None


def test_no_choch_when_the_structure_was_not_trending() -> None:
    """Breaking a higher high is continuation, not a change of character."""
    df = make_df(
        [
            (100, 101, 99, 100),
            (100, 103, 99, 102),   # swing high 103
            (102, 103, 100, 101),
            (101, 106, 100, 105),  # swing high 106 — HIGHER, so an uptrend
            (105, 106, 103, 104),
            (104, 108, 103, 107),  # breaks above 106, but with the trend
        ]
    )
    assert detect_choch(df, strength=1) is not Direction.LONG


def test_choch_needs_enough_history() -> None:
    assert detect_choch(make_df([(1, 2, 0, 1)] * 3), strength=2) is None


# ---------------------------------------------------------------------------
# Structural rejections
# ---------------------------------------------------------------------------
def test_block_must_oppose_the_impulse() -> None:
    bullish_before = (95.0, 100.0, 94.0, 99.0)
    block, rejection = detect_order_block(
        make_df([bullish_before, BULL_IMPULSE, BULL_CONFIRM])
    )
    assert block is None
    assert rejection is not None and rejection.stage == "order_block"


def test_doji_block_is_rejected() -> None:
    doji = (99.0, 104.0, 94.0, 99.02)
    block, rejection = detect_order_block(make_df([doji, BULL_IMPULSE, BULL_CONFIRM]))
    assert block is None
    assert rejection is not None and rejection.stage == "order_block"


def test_too_few_candles_reports_warmup() -> None:
    block, rejection = detect_order_block(make_df([BULL_IMPULSE, BULL_CONFIRM]))
    assert block is None
    assert rejection is not None and rejection.stage == "warmup"


def test_missing_columns_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        detect_order_block(pd.DataFrame({"open": [1.0], "close": [2.0]}))


def test_proximal_and_distal_edges_follow_direction() -> None:
    long_block, _ = detect_order_block(
        make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM]), require_extreme=False
    )
    short_block, _ = detect_order_block(
        make_df([BEAR_OB, BEAR_IMPULSE, BEAR_CONFIRM]), require_extreme=False
    )
    assert long_block is not None and short_block is not None

    assert long_block.proximal == pytest.approx(BULL_OB[1])   # high
    assert long_block.distal == pytest.approx(BULL_OB[2])     # low
    assert short_block.proximal == pytest.approx(BEAR_OB[2])  # low
    assert short_block.distal == pytest.approx(BEAR_OB[1])    # high


def test_block_contains_reports_zone_membership() -> None:
    block, _ = detect_order_block(
        make_df([BULL_OB, BULL_IMPULSE, BULL_CONFIRM]), require_extreme=False
    )
    assert block is not None
    assert block.contains(97.5)
    assert not block.contains(120.0)
