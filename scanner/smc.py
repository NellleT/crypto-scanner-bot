"""Smart Money Concepts: order blocks, displacement, spatial context, CHoCH.

The three-candle structure this module looks for, indexed from the end of a
frame of CLOSED candles:

===========  ========  ==================================================
Index        Role      Requirement (bullish setup)
===========  ========  ==================================================
``[-3]``     Order     Bearish candle — the last down-close before the
             Block     impulse. Its range becomes the entry zone.
``[-2]``     Impulse   Bullish displacement candle.
``[-1]``     Confirm   Closes the structure; its low defines the gap.
===========  ========  ==================================================

v3.1 tightens what counts as tradable in two ways:

* **Displacement threshold** — a gap must be at least ``min_fvg_pct`` of price.
  A one-tick inefficiency is noise, not institutional displacement.
* **Spatial context** — a long is only taken from the *discount* half of the
  current dealing range and a short only from the *premium* half, so the engine
  stops buying the middle of a range at an average price.

All gap and range maths is vectorised: two shifted subtractions for the gaps,
rolling extremes for the range, centred rolling extremes for swing pivots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Final

import pandas as pd

from scanner.candles import DEFAULT_MIN_BODY_RATIO, Candle, validate_ohlcv

logger: Final[logging.Logger] = logging.getLogger(__name__)

#: Candles required to describe one order block: OB, impulse, confirmation.
STRUCTURE_LENGTH: Final[int] = 3

#: Column names written by :func:`fvg_frame`.
BULL_GAP_COLUMN: Final[str] = "fvg_bull_gap"
BEAR_GAP_COLUMN: Final[str] = "fvg_bear_gap"

#: Minimum fair value gap as a percentage of the pre-displacement extreme.
#: Below this the "gap" is spread and tick noise rather than displacement.
DEFAULT_MIN_FVG_PCT: Final[float] = 0.30

#: Candles forming the dealing range that premium/discount is measured against.
DEFAULT_RANGE_LOOKBACK: Final[int] = 50

#: Bars either side of a pivot required to confirm it as a swing point.
DEFAULT_SWING_STRENGTH: Final[int] = 2


class Direction(str, Enum):
    """Trade direction implied by an order block."""

    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def is_long(self) -> bool:
        return self is Direction.LONG

    @property
    def emoji(self) -> str:
        return "🟢" if self.is_long else "🔴"

    @property
    def binance_side(self) -> str:
        """Order side as the Binance REST API expects it."""
        return "BUY" if self.is_long else "SELL"

    @property
    def opposite(self) -> "Direction":
        return Direction.SHORT if self.is_long else Direction.LONG


class RangeZone(str, Enum):
    """Half of the dealing range a price sits in."""

    DISCOUNT = "discount"        # below equilibrium — where longs are cheap
    PREMIUM = "premium"          # above equilibrium — where shorts are dear
    EQUILIBRIUM = "equilibrium"  # exactly at the midpoint

    def favours(self, direction: Direction) -> bool:
        """True when this half is the right side of the range for ``direction``."""
        if direction.is_long:
            return self is RangeZone.DISCOUNT
        return self is RangeZone.PREMIUM


@dataclass(frozen=True, slots=True)
class SwingRange:
    """The dealing range premium/discount is measured against."""

    low: float
    high: float
    lookback: int

    @property
    def size(self) -> float:
        return max(self.high - self.low, 0.0)

    @property
    def equilibrium(self) -> float:
        """The 0.5 Fibonacci level splitting premium from discount."""
        return (self.high + self.low) / 2.0

    @property
    def is_valid(self) -> bool:
        return self.size > 0.0

    def fib_level(self, price: float) -> float:
        """Where ``price`` sits in the range: 0.0 at the low, 1.0 at the high."""
        if not self.is_valid:
            return 0.5
        return (price - self.low) / self.size

    def zone_of(self, price: float) -> RangeZone:
        level = self.fib_level(price)
        if level < 0.5:
            return RangeZone.DISCOUNT
        if level > 0.5:
            return RangeZone.PREMIUM
        return RangeZone.EQUILIBRIUM


@dataclass(frozen=True, slots=True)
class FairValueGap:
    """An unfilled inefficiency left by a displacement candle."""

    is_bullish: bool
    bottom: float
    top: float

    @property
    def size(self) -> float:
        """Absolute height of the gap, in quote currency."""
        return max(self.top - self.bottom, 0.0)

    @property
    def pct(self) -> float:
        """Gap height as a percentage of the pre-displacement extreme.

        The denominator is the near edge of the gap — ``high[-3]`` for a bullish
        setup — which is the reference the displacement rule is written against.
        """
        reference = self.bottom if self.is_bullish else self.top
        if reference <= 0.0:
            return 0.0
        return self.size / reference * 100.0

    def size_pct(self, reference: float) -> float:
        """Gap height as a percentage of an arbitrary reference price."""
        if reference <= 0.0:
            return 0.0
        return self.size / reference * 100.0


@dataclass(frozen=True, slots=True)
class OrderBlock:
    """A validated order block: the zone, the impulse and the gap that proves it."""

    direction: Direction
    candle: Candle          # the order block itself — entry zone
    impulse: Candle         # the displacement candle
    confirmation: Candle    # the candle that completed the structure
    fvg: FairValueGap
    swing_range: SwingRange | None = None
    zone: RangeZone | None = None

    @property
    def proximal(self) -> float:
        """Edge price returns to first — where the limit order rests."""
        return self.candle.high if self.direction.is_long else self.candle.low

    @property
    def distal(self) -> float:
        """Far edge — beyond it the block has failed, so the stop sits here."""
        return self.candle.low if self.direction.is_long else self.candle.high

    @property
    def height(self) -> float:
        """Thickness of the zone."""
        return abs(self.proximal - self.distal)

    @property
    def open_time(self) -> datetime:
        return self.candle.open_time

    def contains(self, price: float) -> bool:
        """True when ``price`` is inside the block range."""
        return self.candle.low <= price <= self.candle.high

    @property
    def dedup_key(self) -> tuple[int, str]:
        """Identity of the structure, for suppressing repeat alerts."""
        return (self.candle.timestamp, self.direction.value)


def fvg_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with vectorised bullish/bearish gap columns attached.

    For each row *i* the columns describe the three-candle window ending there,
    i.e. candles ``i-2`` (order block), ``i-1`` (impulse) and ``i``:

    * ``fvg_bull_gap = low[i] - high[i-2]`` — positive means a bullish FVG.
    * ``fvg_bear_gap = low[i-2] - high[i]`` — positive means a bearish FVG.

    Two shifted subtractions over the whole column, so a 1000-candle frame costs
    one vectorised pass rather than 1000 Python-level comparisons. The first two
    rows are NaN, which compares false everywhere and needs no special casing.
    """
    validate_ohlcv(df)
    enriched = df.copy()
    pre_high = enriched["high"].shift(STRUCTURE_LENGTH - 1)
    pre_low = enriched["low"].shift(STRUCTURE_LENGTH - 1)
    enriched[BULL_GAP_COLUMN] = enriched["low"] - pre_high
    enriched[BEAR_GAP_COLUMN] = pre_low - enriched["high"]
    return enriched


def swing_points(
    df: pd.DataFrame, *, strength: int = DEFAULT_SWING_STRENGTH
) -> tuple["pd.Series[bool]", "pd.Series[bool]"]:
    """Vectorised swing pivots: ``(is_swing_high, is_swing_low)``.

    A pivot is the extreme of a window centred on it, so it needs ``strength``
    bars on *each* side. The centred rolling window leaves the newest
    ``strength`` bars NaN, which is correct rather than inconvenient — an
    unconfirmed pivot must never be treated as structure, and this makes
    look-ahead bias impossible by construction.
    """
    validate_ohlcv(df)
    if strength < 1:
        raise ValueError(f"Swing strength must be >= 1, got {strength}.")

    window = 2 * strength + 1
    rolling_high = df["high"].rolling(window, center=True).max()
    rolling_low = df["low"].rolling(window, center=True).min()
    is_high = (df["high"] >= rolling_high) & rolling_high.notna()
    is_low = (df["low"] <= rolling_low) & rolling_low.notna()
    return is_high.fillna(False), is_low.fillna(False)


def swing_range(
    df: pd.DataFrame, *, lookback: int = DEFAULT_RANGE_LOOKBACK
) -> SwingRange | None:
    """The dealing range over the last ``lookback`` closed candles.

    Taken as the extreme high and extreme low of the window — that *is* the
    swing high and swing low of the range being traded, and using rolling
    extremes keeps it vectorised and free of pivot-confirmation lag. Returns
    ``None`` when the window has no height to divide.
    """
    validate_ohlcv(df)
    if df.empty:
        return None

    window = df.tail(max(lookback, 2))
    high = float(window["high"].max())
    low = float(window["low"].min())
    if not (high > low):
        return None
    return SwingRange(low=low, high=high, lookback=len(window))


def premium_discount_frame(
    df: pd.DataFrame, *, lookback: int = DEFAULT_RANGE_LOOKBACK
) -> pd.DataFrame:
    """Vectorised premium/discount array across the whole frame.

    Adds ``range_low``, ``range_high``, ``equilibrium`` and ``fib_level`` (of
    the close) using trailing rolling extremes, so every row sees only the
    candles up to and including itself.
    """
    validate_ohlcv(df)
    enriched = df.copy()
    window = max(lookback, 2)
    enriched["range_high"] = enriched["high"].rolling(window, min_periods=2).max()
    enriched["range_low"] = enriched["low"].rolling(window, min_periods=2).min()
    span = enriched["range_high"] - enriched["range_low"]
    enriched["equilibrium"] = (enriched["range_high"] + enriched["range_low"]) / 2.0
    enriched["fib_level"] = (enriched["close"] - enriched["range_low"]) / span.where(
        span > 0
    )
    return enriched


def detect_choch(
    df: pd.DataFrame,
    *,
    strength: int = DEFAULT_SWING_STRENGTH,
    lookback: int = DEFAULT_RANGE_LOOKBACK,
) -> Direction | None:
    """Change of Character on the newest closed candle, or ``None``.

    A CHoCH is the first break *against* the prevailing short-term structure:

    * **Bullish** — swing highs were descending (a lower-high sequence, i.e. a
      downtrend) and the newest close breaks above the most recent swing high.
    * **Bearish** — the mirror image on swing lows.

    Requiring the prior sequence to be trending is what separates a genuine
    character change from an ordinary continuation break in an existing trend.
    """
    validate_ohlcv(df)
    if len(df) < 2 * strength + 2:
        return None

    window = df.tail(max(lookback, 2 * strength + 2))
    is_high, is_low = swing_points(window, strength=strength)

    close = float(window["close"].iloc[-1])
    highs = window.loc[is_high, "high"]
    lows = window.loc[is_low, "low"]

    if len(highs) >= 2:
        last, previous = float(highs.iloc[-1]), float(highs.iloc[-2])
        if last < previous and close > last:
            return Direction.LONG

    if len(lows) >= 2:
        last, previous = float(lows.iloc[-1]), float(lows.iloc[-2])
        if last > previous and close < last:
            return Direction.SHORT

    return None


def order_block_mask(
    df: pd.DataFrame,
    *,
    min_body_ratio: float = DEFAULT_MIN_BODY_RATIO,
    min_fvg_pct: float = DEFAULT_MIN_FVG_PCT,
):
    """Boolean masks of every valid long/short structure in ``df``.

    Vectorised counterpart to :func:`detect_order_block`, used for backtesting
    and diagnostics rather than live scanning. Returns ``(long_mask,
    short_mask)`` aligned to the index of ``df``, each true at the *confirmation*
    candle of a structure that clears both the body and displacement rules.
    Spatial filtering is applied separately by the caller.
    """
    frame = fvg_frame(df) if BULL_GAP_COLUMN not in df.columns else df

    body = (frame["close"] - frame["open"]).abs()
    rng = (frame["high"] - frame["low"]).replace(0.0, pd.NA)
    body_ratio = (body / rng).fillna(0.0)

    shift = STRUCTURE_LENGTH - 1
    ob_bearish = (frame["close"] < frame["open"]).shift(shift, fill_value=False)
    ob_bullish = (frame["close"] > frame["open"]).shift(shift, fill_value=False)
    ob_has_body = (body_ratio >= min_body_ratio).shift(shift, fill_value=False)

    impulse_bullish = (frame["close"] > frame["open"]).shift(1, fill_value=False)
    impulse_bearish = (frame["close"] < frame["open"]).shift(1, fill_value=False)
    impulse_has_body = (body_ratio >= min_body_ratio).shift(1, fill_value=False)

    pre_high = frame["high"].shift(shift)
    pre_low = frame["low"].shift(shift)
    bull_pct = frame[BULL_GAP_COLUMN] / pre_high.where(pre_high > 0) * 100.0
    bear_pct = frame[BEAR_GAP_COLUMN] / pre_low.where(pre_low > 0) * 100.0

    long_mask = (
        ob_bearish
        & ob_has_body
        & impulse_bullish
        & impulse_has_body
        & (frame[BULL_GAP_COLUMN] > 0.0)
        & (bull_pct >= min_fvg_pct)
    )
    short_mask = (
        ob_bullish
        & ob_has_body
        & impulse_bearish
        & impulse_has_body
        & (frame[BEAR_GAP_COLUMN] > 0.0)
        & (bear_pct >= min_fvg_pct)
    )
    return long_mask.fillna(False), short_mask.fillna(False)


@dataclass(frozen=True, slots=True)
class StructureRejection:
    """Why the newest three candles do not form a tradable order block."""

    stage: str
    reason: str


def detect_order_block(
    df: pd.DataFrame,
    *,
    min_body_ratio: float = DEFAULT_MIN_BODY_RATIO,
    min_fvg_pct: float = DEFAULT_MIN_FVG_PCT,
    range_lookback: int = DEFAULT_RANGE_LOOKBACK,
    require_extreme: bool = True,
) -> tuple[OrderBlock | None, StructureRejection | None]:
    """Evaluate the newest three CLOSED candles for a tradable order block.

    ``df`` must contain only closed candles and end at the confirmation candle.
    Returns ``(order_block, None)`` on success or ``(None, rejection)`` with the
    stage that failed, so the caller can report a funnel.

    Stages, in order: ``warmup``, ``order_block``, ``fvg``, ``displacement``,
    ``premium_discount``.
    """
    validate_ohlcv(df)

    if len(df) < STRUCTURE_LENGTH:
        return None, StructureRejection(
            "warmup",
            f"need {STRUCTURE_LENGTH} closed candles for an order block, have {len(df)}",
        )

    block = Candle.from_row(df.iloc[-3])
    impulse = Candle.from_row(df.iloc[-2])
    confirmation = Candle.from_row(df.iloc[-1])

    # 1. Structure — a block is the last opposing candle before the impulse.
    if block.is_bearish and impulse.is_bullish:
        direction = Direction.LONG
    elif block.is_bullish and impulse.is_bearish:
        direction = Direction.SHORT
    else:
        return None, StructureRejection(
            "order_block",
            "no order block: the candle before the impulse does not oppose it",
        )

    if block.body_ratio < min_body_ratio or impulse.body_ratio < min_body_ratio:
        return None, StructureRejection(
            "order_block",
            f"order block or impulse is doji-like (bodies "
            f"{block.body_ratio:.3f}/{impulse.body_ratio:.3f} of range, "
            f"minimum {min_body_ratio:g})",
        )

    # 2. Fair Value Gap — the displacement must leave an inefficiency behind.
    if direction is Direction.LONG:
        gap = confirmation.low - block.high
        fvg = FairValueGap(is_bullish=True, bottom=block.high, top=confirmation.low)
        detail = f"low[-1] {confirmation.low:g} vs high[-3] {block.high:g}"
    else:
        gap = block.low - confirmation.high
        fvg = FairValueGap(is_bullish=False, bottom=confirmation.high, top=block.low)
        detail = f"high[-1] {confirmation.high:g} vs low[-3] {block.low:g}"

    if gap <= 0.0:
        return None, StructureRejection(
            "fvg",
            f"{direction.value} order block rejected: no fair value gap ({detail}, "
            f"gap {gap:g}) — displacement left no inefficiency",
        )

    # 3. Displacement threshold — the gap must be wide enough to be institutional.
    if fvg.pct < min_fvg_pct:
        return None, StructureRejection(
            "displacement",
            f"{direction.value} order block rejected: fair value gap is "
            f"{fvg.pct:.3f}% of price, below the {min_fvg_pct:g}% displacement "
            "threshold — noise rather than a move with size behind it",
        )

    # 4. Spatial context — only take extremes, never the middle of the range.
    dealing_range = swing_range(df, lookback=range_lookback)
    zone: RangeZone | None = None
    if dealing_range is not None:
        # The whole block must sit on the correct side, so test the edge nearest
        # equilibrium: the high of a bullish block, the low of a bearish one.
        boundary = block.high if direction.is_long else block.low
        zone = dealing_range.zone_of(boundary)
        if require_extreme and not zone.favours(direction):
            wanted = "discount" if direction.is_long else "premium"
            side = "longs" if direction.is_long else "shorts"
            return None, StructureRejection(
                "premium_discount",
                f"{direction.value} order block rejected: block sits in the "
                f"{zone.value} half at fib {dealing_range.fib_level(boundary):.3f} "
                f"of the {dealing_range.lookback}-bar range "
                f"[{dealing_range.low:g}, {dealing_range.high:g}] — "
                f"{side} are only taken from {wanted}",
            )
    elif require_extreme:
        return None, StructureRejection(
            "premium_discount",
            "no dealing range: the lookback window has no height to divide",
        )

    return (
        OrderBlock(
            direction=direction,
            candle=block,
            impulse=impulse,
            confirmation=confirmation,
            fvg=fvg,
            swing_range=dealing_range,
            zone=zone,
        ),
        None,
    )
