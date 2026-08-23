"""Smart Money Concepts: Order Blocks validated by Fair Value Gaps.

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

The Fair Value Gap is the validation: an order block is only tradable if the
displacement left an unfilled inefficiency behind it.

    bullish FVG:  ``low[-1] > high[-3]``
    bearish FVG:  ``high[-1] < low[-3]``

Both are strict. Equality means price traded through the whole range with no
gap left behind, so there is no inefficiency and no displacement to trade.

Gap detection is vectorised: :func:`fvg_frame` computes both gap series across
an entire frame with two shifted subtractions, so scanning many pairs costs one
pass each rather than a Python loop over candles.
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

    def size_pct(self, reference: float) -> float:
        """Gap height as a percentage of a reference price."""
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

    @property
    def proximal(self) -> float:
        """Edge price returns to first — where the limit order rests.

        The top of a bullish block (price falls back into it from above), the
        bottom of a bearish block.
        """
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


def order_block_mask(df: pd.DataFrame, *, min_body_ratio: float = DEFAULT_MIN_BODY_RATIO):
    """Boolean masks of every valid long/short structure in ``df``.

    Vectorised counterpart to :func:`detect_order_block`, used for backtesting
    and diagnostics rather than live scanning. Returns ``(long_mask,
    short_mask)`` aligned to ``df``'s index, each true at the *confirmation*
    candle of a valid structure.
    """
    frame = fvg_frame(df) if BULL_GAP_COLUMN not in df.columns else df

    body = (frame["close"] - frame["open"]).abs()
    rng = (frame["high"] - frame["low"]).replace(0.0, pd.NA)
    body_ratio = (body / rng).fillna(0.0)

    ob_bearish = (frame["close"] < frame["open"]).shift(STRUCTURE_LENGTH - 1, fill_value=False)
    ob_bullish = (frame["close"] > frame["open"]).shift(STRUCTURE_LENGTH - 1, fill_value=False)
    ob_has_body = (body_ratio >= min_body_ratio).shift(STRUCTURE_LENGTH - 1, fill_value=False)

    impulse_bullish = (frame["close"] > frame["open"]).shift(1, fill_value=False)
    impulse_bearish = (frame["close"] < frame["open"]).shift(1, fill_value=False)
    impulse_has_body = (body_ratio >= min_body_ratio).shift(1, fill_value=False)

    long_mask = (
        ob_bearish & ob_has_body & impulse_bullish & impulse_has_body
        & (frame[BULL_GAP_COLUMN] > 0.0)
    )
    short_mask = (
        ob_bullish & ob_has_body & impulse_bearish & impulse_has_body
        & (frame[BEAR_GAP_COLUMN] > 0.0)
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
) -> tuple[OrderBlock | None, StructureRejection | None]:
    """Evaluate the newest three CLOSED candles for a validated order block.

    ``df`` must contain only closed candles and end at the confirmation candle.
    Returns ``(order_block, None)`` on success or ``(None, rejection)`` with the
    stage that failed, so the caller can report a funnel.
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

    return (
        OrderBlock(
            direction=direction,
            candle=block,
            impulse=impulse,
            confirmation=confirmation,
            fvg=fvg,
        ),
        None,
    )
