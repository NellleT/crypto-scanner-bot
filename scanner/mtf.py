"""Multi-timeframe confirmation: the lower-timeframe entry trigger.

A tagged HTF zone is a *location*, not a trade. This module decides whether the
lower timeframe agrees, which is the difference between buying a level because
price arrived there and buying it because price arrived and then turned.

Confirmation requires, in order:

1. price trading inside the HTF zone on the LTF frame;
2. an LTF **Change of Character** in the direction of the HTF setup — the first
   break against the short-term structure that carried price into the zone;
3. an LTF **Fair Value Gap** in the same direction, formed at or after the
   CHoCH, evidencing that the turn had displacement behind it.

The CHoCH bar is located by replaying :func:`~scanner.smc.detect_choch` over the
tail of the frame rather than by re-deriving pivot logic here, so the live path
and the confirmation path can never disagree about what a CHoCH is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

import pandas as pd

from scanner.smc import (
    BEAR_GAP_COLUMN,
    BULL_GAP_COLUMN,
    DEFAULT_SWING_STRENGTH,
    STRUCTURE_LENGTH,
    Direction,
    detect_choch,
    fvg_frame,
)
from scanner.watchlist import WatchedZone

logger: Final[logging.Logger] = logging.getLogger(__name__)

#: LTF candles searched for a confirmation sequence.
DEFAULT_CONFIRM_WINDOW: Final[int] = 30

#: Minimum LTF gap, in percent. Zero accepts any genuine gap: the displacement
#: filter that matters is applied on the HTF, and demanding a second large gap
#: on a 15m chart would reject most valid entries.
DEFAULT_LTF_MIN_FVG_PCT: Final[float] = 0.0


@dataclass(frozen=True, slots=True)
class LtfTrigger:
    """A confirmed lower-timeframe entry trigger."""

    direction: Direction
    timeframe: str
    choch_timestamp: int
    fvg_timestamp: int
    fvg_pct: float
    price: float

    @property
    def choch_at(self) -> datetime:
        return datetime.fromtimestamp(self.choch_timestamp / 1000, tz=timezone.utc)

    @property
    def confirmed_at(self) -> datetime:
        return datetime.fromtimestamp(self.fvg_timestamp / 1000, tz=timezone.utc)


@dataclass(frozen=True, slots=True)
class ConfirmationRejection:
    """Why the lower timeframe has not confirmed a tagged zone."""

    stage: str   # in_zone | choch | ltf_fvg
    reason: str


def find_choch(
    df: pd.DataFrame,
    direction: Direction,
    *,
    strength: int = DEFAULT_SWING_STRENGTH,
    window: int = DEFAULT_CONFIRM_WINDOW,
    not_before_ms: int | None = None,
) -> int | None:
    """Positional index of the most recent CHoCH in ``direction``, or ``None``.

    Found by replaying the detector bar by bar over the tail, so the definition
    of a CHoCH lives in exactly one place. The window is small, so the repeated
    evaluation costs far less than maintaining a second implementation would.

    ``not_before_ms`` restricts which bar may *break* structure, without
    restricting the history the break is measured against. The lower highs that
    make a turn meaningful usually form on the approach into a zone, so that
    context must stay visible — but the break itself has to happen after price
    arrived, or the "confirmation" belongs to an earlier move entirely.
    """
    if df.empty:
        return None

    start = max(len(df) - window, 0)
    for position in range(len(df) - 1, start - 1, -1):
        if (
            not_before_ms is not None
            and int(df["timestamp"].iloc[position]) < not_before_ms
        ):
            continue
        prefix = df.iloc[: position + 1]
        if detect_choch(prefix, strength=strength, lookback=window * 2) is direction:
            return position
    return None


def find_fvg_after(
    df: pd.DataFrame,
    direction: Direction,
    *,
    start_position: int,
    min_fvg_pct: float = DEFAULT_LTF_MIN_FVG_PCT,
) -> tuple[int, float] | None:
    """First gap in ``direction`` at or after ``start_position``.

    Returns ``(position, gap_pct)``. Vectorised: the whole gap series is computed
    once and then filtered, rather than walking candles.
    """
    if len(df) < STRUCTURE_LENGTH:
        return None

    frame = fvg_frame(df)
    shift = STRUCTURE_LENGTH - 1
    if direction.is_long:
        gap = frame[BULL_GAP_COLUMN]
        reference = frame["high"].shift(shift)
    else:
        gap = frame[BEAR_GAP_COLUMN]
        reference = frame["low"].shift(shift)

    pct = gap / reference.where(reference > 0) * 100.0
    valid = (gap > 0.0) & (pct >= min_fvg_pct)

    positions = [p for p in range(max(start_position, 0), len(frame)) if bool(valid.iloc[p])]
    if not positions:
        return None
    position = positions[0]
    return position, float(pct.iloc[position])


def confirm_entry(
    df: pd.DataFrame,
    zone: WatchedZone,
    *,
    timeframe: str,
    strength: int = DEFAULT_SWING_STRENGTH,
    window: int = DEFAULT_CONFIRM_WINDOW,
    min_fvg_pct: float = DEFAULT_LTF_MIN_FVG_PCT,
    not_before_ms: int | None = None,
) -> tuple[LtfTrigger | None, ConfirmationRejection | None]:
    """Decide whether the lower timeframe confirms a tagged HTF zone.

    ``df`` must contain only CLOSED LTF candles. Returns ``(trigger, None)`` on
    confirmation, or ``(None, rejection)`` naming the stage that failed.

    ``not_before_ms`` defaults to the moment the zone was tagged, so a turn that
    predates price returning to the zone cannot be counted as confirmation of
    it.
    """
    if not_before_ms is None:
        not_before_ms = zone.tagged_ms
    if len(df) < STRUCTURE_LENGTH:
        return None, ConfirmationRejection(
            "in_zone", f"need {STRUCTURE_LENGTH} closed LTF candles, have {len(df)}"
        )

    recent = df.tail(window)

    # 1. Price must actually be working inside the zone on the LTF.
    in_zone = (recent["low"] <= zone.zone_high) & (recent["high"] >= zone.zone_low)
    if not bool(in_zone.any()):
        return None, ConfirmationRejection(
            "in_zone",
            f"no LTF candle in the last {len(recent)} traded inside "
            f"[{zone.zone_low:g}, {zone.zone_high:g}]",
        )

    # 2. A change of character in the direction of the HTF setup.
    choch_position = find_choch(
        recent,
        zone.direction,
        strength=strength,
        window=window,
        not_before_ms=not_before_ms,
    )
    if choch_position is None:
        return None, ConfirmationRejection(
            "choch",
            f"price is in the zone but the LTF has not turned — no "
            f"{zone.direction.value} change of character in the last {len(recent)} candles",
        )

    # 3. Displacement out of the turn, evidenced by an LTF gap.
    found = find_fvg_after(
        recent, zone.direction, start_position=choch_position, min_fvg_pct=min_fvg_pct
    )
    if found is None:
        return None, ConfirmationRejection(
            "ltf_fvg",
            f"{zone.direction.value} CHoCH formed but left no fair value gap — "
            "the turn had no displacement behind it",
        )

    fvg_position, fvg_pct = found
    return (
        LtfTrigger(
            direction=zone.direction,
            timeframe=timeframe,
            choch_timestamp=int(recent["timestamp"].iloc[choch_position]),
            fvg_timestamp=int(recent["timestamp"].iloc[fvg_position]),
            fvg_pct=fvg_pct,
            price=float(recent["close"].iloc[fvg_position]),
        ),
        None,
    )
