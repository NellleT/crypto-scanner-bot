"""Unit tests for lower-timeframe entry confirmation.

Each stage of the confirmation sequence is failed in isolation, so a failure
names which half of "price arrived and then turned" is missing.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from scanner.mtf import confirm_entry, find_choch, find_fvg_after
from scanner.smc import Direction
from scanner.watchlist import InvalidationReason, WatchState
from tests.test_smc import make_df
from tests.test_watchlist import make_zone

_MIN_MS = 60_000


def ltf(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build a 15m-style frame from (open, high, low, close) rows."""
    return pd.DataFrame(
        {
            "timestamp": [1_700_000_000_000 + i * 15 * _MIN_MS for i in range(len(bars))],
            "open": [b[0] for b in bars],
            "high": [b[1] for b in bars],
            "low": [b[2] for b in bars],
            "close": [b[3] for b in bars],
            "volume": [10.0] * len(bars),
        }
    )


# A long HTF zone spans 95-100. Price works inside it making LOWER swing highs
# (105 then 101), then reclaims the last one — a change of character — and gaps
# away from the turn. Two confirmed pivots are required, which is why the
# approach needs this many bars.
APPROACH = [
    (99.0, 100.0, 97.0, 98.0),
    (98.0, 105.0, 97.0, 99.0),    # swing high 105
    (98.0, 98.5, 95.5, 96.0),
    (96.0, 101.0, 95.2, 100.0),   # swing high 101 — lower, so a downtrend
    (98.0, 99.0, 95.0, 96.0),
]
CHOCH_BAR = (96.0, 103.0, 95.8, 102.5)   # closes above 101 -> bullish CHoCH

CONFIRMING_LONG = APPROACH + [
    CHOCH_BAR,
    (102.5, 104.0, 99.5, 103.5),   # low 99.5 > high[-3] 99  -> LTF FVG
    (103.5, 106.0, 104.2, 105.5),  # low 104.2 > high[-3] 103 -> LTF FVG
]

# Same approach into the zone, but price never reclaims the swing high.
NO_CHOCH_LONG = APPROACH + [
    (96.0, 99.0, 95.0, 97.0),
    (97.0, 98.5, 95.2, 96.0),
    (96.0, 98.0, 95.0, 97.0),
]

# The turn happens, but the bar after it overlaps, so no gap is left behind.
CHOCH_WITHOUT_FVG = APPROACH + [CHOCH_BAR, (102.5, 104.0, 98.0, 103.0)]

# Price never comes near the 95-100 zone at all.
AWAY_FROM_ZONE = [(120.0 + i, 121.0 + i, 119.0 + i, 120.5 + i) for i in range(10)]


# ---------------------------------------------------------------------------
# Full confirmation
# ---------------------------------------------------------------------------
def test_choch_followed_by_an_fvg_confirms_the_entry() -> None:
    trigger, rejection = confirm_entry(
        ltf(CONFIRMING_LONG), make_zone(Direction.LONG), timeframe="15m", strength=1
    )
    assert rejection is None, rejection
    assert trigger is not None
    assert trigger.direction is Direction.LONG
    assert trigger.timeframe == "15m"
    assert trigger.fvg_pct > 0
    # The gap must not predate the turn it is supposed to evidence.
    assert trigger.fvg_timestamp >= trigger.choch_timestamp


# ---------------------------------------------------------------------------
# Stage-by-stage rejection
# ---------------------------------------------------------------------------
def test_price_outside_the_zone_is_rejected_first() -> None:
    """A turn somewhere else is not an entry into this zone."""
    trigger, rejection = confirm_entry(
        ltf(AWAY_FROM_ZONE), make_zone(Direction.LONG), timeframe="15m", strength=1
    )
    assert trigger is None
    assert rejection is not None and rejection.stage == "in_zone"


def test_arriving_without_turning_is_rejected() -> None:
    """This is the trap v3.1 exists to avoid: price in the zone is not a signal."""
    trigger, rejection = confirm_entry(
        ltf(NO_CHOCH_LONG), make_zone(Direction.LONG), timeframe="15m", strength=1
    )
    assert trigger is None
    assert rejection is not None and rejection.stage == "choch"
    assert "has not turned" in rejection.reason


def test_a_turn_without_displacement_is_rejected() -> None:
    """A CHoCH that leaves no gap had no size behind it."""
    trigger, rejection = confirm_entry(
        ltf(CHOCH_WITHOUT_FVG), make_zone(Direction.LONG), timeframe="15m", strength=1
    )
    assert trigger is None
    assert rejection is not None and rejection.stage == "ltf_fvg"


def test_a_short_zone_is_not_confirmed_by_a_bullish_turn() -> None:
    """Direction must match: a rally is not confirmation for a short."""
    zone = make_zone(Direction.SHORT)   # zone spans 100-105
    bars = [(104.0, 104.5, 102.0, 102.5)] + CONFIRMING_LONG
    trigger, rejection = confirm_entry(
        ltf(bars), zone, timeframe="15m", strength=1
    )
    assert trigger is None
    assert rejection is not None and rejection.stage in {"choch", "ltf_fvg"}


def test_too_few_candles_is_reported() -> None:
    trigger, rejection = confirm_entry(
        ltf(CONFIRMING_LONG[:2]), make_zone(), timeframe="15m", strength=1
    )
    assert trigger is None
    assert rejection is not None and rejection.stage == "in_zone"


# ---------------------------------------------------------------------------
# Helper behaviour
# ---------------------------------------------------------------------------
def test_find_choch_locates_the_breaking_bar() -> None:
    frame = ltf(CONFIRMING_LONG)
    position = find_choch(frame, Direction.LONG, strength=1)
    assert position is not None
    # The break happens at bar 5 or later, never in the approach itself.
    assert position >= 5


def test_find_choch_returns_none_without_a_turn() -> None:
    assert find_choch(ltf(NO_CHOCH_LONG), Direction.LONG, strength=1) is None


def test_find_fvg_after_ignores_gaps_before_the_turn() -> None:
    """A gap that formed on the way down cannot evidence the turn upward."""
    frame = ltf(CONFIRMING_LONG)
    early = find_fvg_after(frame, Direction.LONG, start_position=0)
    late = find_fvg_after(frame, Direction.LONG, start_position=7)
    assert early is not None and late is not None
    assert early[0] == 6      # the first gap sits at bar 6
    assert late[0] == 7       # searching from 7 skips it


def test_find_fvg_after_honours_a_minimum_size() -> None:
    frame = ltf(CONFIRMING_LONG)
    assert find_fvg_after(frame, Direction.LONG, start_position=0) is not None
    assert (
        find_fvg_after(frame, Direction.LONG, start_position=0, min_fvg_pct=50.0)
        is None
    )


def test_find_fvg_after_needs_a_full_structure() -> None:
    assert find_fvg_after(ltf(CONFIRMING_LONG[:2]), Direction.LONG, start_position=0) is None


# ---------------------------------------------------------------------------
# The turn must belong to THIS visit to the zone
# ---------------------------------------------------------------------------
def test_a_turn_before_the_tag_does_not_confirm() -> None:
    """A CHoCH that predates price returning to the zone is a different move.

    Regression guard: `confirm_entry` takes `.tail(window)` internally, so
    without an explicit floor `find_choch` scans back past the tag and accepts
    an earlier turn — producing entries timestamped before the tag that caused
    them.
    """
    frame = ltf(CONFIRMING_LONG)
    zone = make_zone(Direction.LONG)

    # Price only returns to the zone AFTER every candle in the frame, so the
    # turn visible here belongs to an earlier move.
    zone.tagged_ms = int(frame["timestamp"].iloc[-1]) + 1
    trigger, rejection = confirm_entry(
        frame, zone, timeframe="15m", strength=1
    )
    assert trigger is None
    assert rejection is not None and rejection.stage == "choch"


def test_a_turn_at_or_after_the_tag_still_confirms() -> None:
    frame = ltf(CONFIRMING_LONG)
    zone = make_zone(Direction.LONG)
    zone.tagged_ms = int(frame["timestamp"].iloc[0])

    trigger, _ = confirm_entry(frame, zone, timeframe="15m", strength=1)
    assert trigger is not None
    assert trigger.choch_timestamp >= zone.tagged_ms
    assert trigger.fvg_timestamp >= trigger.choch_timestamp


def test_find_choch_honours_the_floor() -> None:
    frame = ltf(CONFIRMING_LONG)
    unrestricted = find_choch(frame, Direction.LONG, strength=1)
    assert unrestricted is not None

    floor = int(frame["timestamp"].iloc[unrestricted]) + 1
    assert find_choch(frame, Direction.LONG, strength=1, not_before_ms=floor) is None


# ---------------------------------------------------------------------------
# Price state at the trigger — "dead on arrival" (v3.1.1)
# ---------------------------------------------------------------------------
# On the full CONFIRMING_LONG sequence the most recent CHoCH is bar 7 (close
# 105.5 above the descending 105/101 highs) and its gap is bar 7 too.
THROUGH_THE_STOP = CONFIRMING_LONG + [(105.5, 105.6, 93.0, 93.5)]   # closes under 94
PAST_THE_TARGET = CONFIRMING_LONG + [(105.5, 126.0, 105.0, 125.0)]  # closes over 124


def mirror(bars: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """Reflect bars through 100, turning a bullish sequence into a bearish one."""
    return [(200.0 - o, 200.0 - l, 200.0 - h, 200.0 - c) for o, h, l, c in bars]


def test_a_trigger_that_closes_through_the_stop_is_dead_on_arrival() -> None:
    """Price swept under the zone and turned while still beyond the stop.

    The zone sits above the turn; its stop (105.6) is above the trigger close
    (105.5). Alerting this sends an order whose stop has already traded.
    """
    zone = dataclasses.replace(
        make_zone(Direction.LONG), zone_low=105.8, zone_high=110.0, proximal=110.0,
        distal=105.8, entry=110.0, stop_loss=105.6, take_profit=123.2,
    )
    trigger, rejection = confirm_entry(ltf(CONFIRMING_LONG), zone, timeframe="15m", strength=1)
    assert trigger is None
    assert rejection is not None and rejection.stage == "stop_breached"
    assert rejection.invalidates is InvalidationReason.STOP_BREACHED


def test_price_closing_through_the_stop_after_the_trigger_is_dead_on_arrival() -> None:
    """The trigger itself was fine, but the newest close is already under the
    stop — a scan that runs late must not alert a trade that has failed."""
    trigger, rejection = confirm_entry(
        ltf(THROUGH_THE_STOP), make_zone(Direction.LONG), timeframe="15m", strength=1
    )
    assert trigger is None
    assert rejection is not None and rejection.stage == "stop_breached"
    assert rejection.invalidates is InvalidationReason.STOP_BREACHED


def test_a_wick_through_the_stop_that_closes_back_inside_still_confirms() -> None:
    """Closes decide, as everywhere else: a wick is a sweep, not a failure."""
    swept = CONFIRMING_LONG + [(105.5, 105.8, 93.0, 104.8)]   # low 93 < stop 94
    trigger, rejection = confirm_entry(
        ltf(swept), make_zone(Direction.LONG), timeframe="15m", strength=1
    )
    assert rejection is None and trigger is not None


def test_a_trigger_past_the_target_is_retired() -> None:
    trigger, rejection = confirm_entry(
        ltf(PAST_THE_TARGET), make_zone(Direction.LONG), timeframe="15m", strength=1
    )
    assert trigger is None
    assert rejection is not None and rejection.stage == "target_reached"
    assert rejection.invalidates is InvalidationReason.TARGET_REACHED


def test_the_price_check_mirrors_for_shorts() -> None:
    zone = make_zone(Direction.SHORT)   # 100-105, stop 106, target 76
    trigger, rejection = confirm_entry(ltf(mirror(CONFIRMING_LONG)), zone,
                                       timeframe="15m", strength=1)
    assert rejection is None and trigger is not None
    assert trigger.direction is Direction.SHORT

    trigger, rejection = confirm_entry(ltf(mirror(THROUGH_THE_STOP)), zone,
                                       timeframe="15m", strength=1)
    assert trigger is None
    assert rejection is not None and rejection.invalidates is InvalidationReason.STOP_BREACHED


def test_ordinary_rejections_leave_the_zone_waiting() -> None:
    """Only a dead setup invalidates; "not yet" must not."""
    _, rejection = confirm_entry(
        ltf(NO_CHOCH_LONG), make_zone(Direction.LONG), timeframe="15m", strength=1
    )
    assert rejection is not None and rejection.invalidates is None


def test_the_live_scanner_retires_a_dead_on_arrival_zone(tmp_path, monkeypatch) -> None:
    """The live path must invalidate, not merely skip — otherwise the stale
    trigger is re-found every pass and fires if price drifts back."""
    from scanner.bot import ScannerBot
    from scanner.config import Settings

    monkeypatch.setenv("DRY_RUN", "true")
    settings = dataclasses.replace(
        Settings.from_env(), watchlist_file=tmp_path / "watchlist.json", swing_strength=1
    )
    bot = ScannerBot(settings)
    zone = make_zone(Direction.LONG)
    zone.state = WatchState.TAGGED
    zone.tagged_ms = int(ltf(THROUGH_THE_STOP)["timestamp"].iloc[0])

    trigger, suppressed = bot._try_confirm(zone, ltf(THROUGH_THE_STOP))
    assert trigger is None and not suppressed
    assert zone.state is WatchState.INVALIDATED
    assert zone.invalidation is InvalidationReason.STOP_BREACHED
