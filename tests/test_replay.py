"""Unit tests for the replay's fidelity to the live scanner.

The live scanner wakes once per HTF close, retires dead zones *before* it looks
for a trigger, and keeps a tagged zone alive until it breaks or ages out. A
replay that searched every LTF bar for a fixed horizon instead confirmed 135 of
481 three-year entries after the live scanner would already have killed the
zone. These tests pin each of those rules.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from scanner.analytics import _confirm_from_tag, _tagged_deadline
from scanner.smc import Direction
from scanner.watchlist import InvalidationReason
from tests.test_mtf import CONFIRMING_LONG, NO_CHOCH_LONG, ltf
from tests.test_watchlist import make_zone

_HOUR_MS = 3_600_000
_M15_MS = 15 * 60_000


def hourly(closes: list[float], start_ms: int) -> pd.DataFrame:
    """1h candles with the given closes; ranges kept inside 99-101 unless the
    close itself sits outside."""
    return pd.DataFrame(
        {
            "timestamp": [start_ms + i * _HOUR_MS for i in range(len(closes))],
            "open": closes,
            "high": [max(c, 101.0) for c in closes],
            "low": [min(c, 99.0) for c in closes],
            "close": closes,
            "volume": [1.0] * len(closes),
        }
    )


def tagged_long(tagged_ms: int):
    zone = make_zone(Direction.LONG, created_ms=tagged_ms - 2 * _HOUR_MS)   # 95-100, stop 94
    zone.tagged_ms = tagged_ms
    return zone


def confirm(frame: pd.DataFrame, zone, **kwargs):
    return _confirm_from_tag(
        frame, zone, tagged_ms=zone.tagged_ms, strategy_timeframe="15m",
        confirm_window=30, min_fvg_pct=0.0, swing_strength=1, ltf_bar_ms=_M15_MS,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# When a tagged zone dies
# ---------------------------------------------------------------------------
def test_a_close_beyond_the_distal_after_the_tag_kills_the_zone_at_that_close() -> None:
    tag = 1_700_000_000_000
    zone = tagged_long(tag)
    htf = hourly([98.0, 97.0, 94.5, 98.0], tag)       # the 3rd candle closes under 95
    deadline, reason = _tagged_deadline(zone, htf, max_zone_age_ms=None, htf_bar_ms=_HOUR_MS)
    assert reason is InvalidationReason.STRUCTURE_BREAK
    assert deadline == tag + 3 * _HOUR_MS              # that candle's close


def test_a_wick_beyond_the_distal_does_not_kill_the_zone() -> None:
    tag = 1_700_000_000_000
    zone = tagged_long(tag)
    htf = hourly([98.0, 98.0, 98.0], tag)
    htf.loc[1, "low"] = 90.0                           # deep wick, close back inside
    assert _tagged_deadline(zone, htf, max_zone_age_ms=None, htf_bar_ms=_HOUR_MS) == (None, None)


def test_a_tagged_zone_still_ages_out() -> None:
    tag = 1_700_000_000_000
    zone = tagged_long(tag)                            # created two hours before the tag
    htf = hourly([98.0] * 6, tag)
    deadline, reason = _tagged_deadline(
        zone, htf, max_zone_age_ms=4 * _HOUR_MS, htf_bar_ms=_HOUR_MS
    )
    assert reason is InvalidationReason.EXPIRED
    # First candle older than 4h opens at tag+3h (age 5h); it is known at its close.
    assert deadline == tag + 4 * _HOUR_MS


def test_the_tagging_candle_itself_is_not_re_judged() -> None:
    tag = 1_700_000_000_000
    zone = tagged_long(tag)
    htf = hourly([94.0, 98.0], tag)                    # only the tag candle is outside
    assert _tagged_deadline(zone, htf, max_zone_age_ms=None, htf_bar_ms=_HOUR_MS) == (None, None)


# ---------------------------------------------------------------------------
# When the live scanner would look, and what it would see
# ---------------------------------------------------------------------------
def test_confirmation_is_dispatched_at_the_next_scanner_pass() -> None:
    """The trigger completes mid-hour; the alert goes out at the hourly pass."""
    frame = ltf(CONFIRMING_LONG)
    zone = tagged_long(int(frame["timestamp"].iloc[0]))
    result = confirm(frame, zone, pass_ms=_HOUR_MS)
    assert result.trigger is not None
    assert result.alert_ms == zone.tagged_ms + 2 * _HOUR_MS
    assert result.trigger.fvg_timestamp + _M15_MS <= result.alert_ms


def test_passes_at_every_ltf_close_find_the_earliest_trigger() -> None:
    frame = ltf(CONFIRMING_LONG)
    zone = tagged_long(int(frame["timestamp"].iloc[0]))
    result = confirm(frame, zone)                      # pass_ms=0: every LTF close
    assert result.trigger is not None
    assert result.alert_ms == zone.tagged_ms + 7 * _M15_MS   # bar 6 closes


def test_a_zone_that_dies_before_the_pass_cannot_confirm() -> None:
    frame = ltf(CONFIRMING_LONG)
    zone = tagged_long(int(frame["timestamp"].iloc[0]))
    dead = confirm(frame, zone, pass_ms=_HOUR_MS,
                   deadline_ms=zone.tagged_ms + 2 * _HOUR_MS,
                   death=InvalidationReason.STRUCTURE_BREAK)
    assert dead.trigger is None
    assert dead.invalidation is InvalidationReason.STRUCTURE_BREAK

    alive = confirm(frame, zone, pass_ms=_HOUR_MS,
                    deadline_ms=zone.tagged_ms + 3 * _HOUR_MS,
                    death=InvalidationReason.STRUCTURE_BREAK)
    assert alive.trigger is not None


def test_a_dead_on_arrival_trigger_retires_the_zone_in_the_replay() -> None:
    frame = ltf(CONFIRMING_LONG)
    zone = dataclasses.replace(
        tagged_long(int(frame["timestamp"].iloc[0])), zone_low=105.8, zone_high=110.0,
        proximal=110.0, distal=105.8, entry=110.0, stop_loss=105.6, take_profit=123.2,
    )
    result = confirm(frame, zone)
    assert result.trigger is None
    assert result.invalidation is InvalidationReason.STOP_BREACHED
    assert result.alert_ms == zone.tagged_ms + 8 * _M15_MS   # first pass that sees bar 7


def test_running_out_of_data_leaves_the_zone_open() -> None:
    frame = ltf(CONFIRMING_LONG[:5])                  # the turn never arrives
    zone = tagged_long(int(frame["timestamp"].iloc[0]))
    result = confirm(frame, zone)
    assert result.trigger is None and result.invalidation is None
    assert result.reason == "open"


def test_a_horizon_cap_reports_the_last_stage_reached() -> None:
    frame = ltf(NO_CHOCH_LONG)                        # in the zone, never turns
    zone = tagged_long(int(frame["timestamp"].iloc[0]))
    # Pass 7 is the first with enough candles to evaluate (confirm_window // 4).
    result = confirm(frame, zone, horizon=7)
    assert result.trigger is None and result.invalidation is None
    assert result.reason == "choch"
