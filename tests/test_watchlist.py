"""Unit tests for the zone lifecycle and its persistence.

The lifecycle spans many candles while a scheduled run is a fresh process, so
both the state transitions and the round-trip through disk matter.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scanner.smc import Direction
from scanner.watchlist import (
    InvalidationReason,
    WatchedZone,
    Watchlist,
    WatchState,
)

_HOUR_MS = 3_600_000
_BASE_MS = 1_700_000_000_000


def make_zone(
    direction: Direction = Direction.LONG,
    *,
    symbol: str = "BTC/USDT",
    created_ms: int = _BASE_MS,
) -> WatchedZone:
    """A long zone at 95-100, stop 94, target 124 (1:4 on a 6-wide risk)."""
    if direction.is_long:
        return WatchedZone(
            symbol=symbol,
            timeframe="1h",
            direction=direction,
            block_timestamp=created_ms - _HOUR_MS,
            zone_low=95.0,
            zone_high=100.0,
            proximal=100.0,
            distal=95.0,
            entry=100.0,
            stop_loss=94.0,
            take_profit=124.0,
            quantity=10.0,
            fvg_pct=1.5,
            fib_level=0.25,
            created_ms=created_ms,
        )
    return WatchedZone(
        symbol=symbol,
        timeframe="1h",
        direction=direction,
        block_timestamp=created_ms - _HOUR_MS,
        zone_low=100.0,
        zone_high=105.0,
        proximal=100.0,
        distal=105.0,
        entry=100.0,
        stop_loss=106.0,
        take_profit=76.0,
        quantity=10.0,
        fvg_pct=1.5,
        fib_level=0.75,
        created_ms=created_ms,
    )


def candles(rows: list[tuple[int, float, float, float]]) -> pd.DataFrame:
    """rows are (offset_hours, high, low, close)."""
    return pd.DataFrame(
        {
            "timestamp": [_BASE_MS + h * _HOUR_MS for h, _, _, _ in rows],
            "open": [c for _, _, _, c in rows],
            "high": [h for _, h, _, _ in rows],
            "low": [l for _, _, l, _ in rows],
            "close": [c for _, _, _, c in rows],
            "volume": [1.0] * len(rows),
        }
    )


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------
def test_price_returning_to_the_zone_tags_it() -> None:
    watchlist = Watchlist()
    zone = make_zone()
    watchlist.add(zone)

    events = watchlist.update_from_htf("BTC/USDT", candles([(1, 112, 98, 110)]))

    assert zone.state is WatchState.TAGGED
    assert zone.tagged_ms == _BASE_MS + _HOUR_MS
    assert [e.kind for e in events] == ["tagged"]


def test_price_staying_away_leaves_the_zone_pending() -> None:
    watchlist = Watchlist()
    zone = make_zone()
    watchlist.add(zone)

    watchlist.update_from_htf("BTC/USDT", candles([(1, 115, 108, 112)]))
    assert zone.state is WatchState.PENDING


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------
def test_target_reached_before_the_tag_kills_the_zone() -> None:
    """The move happened without us; the liquidity is already gone."""
    watchlist = Watchlist()
    zone = make_zone()
    watchlist.add(zone)

    events = watchlist.update_from_htf("BTC/USDT", candles([(1, 125, 118, 124)]))

    assert zone.state is WatchState.INVALIDATED
    assert zone.invalidation is InvalidationReason.TP_BEFORE_TAG
    assert [e.kind for e in events] == ["invalidated"]


def test_close_beyond_the_distal_edge_kills_the_zone() -> None:
    watchlist = Watchlist()
    zone = make_zone()
    watchlist.add(zone)

    watchlist.update_from_htf("BTC/USDT", candles([(1, 99, 90, 92)]))

    assert zone.state is WatchState.INVALIDATED
    assert zone.invalidation is InvalidationReason.STRUCTURE_BREAK


def test_a_wick_below_the_distal_edge_does_not_break_structure() -> None:
    """Only a CLOSE invalidates; wicks through a zone are the normal case."""
    watchlist = Watchlist()
    zone = make_zone()
    watchlist.add(zone)

    watchlist.update_from_htf("BTC/USDT", candles([(1, 101, 90, 97)]))

    assert zone.state is WatchState.TAGGED
    assert zone.invalidation is None


def test_short_zone_invalidations_are_mirrored() -> None:
    watchlist = Watchlist()
    zone = make_zone(Direction.SHORT)
    watchlist.add(zone)

    watchlist.update_from_htf("BTC/USDT", candles([(1, 112, 108, 110)]))

    assert zone.state is WatchState.INVALIDATED
    assert zone.invalidation is InvalidationReason.STRUCTURE_BREAK


def test_short_target_before_tag_kills_the_zone() -> None:
    watchlist = Watchlist()
    zone = make_zone(Direction.SHORT)
    watchlist.add(zone)

    watchlist.update_from_htf("BTC/USDT", candles([(1, 90, 75, 80)]))

    assert zone.invalidation is InvalidationReason.TP_BEFORE_TAG


def test_age_expires_a_zone_that_never_traded() -> None:
    watchlist = Watchlist(max_age_ms=2 * _HOUR_MS)
    zone = make_zone()
    watchlist.add(zone)

    watchlist.update_from_htf("BTC/USDT", candles([(5, 115, 108, 112)]))

    assert zone.state is WatchState.INVALIDATED
    assert zone.invalidation is InvalidationReason.EXPIRED


def test_the_first_terminal_event_wins() -> None:
    """Order matters: a zone killed on Monday cannot be tagged on Tuesday."""
    watchlist = Watchlist()
    zone = make_zone()
    watchlist.add(zone)

    watchlist.update_from_htf(
        "BTC/USDT",
        candles([(1, 125, 120, 124), (2, 101, 96, 99)]),  # target, then a tag
    )

    assert zone.state is WatchState.INVALIDATED
    assert zone.invalidation is InvalidationReason.TP_BEFORE_TAG
    assert zone.tagged_ms is None


def test_the_structures_own_candles_cannot_invalidate_it() -> None:
    """Candles at or before the block formed are history, not new information."""
    watchlist = Watchlist()
    zone = make_zone(created_ms=_BASE_MS + 5 * _HOUR_MS)
    watchlist.add(zone)

    watchlist.update_from_htf("BTC/USDT", candles([(0, 125, 90, 124)]))

    assert zone.state is WatchState.PENDING


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------
def test_adding_the_same_zone_twice_is_idempotent() -> None:
    watchlist = Watchlist()
    assert watchlist.add(make_zone()) is True
    assert watchlist.add(make_zone()) is False
    assert len(watchlist) == 1


def test_active_filters_by_symbol_and_state() -> None:
    watchlist = Watchlist()
    watchlist.add(make_zone(symbol="BTC/USDT"))
    watchlist.add(make_zone(symbol="ETH/USDT"))
    assert len(watchlist.active()) == 2
    assert len(watchlist.active("BTC/USDT")) == 1

    for zone in watchlist:
        zone.state = WatchState.TRIGGERED
    assert watchlist.active() == []


def test_prune_keeps_active_zones_and_trims_finished_ones() -> None:
    watchlist = Watchlist()
    for index in range(10):
        zone = make_zone(created_ms=_BASE_MS + index * _HOUR_MS)
        zone.state = WatchState.INVALIDATED
        watchlist.add(zone)
    live = make_zone(created_ms=_BASE_MS + 99 * _HOUR_MS)
    watchlist.add(live)

    watchlist.prune(keep_terminal=3)

    assert len(watchlist.active()) == 1
    assert len(watchlist) == 4  # 3 terminal + 1 active


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_round_trip_through_disk_preserves_state(tmp_path) -> None:
    """Without this the state machine cannot survive a scheduled run."""
    path = tmp_path / "watchlist.json"
    watchlist = Watchlist()
    zone = make_zone()
    zone.state = WatchState.TAGGED
    zone.tagged_ms = _BASE_MS + _HOUR_MS
    watchlist.add(zone)
    watchlist.save(path)

    restored = Watchlist.load(path)
    assert len(restored) == 1
    loaded = next(iter(restored))
    assert loaded.state is WatchState.TAGGED
    assert loaded.tagged_ms == _BASE_MS + _HOUR_MS
    assert loaded.key == zone.key
    assert loaded.entry == pytest.approx(zone.entry)
    assert loaded.direction is Direction.LONG


def test_missing_file_starts_empty(tmp_path) -> None:
    assert len(Watchlist.load(tmp_path / "absent.json")) == 0


def test_corrupt_file_starts_empty_rather_than_crashing(tmp_path) -> None:
    path = tmp_path / "watchlist.json"
    path.write_text("{ not json", encoding="utf-8")
    assert len(Watchlist.load(path)) == 0


def test_future_schema_is_discarded(tmp_path) -> None:
    path = tmp_path / "watchlist.json"
    path.write_text(json.dumps({"version": 999, "zones": []}), encoding="utf-8")
    assert len(Watchlist.load(path)) == 0


def test_malformed_entries_are_skipped_individually(tmp_path) -> None:
    path = tmp_path / "watchlist.json"
    good = make_zone().to_dict()
    path.write_text(
        json.dumps({"version": 1, "zones": [good, {"symbol": "BROKEN"}]}),
        encoding="utf-8",
    )
    assert len(Watchlist.load(path)) == 1


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path) -> None:
    path = tmp_path / "watchlist.json"
    watchlist = Watchlist()
    watchlist.add(make_zone())
    watchlist.save(path)
    watchlist.save(path)

    assert path.is_file()
    assert [p.name for p in tmp_path.iterdir()] == ["watchlist.json"]


def test_counts_break_down_by_state_and_reason() -> None:
    watchlist = Watchlist()
    live = make_zone(symbol="A/USDT")
    dead = make_zone(symbol="B/USDT")
    dead.state = WatchState.INVALIDATED
    dead.invalidation = InvalidationReason.TP_BEFORE_TAG
    watchlist.add(live)
    watchlist.add(dead)

    counts = watchlist.counts()
    assert counts["pending"] == 1
    assert counts["invalidated"] == 1
    assert counts["invalidated_tp_before_tag"] == 1
