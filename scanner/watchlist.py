"""Active zone tracking: the state machine between an HTF setup and an entry.

v3.0 placed a resting limit order the moment a block was validated. v3.1 does
not: a validated HTF block becomes a *watched zone*, and an order is only built
once price returns to that zone **and** the lower timeframe confirms with a
change of character. The zone can also die before that ever happens.

::

    PENDING ──price enters zone──▶ TAGGED ──LTF CHoCH + FVG──▶ TRIGGERED
       │                             │
       ├── take-profit reached first ┤
       ├── HTF close beyond distal ──┤
       └── max age exceeded ─────────┴──▶ INVALIDATED

**Why persistence matters.** The lifecycle spans many candles, but a scheduled
run is a fresh process. Held only in memory, every zone would be re-created from
scratch each run and no zone could ever reach TRIGGERED. The watchlist is
therefore written to disk after each pass and reloaded on start.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterable, Iterator

import pandas as pd

from scanner.smc import Direction, OrderBlock

logger: Final[logging.Logger] = logging.getLogger(__name__)

#: Schema version, so an older file is discarded rather than misread.
SCHEMA_VERSION: Final[int] = 1


class WatchState(str, Enum):
    """Where a watched zone is in its lifecycle."""

    PENDING = "pending"          # waiting for price to return to the zone
    TAGGED = "tagged"            # price is in the zone, awaiting LTF confirmation
    TRIGGERED = "triggered"      # LTF confirmed; an order was built
    INVALIDATED = "invalidated"  # died before it could trigger

    @property
    def is_active(self) -> bool:
        """True while the zone is still waiting for something to happen."""
        return self in (WatchState.PENDING, WatchState.TAGGED)


class InvalidationReason(str, Enum):
    """Why a zone stopped being tradable."""

    TP_BEFORE_TAG = "tp_before_tag"        # liquidity already swept
    STRUCTURE_BREAK = "structure_break"    # HTF closed beyond the distal edge
    EXPIRED = "expired"                    # too old to be relevant

    @property
    def detail(self) -> str:
        return {
            InvalidationReason.TP_BEFORE_TAG: (
                "price reached the 1:4 target before returning to the zone — the "
                "move happened without us and the liquidity is gone"
            ),
            InvalidationReason.STRUCTURE_BREAK: (
                "an HTF candle closed beyond the distal edge, so the structural "
                "extreme that defined the zone no longer holds"
            ),
            InvalidationReason.EXPIRED: "zone exceeded its maximum age",
        }[self]


@dataclass(slots=True)
class WatchedZone:
    """One HTF order block being tracked toward a possible entry."""

    symbol: str
    timeframe: str
    direction: Direction
    block_timestamp: int
    zone_low: float
    zone_high: float
    proximal: float
    distal: float
    entry: float
    stop_loss: float
    take_profit: float
    quantity: float
    fvg_pct: float
    fib_level: float
    created_ms: int
    state: WatchState = WatchState.PENDING
    tagged_ms: int | None = None
    triggered_ms: int | None = None
    invalidation: InvalidationReason | None = None

    @property
    def key(self) -> tuple[str, str, int, str]:
        """Stable identity — one zone per block per symbol per timeframe."""
        return (self.symbol, self.timeframe, self.block_timestamp, self.direction.value)

    @property
    def is_active(self) -> bool:
        return self.state.is_active

    @property
    def created_at(self) -> datetime:
        return datetime.fromtimestamp(self.created_ms / 1000, tz=timezone.utc)

    def contains(self, price: float) -> bool:
        return self.zone_low <= price <= self.zone_high

    def age_ms(self, now_ms: int) -> int:
        return max(now_ms - self.created_ms, 0)

    def reached_take_profit(self, high: float, low: float) -> bool:
        """True when a candle traded through the target."""
        return high >= self.take_profit if self.direction.is_long else low <= self.take_profit

    def broke_structure(self, close: float) -> bool:
        """True when a close invalidates the structural extreme."""
        return close < self.distal if self.direction.is_long else close > self.distal

    def touched(self, high: float, low: float) -> bool:
        """True when a candle traded into the zone."""
        return low <= self.zone_high and high >= self.zone_low

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "direction": self.direction.value,
            "block_timestamp": self.block_timestamp,
            "zone_low": self.zone_low,
            "zone_high": self.zone_high,
            "proximal": self.proximal,
            "distal": self.distal,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "quantity": self.quantity,
            "fvg_pct": self.fvg_pct,
            "fib_level": self.fib_level,
            "created_ms": self.created_ms,
            "state": self.state.value,
            "tagged_ms": self.tagged_ms,
            "triggered_ms": self.triggered_ms,
            "invalidation": self.invalidation.value if self.invalidation else None,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WatchedZone":
        invalidation = raw.get("invalidation")
        return cls(
            symbol=str(raw["symbol"]),
            timeframe=str(raw["timeframe"]),
            direction=Direction(raw["direction"]),
            block_timestamp=int(raw["block_timestamp"]),
            zone_low=float(raw["zone_low"]),
            zone_high=float(raw["zone_high"]),
            proximal=float(raw["proximal"]),
            distal=float(raw["distal"]),
            entry=float(raw["entry"]),
            stop_loss=float(raw["stop_loss"]),
            take_profit=float(raw["take_profit"]),
            quantity=float(raw["quantity"]),
            fvg_pct=float(raw["fvg_pct"]),
            fib_level=float(raw["fib_level"]),
            created_ms=int(raw["created_ms"]),
            state=WatchState(raw.get("state", WatchState.PENDING.value)),
            tagged_ms=raw.get("tagged_ms"),
            triggered_ms=raw.get("triggered_ms"),
            invalidation=InvalidationReason(invalidation) if invalidation else None,
        )

    @classmethod
    def from_order_block(
        cls,
        symbol: str,
        timeframe: str,
        block: OrderBlock,
        *,
        entry: float,
        stop_loss: float,
        take_profit: float,
        quantity: float,
        created_ms: int,
    ) -> "WatchedZone":
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            direction=block.direction,
            block_timestamp=block.candle.timestamp,
            zone_low=block.candle.low,
            zone_high=block.candle.high,
            proximal=block.proximal,
            distal=block.distal,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            fvg_pct=block.fvg.pct,
            fib_level=(
                block.swing_range.fib_level(block.proximal)
                if block.swing_range is not None
                else 0.5
            ),
            created_ms=created_ms,
        )


@dataclass(frozen=True, slots=True)
class ZoneEvent:
    """Something that happened to a zone during an update."""

    zone: WatchedZone
    kind: str          # tagged | invalidated
    detail: str


@dataclass(slots=True)
class Watchlist:
    """All zones being tracked, keyed by identity."""

    zones: dict[tuple[str, str, int, str], WatchedZone] = field(default_factory=dict)
    max_age_ms: int | None = None

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.zones)

    def __iter__(self) -> Iterator[WatchedZone]:
        return iter(self.zones.values())

    def add(self, zone: WatchedZone) -> bool:
        """Register a zone. Returns ``False`` if it is already known."""
        if zone.key in self.zones:
            return False
        self.zones[zone.key] = zone
        return True

    def active(self, symbol: str | None = None) -> list[WatchedZone]:
        return [
            z
            for z in self.zones.values()
            if z.is_active and (symbol is None or z.symbol == symbol)
        ]

    def prune(self, *, keep_terminal: int = 200) -> int:
        """Drop the oldest finished zones so the file cannot grow forever."""
        terminal = sorted(
            (z for z in self.zones.values() if not z.is_active),
            key=lambda z: z.created_ms,
        )
        removed = 0
        for zone in terminal[: max(len(terminal) - keep_terminal, 0)]:
            del self.zones[zone.key]
            removed += 1
        return removed

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def update_from_htf(
        self, symbol: str, df: pd.DataFrame, *, now_ms: int | None = None
    ) -> list[ZoneEvent]:
        """Advance every active zone for ``symbol`` against fresh HTF candles.

        Candles are replayed in time order and the first terminal event wins,
        because the order genuinely matters: a zone whose target was reached
        before price ever came back is dead, and must not later be marked as
        tagged by a candle that only arrived afterwards.
        """
        events: list[ZoneEvent] = []
        zones = self.active(symbol)
        if not zones or df.empty:
            return events

        now = int(df["timestamp"].iloc[-1]) if now_ms is None else now_ms
        columns = df[["timestamp", "high", "low", "close"]].to_numpy()

        for zone in zones:
            for timestamp, high, low, close in columns:
                ts = int(timestamp)
                if ts <= zone.block_timestamp or ts < zone.created_ms:
                    continue  # the structure's own candles cannot invalidate it

                if zone.state is WatchState.PENDING and zone.reached_take_profit(
                    float(high), float(low)
                ):
                    events.append(
                        self._invalidate(zone, InvalidationReason.TP_BEFORE_TAG)
                    )
                    break

                if zone.broke_structure(float(close)):
                    events.append(
                        self._invalidate(zone, InvalidationReason.STRUCTURE_BREAK)
                    )
                    break

                if zone.state is WatchState.PENDING and zone.touched(
                    float(high), float(low)
                ):
                    zone.state = WatchState.TAGGED
                    zone.tagged_ms = ts
                    events.append(
                        ZoneEvent(
                            zone,
                            "tagged",
                            f"price returned to the {zone.direction.value} zone "
                            f"[{zone.zone_low:g}, {zone.zone_high:g}] — awaiting LTF "
                            "confirmation",
                        )
                    )

            if (
                zone.is_active
                and self.max_age_ms is not None
                and zone.age_ms(now) > self.max_age_ms
            ):
                events.append(self._invalidate(zone, InvalidationReason.EXPIRED))

        return events

    def _invalidate(self, zone: WatchedZone, reason: InvalidationReason) -> ZoneEvent:
        zone.state = WatchState.INVALIDATED
        zone.invalidation = reason
        return ZoneEvent(zone, "invalidated", reason.detail)

    def mark_triggered(self, zone: WatchedZone, *, when_ms: int) -> None:
        zone.state = WatchState.TRIGGERED
        zone.triggered_ms = when_ms

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SCHEMA_VERSION,
            "zones": [z.to_dict() for z in self.zones.values()],
        }

    def save(self, path: Path) -> None:
        """Write atomically, so a crash mid-write cannot corrupt the file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(self.to_dict(), stream, indent=2)
            os.replace(tmp_name, path)
        except Exception:
            with suppress_errors():
                os.unlink(tmp_name)
            raise

    @classmethod
    def load(cls, path: Path, *, max_age_ms: int | None = None) -> "Watchlist":
        """Read a saved watchlist. A missing or unreadable file starts empty."""
        watchlist = cls(max_age_ms=max_age_ms)
        if not path.is_file():
            return watchlist

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Could not read the watchlist at %s (%s); starting empty.", path, exc)
            return watchlist

        if int(raw.get("version", 0)) != SCHEMA_VERSION:
            logger.warning(
                "Watchlist at %s is schema v%s, expected v%d; starting empty.",
                path,
                raw.get("version"),
                SCHEMA_VERSION,
            )
            return watchlist

        for entry in raw.get("zones", []):
            try:
                zone = WatchedZone.from_dict(entry)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed watchlist entry (%s).", exc)
                continue
            watchlist.zones[zone.key] = zone

        logger.info(
            "Loaded %d watched zone(s) from %s (%d active).",
            len(watchlist.zones),
            path,
            len(watchlist.active()),
        )
        return watchlist

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        """Zones per state, plus a breakdown of invalidation reasons."""
        tally: dict[str, int] = {state.value: 0 for state in WatchState}
        for zone in self.zones.values():
            tally[zone.state.value] += 1
            if zone.invalidation is not None:
                key = f"invalidated_{zone.invalidation.value}"
                tally[key] = tally.get(key, 0) + 1
        return tally


class suppress_errors:
    """Tiny context manager: ignore cleanup failures during error handling."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> bool:
        return True


def zones_from(records: Iterable[dict[str, Any]]) -> list[WatchedZone]:
    """Rebuild zones from raw dicts, skipping anything malformed."""
    zones: list[WatchedZone] = []
    for record in records:
        try:
            zones.append(WatchedZone.from_dict(record))
        except (KeyError, TypeError, ValueError):
            continue
    return zones
