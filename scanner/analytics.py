"""Historical simulation and signal-frequency analytics.

Replays the full v3.1 pipeline over a stored dataset so the funnel can be
measured rather than guessed: how many order blocks form, how many die at each
filter, how many reach the watchlist, and how many of those the lower timeframe
actually confirms.

The replay is deliberately **event-driven and causal**. Each HTF bar is
evaluated against only the candles up to and including itself, a zone is then
advanced bar by bar through its lifecycle, and lower-timeframe confirmation is
searched only in the candles that closed *after* the zone was tagged. Anything
cheaper — scanning the whole frame at once and matching zones to outcomes
afterwards — would quietly use information the live scanner could not have had.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final, Mapping, Sequence

import pandas as pd

from scanner.mtf import LtfTrigger, confirm_entry
from scanner.smc import Direction
from scanner.strategy import FilterStage, OrderBlockStrategy
from scanner.watchlist import InvalidationReason, WatchedZone, WatchState

logger: Final[logging.Logger] = logging.getLogger(__name__)


def _money(value: float) -> str:
    """Render a price readably across BTC-scale and sub-dollar instruments."""
    if value >= 1_000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:,.4f}"
    return f"{value:.6f}"

#: LTF bars searched forward from a tag before giving up on confirmation.
DEFAULT_CONFIRM_HORIZON: Final[int] = 40


@dataclass(frozen=True, slots=True)
class ConfirmedEntry:
    """A single confirmed entry, with everything needed to audit it by hand.

    Every timestamp is the *open* of the candle concerned, so each one can be
    located directly on a chart. The four together tell the whole story: when
    the structure formed, when price came back to it, when the lower timeframe
    turned, and when that turn produced displacement.
    """

    symbol: str
    direction: Direction
    timeframe: str
    ltf_timeframe: str
    zone_low: float
    zone_high: float
    fvg_pct: float
    fib_level: float
    zone_half: str
    block_ms: int        # HTF order-block candle
    created_ms: int      # HTF candle that completed the structure
    tagged_ms: int       # HTF candle where price re-entered the zone
    choch_ms: int        # LTF candle whose close broke structure
    ltf_fvg_ms: int      # LTF candle that left the confirming gap
    ltf_fvg_pct: float
    entry: float
    stop_loss: float
    take_profit: float
    quantity: float

    @staticmethod
    def _at(ms: int) -> datetime:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

    @property
    def block_at(self) -> datetime:
        return self._at(self.block_ms)

    @property
    def tagged_at(self) -> datetime:
        return self._at(self.tagged_ms)

    @property
    def choch_at(self) -> datetime:
        return self._at(self.choch_ms)

    @property
    def confirmed_at(self) -> datetime:
        return self._at(self.ltf_fvg_ms)

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop_loss)

    @property
    def risk_pct(self) -> float:
        return self.risk_per_unit / self.entry * 100.0 if self.entry > 0 else 0.0

    @property
    def reward_ratio(self) -> float:
        risk = self.risk_per_unit
        return abs(self.take_profit - self.entry) / risk if risk > 0 else 0.0

    @property
    def hours_waiting(self) -> float:
        """Hours between the structure forming and price returning to it."""
        return max(self.tagged_ms - self.created_ms, 0) / 3_600_000

    @property
    def hours_to_confirm(self) -> float:
        """Hours between the tag and the lower timeframe confirming."""
        return max(self.ltf_fvg_ms - self.tagged_ms, 0) / 3_600_000


@dataclass(slots=True)
class SimulationReport:
    """Aggregated funnel and frequency statistics for one replay."""

    symbols: tuple[str, ...]
    timeframe: str
    ltf_timeframe: str
    min_fvg_pct: float
    max_stop_pct: float = 0.0
    start: datetime | None = None
    end: datetime | None = None
    evaluations: int = 0

    # HTF funnel
    detected: int = 0                 # structural order blocks found
    fvg_rejected: int = 0             # no gap at all
    displacement_rejected: int = 0    # gap below min_fvg_pct
    pd_rejected: int = 0              # wrong half of the dealing range
    stop_width_rejected: int = 0      # stop wider than max_stop_pct
    risk_rejected: int = 0            # not sizeable
    watchlist: int = 0                # accepted onto the watchlist

    # Zone lifecycle
    tagged: int = 0
    tp_before_tag: int = 0
    structure_break: int = 0
    expired: int = 0
    still_open: int = 0
    confirmed: int = 0                # LTF-confirmed entries

    # LTF rejection breakdown
    ltf_rejections: Counter[str] = field(default_factory=Counter)
    entries: list[ConfirmedEntry] = field(default_factory=list)
    per_symbol: Counter[str] = field(default_factory=Counter)

    @property
    def days(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return max((self.end - self.start).total_seconds() / 86_400.0, 0.0)

    @property
    def weeks(self) -> float:
        return self.days / 7.0

    @property
    def signals_per_day(self) -> float:
        return self.confirmed / self.days if self.days > 0 else 0.0

    @property
    def signals_per_week(self) -> float:
        return self.confirmed / self.weeks if self.weeks > 0 else 0.0

    @property
    def signals_per_day_per_asset(self) -> float:
        assets = max(len(self.symbols), 1)
        return self.signals_per_day / assets

    @property
    def watchlist_conversion_pct(self) -> float:
        """Share of detected blocks that survived every HTF filter."""
        return self.watchlist / self.detected * 100.0 if self.detected else 0.0

    @property
    def confirmation_rate_pct(self) -> float:
        """Share of watched zones the lower timeframe actually confirmed."""
        return self.confirmed / self.watchlist * 100.0 if self.watchlist else 0.0

    def funnel_line(self) -> str:
        """One-line funnel for the log, in filter order."""
        return (
            f"detected={self.detected}, "
            f"fvg_rejected={self.fvg_rejected + self.displacement_rejected}, "
            f"pd_rejected={self.pd_rejected}, "
            f"stop_width_rejected={self.stop_width_rejected}, "
            f"final_watchlist={self.watchlist}"
        )

    def render(self) -> str:
        """Full human-readable report."""
        span = (
            f"{self.start:%Y-%m-%d} to {self.end:%Y-%m-%d}"
            if self.start and self.end
            else "unknown span"
        )
        width = 66
        lines = [
            "=" * width,
            "HISTORICAL SIMULATION — v3.1 SMC pipeline",
            "=" * width,
            f"Assets            : {len(self.symbols)} ({', '.join(self.symbols)})",
            f"Timeframes        : {self.timeframe} structure / {self.ltf_timeframe} entry",
            f"Period            : {span}  ({self.days:.1f} days, {self.weeks:.1f} weeks)",
            f"HTF evaluations   : {self.evaluations:,}",
            "",
            "-- HTF funnel " + "-" * (width - 14),
            f"{'Order blocks detected':<40}{self.detected:>10}",
            f"{'  rejected: no FVG':<40}{self.fvg_rejected:>10}"
            f"{self._pct(self.fvg_rejected, self.detected):>14}",
            f"{'  rejected: FVG < ' + f'{self.min_fvg_pct:g}%':<40}"
            f"{self.displacement_rejected:>10}"
            f"{self._pct(self.displacement_rejected, self.detected):>14}",
            f"{'  rejected: premium/discount':<40}{self.pd_rejected:>10}"
            f"{self._pct(self.pd_rejected, self.detected):>14}",
            f"{'  rejected: stop wider than ' + f'{self.max_stop_pct:g}%':<40}"
            f"{self.stop_width_rejected:>10}"
            f"{self._pct(self.stop_width_rejected, self.detected):>14}",
            f"{'  rejected: not sizeable':<40}{self.risk_rejected:>10}"
            f"{self._pct(self.risk_rejected, self.detected):>14}",
            f"{'Converted to watchlist':<40}{self.watchlist:>10}"
            f"{self._pct(self.watchlist, self.detected):>14}",
            "",
            "-- Zone lifecycle " + "-" * (width - 18),
            f"{'Tagged (price returned to zone)':<40}{self.tagged:>10}"
            f"{self._pct(self.tagged, self.watchlist):>14}",
            f"{'  invalidated: TP hit before tag':<40}{self.tp_before_tag:>10}"
            f"{self._pct(self.tp_before_tag, self.watchlist):>14}",
            f"{'  invalidated: HTF structure break':<40}{self.structure_break:>10}"
            f"{self._pct(self.structure_break, self.watchlist):>14}",
            f"{'  invalidated: expired':<40}{self.expired:>10}"
            f"{self._pct(self.expired, self.watchlist):>14}",
            f"{'  still open at end of data':<40}{self.still_open:>10}"
            f"{self._pct(self.still_open, self.watchlist):>14}",
            "",
            "-- LTF confirmation " + "-" * (width - 20),
            f"{'CONFIRMED ENTRIES':<40}{self.confirmed:>10}"
            f"{self._pct(self.confirmed, self.watchlist):>14}",
        ]

        for stage, count in self.ltf_rejections.most_common():
            lines.append(f"{'  unconfirmed: ' + stage:<40}{count:>10}")

        lines += [
            "",
            "-- Signal frequency " + "-" * (width - 20),
            f"{'Entries per day (all assets)':<40}{self.signals_per_day:>10.2f}",
            f"{'Entries per week (all assets)':<40}{self.signals_per_week:>10.2f}",
            f"{'Entries per day per asset':<40}{self.signals_per_day_per_asset:>10.3f}",
        ]

        if self.per_symbol:
            lines += ["", "-- Entries by asset " + "-" * (width - 20)]
            for symbol in self.symbols:
                count = self.per_symbol.get(symbol, 0)
                per_week = count / self.weeks if self.weeks > 0 else 0.0
                lines.append(f"{'  ' + symbol:<40}{count:>10}{per_week:>13.2f}/wk")

        lines.append("=" * width)
        return "\n".join(lines)

    @staticmethod
    def _pct(part: int, whole: int) -> str:
        return f"({part / whole * 100:.1f}%)" if whole else ""

    def render_entries(self, *, detailed: bool = True) -> str:
        """Every confirmed entry, laid out so each one can be checked by hand.

        Prices are rendered per symbol at a width that suits its magnitude, so a
        BTC level and an XRP level are both readable in the same table.
        """
        if not self.entries:
            return "No confirmed entries in this period."

        width = 118
        lines = [
            "=" * width,
            f"CONFIRMED ENTRIES — {len(self.entries)} setups, "
            f"{self.timeframe} structure confirmed on {self.ltf_timeframe}",
            "=" * width,
            "All times UTC, at the OPEN of the candle named.",
            "",
            f"{'#':>2}  {'SYMBOL':<10}{'DIR':<6}{'OB ZONE (low - high)':<28}"
            f"{'FVG%':>6}{'FIB':>6}  {'BLOCK FORMED':<16}{'TAGGED':<16}{'ENTRY TRIGGER':<16}",
            "-" * width,
        ]

        for index, entry in enumerate(sorted(self.entries, key=lambda e: e.choch_ms), 1):
            zone = (
                f"{_money(entry.zone_low)} - {_money(entry.zone_high)}"
            )
            lines.append(
                f"{index:>2}  {entry.symbol:<10}{entry.direction.value:<6}{zone:<28}"
                f"{entry.fvg_pct:>6.2f}{entry.fib_level:>6.2f}  "
                f"{entry.block_at:%m-%d %H:%M}    "
                f"{entry.tagged_at:%m-%d %H:%M}    "
                f"{entry.confirmed_at:%m-%d %H:%M}"
            )

        if not detailed:
            lines.append("=" * width)
            return "\n".join(lines)

        lines += ["", "-- Levels and timing " + "-" * (width - 21), ""]
        for index, entry in enumerate(sorted(self.entries, key=lambda e: e.choch_ms), 1):
            lines += [
                f"[{index}] {entry.symbol} {entry.direction.value}"
                f"   ({entry.zone_half} half, fib {entry.fib_level:.2f})",
                f"     structure : {entry.timeframe} block at {entry.block_at:%Y-%m-%d %H:%M}"
                f", zone {_money(entry.zone_low)} - {_money(entry.zone_high)}"
                f", displacement {entry.fvg_pct:.2f}%",
                f"     tagged    : {entry.tagged_at:%Y-%m-%d %H:%M}"
                f"  ({entry.hours_waiting:.0f}h after the block formed)",
                f"     CHoCH     : {entry.ltf_timeframe} close at {entry.choch_at:%Y-%m-%d %H:%M}",
                f"     LTF FVG   : {entry.confirmed_at:%Y-%m-%d %H:%M}"
                f"  ({entry.ltf_fvg_pct:.2f}%,"
                f" {entry.hours_to_confirm:.1f}h after the tag)",
                f"     order     : entry {_money(entry.entry)}"
                f"  stop {_money(entry.stop_loss)}"
                f"  target {_money(entry.take_profit)}"
                f"   risk {entry.risk_pct:.2f}%  R:R 1:{entry.reward_ratio:.0f}"
                f"  qty {entry.quantity:,.4f}".rstrip("0").rstrip("."),
                "",
            ]

        lines.append("=" * width)
        return "\n".join(lines)

    def entries_csv(self) -> str:
        """The same entries as CSV, for pasting into a spreadsheet."""
        header = (
            "symbol,direction,zone_low,zone_high,fvg_pct,fib_level,zone_half,"
            "block_utc,tagged_utc,choch_utc,ltf_fvg_utc,ltf_fvg_pct,"
            "entry,stop_loss,take_profit,risk_pct,reward_ratio,quantity"
        )
        rows = [header]
        for e in sorted(self.entries, key=lambda x: x.choch_ms):
            rows.append(
                f"{e.symbol},{e.direction.value},{e.zone_low:.10g},{e.zone_high:.10g},"
                f"{e.fvg_pct:.4f},{e.fib_level:.4f},{e.zone_half},"
                f"{e.block_at:%Y-%m-%d %H:%M},{e.tagged_at:%Y-%m-%d %H:%M},"
                f"{e.choch_at:%Y-%m-%d %H:%M},{e.confirmed_at:%Y-%m-%d %H:%M},"
                f"{e.ltf_fvg_pct:.4f},{e.entry:.10g},{e.stop_loss:.10g},"
                f"{e.take_profit:.10g},{e.risk_pct:.4f},{e.reward_ratio:.2f},"
                f"{e.quantity:.10g}"
            )
        return "\n".join(rows)


def _confirm_from_tag(
    ltf: pd.DataFrame,
    zone: WatchedZone,
    *,
    tagged_ms: int,
    strategy_timeframe: str,
    confirm_window: int,
    horizon: int,
    min_fvg_pct: float,
    swing_strength: int,
) -> tuple[LtfTrigger | None, str]:
    """Search forward from a tag for an LTF confirmation.

    Only candles that closed at or after the tag are visible, so the simulation
    cannot confirm an entry using a turn that happened before price arrived.

    Returns the trigger itself rather than a flag, so the caller can record the
    exact CHoCH and gap timestamps an entry was accepted on — without them a
    reported entry cannot be checked against a chart.
    """
    timestamps = ltf["timestamp"].to_numpy()
    start = int(timestamps.searchsorted(tagged_ms, side="left"))
    if start >= len(ltf):
        return None, "no LTF data after the tag"

    last_reason = "no LTF data after the tag"
    for position in range(start, min(start + horizon, len(ltf))):
        prefix = ltf.iloc[: position + 1]
        if len(prefix) < confirm_window // 4:
            continue
        trigger, rejection = confirm_entry(
            prefix,
            zone,
            timeframe=strategy_timeframe,
            strength=swing_strength,
            window=confirm_window,
            min_fvg_pct=min_fvg_pct,
        )
        if trigger is not None:
            return trigger, "confirmed"
        if rejection is not None:
            last_reason = rejection.stage
    return None, last_reason


def simulate(
    frames: Mapping[str, pd.DataFrame],
    ltf_frames: Mapping[str, pd.DataFrame],
    strategy: OrderBlockStrategy,
    *,
    timeframe: str,
    ltf_timeframe: str,
    swing_strength: int,
    confirm_window: int,
    ltf_min_fvg_pct: float,
    max_zone_age_ms: int | None = None,
    confirm_horizon: int = DEFAULT_CONFIRM_HORIZON,
) -> SimulationReport:
    """Replay the whole pipeline over stored candles and report the funnel."""
    symbols = tuple(frames)
    report = SimulationReport(
        symbols=symbols,
        timeframe=timeframe,
        ltf_timeframe=ltf_timeframe,
        min_fvg_pct=strategy.min_fvg_pct,
        max_stop_pct=strategy.max_stop_pct,
    )

    starts: list[int] = []
    ends: list[int] = []

    for symbol, htf in frames.items():
        if htf.empty:
            continue
        starts.append(int(htf["timestamp"].iloc[0]))
        ends.append(int(htf["timestamp"].iloc[-1]))
        ltf = ltf_frames.get(symbol, pd.DataFrame(columns=htf.columns))

        begin = max(strategy.required_candles, 3)
        for index in range(begin, len(htf) + 1):
            window = htf.iloc[:index]
            result = strategy.evaluate(window, symbol, timeframe)
            report.evaluations += 1
            stage = result.stage

            if not stage.reached_order_block:
                continue  # never formed a structure — not a detection

            report.detected += 1
            if stage is FilterStage.FVG:
                report.fvg_rejected += 1
                continue
            if stage is FilterStage.DISPLACEMENT:
                report.displacement_rejected += 1
                continue
            if stage is FilterStage.PREMIUM_DISCOUNT:
                report.pd_rejected += 1
                continue
            if stage is FilterStage.STOP_WIDTH:
                report.stop_width_rejected += 1
                continue
            if stage is FilterStage.RISK:
                report.risk_rejected += 1
                continue

            signal = result.signal
            assert signal is not None
            report.watchlist += 1

            zone = WatchedZone.from_order_block(
                symbol,
                timeframe,
                signal.block,
                entry=signal.plan.entry,
                stop_loss=signal.plan.stop_loss,
                take_profit=signal.plan.take_profit,
                quantity=signal.plan.quantity,
                created_ms=int(window["timestamp"].iloc[-1]),
            )

            outcome = _replay_zone(
                zone,
                htf.iloc[index:],
                max_zone_age_ms=max_zone_age_ms,
            )

            if outcome is InvalidationReason.TP_BEFORE_TAG:
                report.tp_before_tag += 1
                continue
            if outcome is InvalidationReason.STRUCTURE_BREAK:
                report.structure_break += 1
                continue
            if outcome is InvalidationReason.EXPIRED:
                report.expired += 1
                continue
            if zone.state is not WatchState.TAGGED or zone.tagged_ms is None:
                report.still_open += 1
                continue

            report.tagged += 1
            trigger, reason = _confirm_from_tag(
                ltf,
                zone,
                tagged_ms=zone.tagged_ms,
                strategy_timeframe=ltf_timeframe,
                confirm_window=confirm_window,
                horizon=confirm_horizon,
                min_fvg_pct=ltf_min_fvg_pct,
                swing_strength=swing_strength,
            )
            if trigger is not None:
                report.confirmed += 1
                report.per_symbol[symbol] += 1
                report.entries.append(
                    ConfirmedEntry(
                        symbol=symbol,
                        direction=zone.direction,
                        timeframe=timeframe,
                        ltf_timeframe=ltf_timeframe,
                        zone_low=zone.zone_low,
                        zone_high=zone.zone_high,
                        fvg_pct=zone.fvg_pct,
                        fib_level=zone.fib_level,
                        zone_half=(
                            signal.block.zone.value
                            if signal.block.zone is not None
                            else "unknown"
                        ),
                        block_ms=zone.block_timestamp,
                        created_ms=zone.created_ms,
                        tagged_ms=zone.tagged_ms,
                        choch_ms=trigger.choch_timestamp,
                        ltf_fvg_ms=trigger.fvg_timestamp,
                        ltf_fvg_pct=trigger.fvg_pct,
                        entry=zone.entry,
                        stop_loss=zone.stop_loss,
                        take_profit=zone.take_profit,
                        quantity=zone.quantity,
                    )
                )
            else:
                report.ltf_rejections[reason] += 1

    if starts and ends:
        report.start = datetime.fromtimestamp(min(starts) / 1000, tz=timezone.utc)
        report.end = datetime.fromtimestamp(max(ends) / 1000, tz=timezone.utc)

    return report


def _replay_zone(
    zone: WatchedZone,
    future: pd.DataFrame,
    *,
    max_zone_age_ms: int | None,
) -> InvalidationReason | None:
    """Advance one zone through the candles that followed it.

    Returns the reason it died, or ``None`` if it survived to be tagged or is
    still waiting when the data runs out.
    """
    if future.empty:
        return None

    for timestamp, high, low, close in future[
        ["timestamp", "high", "low", "close"]
    ].to_numpy():
        ts = int(timestamp)

        if zone.state is WatchState.PENDING and zone.reached_take_profit(
            float(high), float(low)
        ):
            return InvalidationReason.TP_BEFORE_TAG

        if zone.broke_structure(float(close)):
            return InvalidationReason.STRUCTURE_BREAK

        if zone.state is WatchState.PENDING and zone.touched(float(high), float(low)):
            zone.state = WatchState.TAGGED
            zone.tagged_ms = ts
            return None  # hand over to lower-timeframe confirmation

        if max_zone_age_ms is not None and zone.age_ms(ts) > max_zone_age_ms:
            return InvalidationReason.EXPIRED

    return None


def summarise_symbols(frames: Mapping[str, pd.DataFrame]) -> str:
    """One-line description of the dataset a report was built from."""
    if not frames:
        return "no data"
    sizes: Sequence[int] = [len(f) for f in frames.values()]
    return f"{len(frames)} symbols, {min(sizes)}-{max(sizes)} candles each"
