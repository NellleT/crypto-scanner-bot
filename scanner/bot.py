"""Scanner orchestration.

Owns the scan/sleep cycle, the multi-timeframe state machine and graceful
shutdown. One pass is four phases:

1. **Fetch** — HTF candles for every symbol, concurrently.
2. **Advance** — replay the new HTF candles against every live zone, tagging
   the ones price returned to and killing the ones that died.
3. **Detect** — evaluate the newest HTF structure and admit new zones.
4. **Confirm** — for tagged zones only, fetch the LTF and look for a change of
   character; that, and only that, produces an order.

A failure scanning one symbol never aborts the pass, and a failure of an entire
pass never terminates the process.
"""

from __future__ import annotations

import logging
import math
import signal
import threading
import time
from collections import Counter
from types import FrameType
from typing import Final

import pandas as pd

from scanner.analytics import SimulationReport, simulate
from scanner.config import Settings
from scanner.exchange import MarketDataClient, MarketDataError
from scanner.execution import ExecutionOrder, build_execution_order
from scanner.mtf import LtfTrigger, confirm_entry
from scanner.notifier import ConsoleNotifier, Notifier, TelegramNotifier
from scanner.strategy import FilterStage, OrderBlockStrategy, StrategyResult, TradeSignal
from scanner.watchlist import WatchedZone, Watchlist, WatchState

logger: Final[logging.Logger] = logging.getLogger(__name__)

#: Never sleep less than this between passes, even if clocks drift.
_MIN_SLEEP_SECONDS: Final[float] = 5.0


class ScannerBot:
    """Polls for extreme order blocks and confirms entries on a lower timeframe."""

    def __init__(
        self,
        settings: Settings,
        *,
        market_data: MarketDataClient | None = None,
        notifier: Notifier | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._settings = settings
        self._stop_event = stop_event or threading.Event()

        self._market_data = market_data or MarketDataClient(
            settings.exchange_id,
            timeout_seconds=settings.http_timeout_seconds,
            max_retries=settings.max_retries,
            retry_backoff_seconds=settings.retry_backoff_seconds,
            stop_event=self._stop_event,
        )

        self._notifier: Notifier = notifier or self._build_notifier(settings)

        self._strategy = OrderBlockStrategy(
            min_body_ratio=settings.min_body_ratio,
            min_fvg_pct=settings.min_fvg_pct,
            range_lookback=settings.range_lookback,
            require_extreme=settings.require_extreme,
            stop_buffer_pct=settings.stop_buffer_pct,
            max_stop_pct=settings.max_stop_pct,
            reward_ratio=settings.reward_ratio,
            account_equity=settings.account_equity,
            risk_per_trade_pct=settings.risk_per_trade_pct,
        )

        self._max_zone_age_ms: int | None = (
            int(settings.max_zone_age_hours * 3_600_000)
            if settings.max_zone_age_hours > 0
            else None
        )
        self._watchlist = Watchlist.load(
            settings.watchlist_file, max_age_ms=self._max_zone_age_ms
        )

        self._timeframe_seconds = self._market_data.timeframe_seconds(settings.timeframe)
        self._symbols: tuple[str, ...] = settings.symbols
        self._latest_close_epoch: float | None = None

    def _build_notifier(self, settings: Settings) -> Notifier:
        if settings.dry_run:
            logger.warning("DRY_RUN enabled — alerts will be logged, not sent.")
            return ConsoleNotifier()
        return TelegramNotifier(
            settings.telegram_bot_token,
            settings.telegram_chat_id,
            timeout_seconds=settings.http_timeout_seconds,
            max_retries=settings.max_retries,
            retry_backoff_seconds=settings.retry_backoff_seconds,
            stop_event=self._stop_event,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def install_signal_handlers(self) -> None:
        """Route SIGINT/SIGTERM into the cooperative stop event."""

        def _handle(signum: int, _frame: FrameType | None) -> None:
            logger.info("Received signal %s — shutting down after current step.", signum)
            self._stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle)
            except (ValueError, OSError, AttributeError):
                logger.debug("Could not install handler for signal %s.", sig)

    def startup_checks(self) -> None:
        """Validate exchange, timeframes, symbols and notifier before looping."""
        self._market_data.validate_timeframe(self._settings.timeframe)
        self._market_data.validate_timeframe(self._settings.ltf_timeframe)

        try:
            supported = self._market_data.filter_supported_symbols(self._settings.symbols)
        except MarketDataError as exc:
            logger.warning("Could not verify symbols against the exchange: %s", exc)
            supported = self._settings.symbols

        if not supported:
            raise MarketDataError(
                "None of the configured symbols are tradable on "
                f"{self._settings.exchange_id}."
            )
        if len(supported) != len(self._settings.symbols):
            dropped = set(self._settings.symbols) - set(supported)
            logger.warning("Dropped unsupported symbols: %s", ", ".join(sorted(dropped)))
        self._symbols = supported

        verifier = getattr(self._notifier, "verify_credentials", None)
        if callable(verifier) and not verifier():
            logger.warning(
                "Telegram credentials could not be verified — alerts may not be delivered."
            )

    def close(self) -> None:
        self.save_watchlist()
        for resource in (self._notifier, self._market_data):
            closer = getattr(resource, "close", None)
            if callable(closer):
                closer()

    def save_watchlist(self) -> None:
        """Persist zone state; a failure here must not lose the scan."""
        try:
            self._watchlist.prune()
            self._watchlist.save(self._settings.watchlist_file)
        except OSError as exc:
            logger.error("Could not save the watchlist: %s", exc)

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------
    def fetch_htf(self) -> dict[str, pd.DataFrame]:
        """HTF candles for every symbol, fetched concurrently."""
        jobs = [
            (symbol, self._settings.timeframe, self._settings.candle_limit)
            for symbol in self._symbols
        ]
        frames = self._market_data.fetch_many(
            jobs, max_workers=self._settings.max_workers
        )
        return {
            symbol: frame
            for (symbol, _tf), frame in frames.items()
            if not frame.empty
        }

    def fetch_ltf(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """LTF candles, fetched only for symbols with a tagged zone."""
        if not symbols:
            return {}
        jobs = [
            (symbol, self._settings.ltf_timeframe, self._settings.ltf_candle_limit)
            for symbol in symbols
        ]
        frames = self._market_data.fetch_many(
            jobs, max_workers=self._settings.max_workers
        )
        return {
            symbol: frame
            for (symbol, _tf), frame in frames.items()
            if not frame.empty
        }

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------
    def scan_symbol(self, symbol: str, df: pd.DataFrame) -> StrategyResult:
        """Evaluate one symbol's newest closed HTF candles."""
        close_epoch = (int(df["timestamp"].iloc[-1]) / 1000.0) + self._timeframe_seconds
        if self._latest_close_epoch is None or close_epoch > self._latest_close_epoch:
            self._latest_close_epoch = close_epoch
        return self._strategy.evaluate(df, symbol, self._settings.timeframe)

    def scan_once(self) -> list[LtfTrigger]:
        """Run one full multi-timeframe pass, dispatching confirmed entries."""
        funnel: Counter[str] = Counter()
        triggers: list[LtfTrigger] = []

        try:
            htf_frames = self.fetch_htf()
        except Exception:
            logger.exception("HTF fetch failed; skipping this pass.")
            return triggers

        if self._stop_event.is_set():
            return triggers

        # Phase 2 — advance every live zone against the new candles first, so a
        # zone that died is never re-confirmed by later logic in the same pass.
        for symbol, frame in htf_frames.items():
            for event in self._watchlist.update_from_htf(symbol, frame):
                level = logger.info if event.kind == "invalidated" else logger.info
                level("%s: zone %s — %s", symbol, event.kind, event.detail)
                funnel[f"zone_{event.kind}"] += 1

        # Phase 3 — admit new structures.
        for symbol, frame in htf_frames.items():
            if self._stop_event.is_set():
                break
            try:
                result = self.scan_symbol(symbol, frame)
            except Exception:
                logger.exception("Unexpected error while scanning %s.", symbol)
                funnel["error"] += 1
                continue

            funnel[result.stage.value] += 1
            if not result.matched:
                logger.debug("%s: %s", symbol, result.reason)
                continue

            signal_result = result.signal
            assert signal_result is not None
            if self._admit(signal_result, frame):
                logger.info("%s: %s", symbol, result.reason)
                funnel["zone_added"] += 1

        # Phase 4 — confirm tagged zones on the lower timeframe.
        tagged = [z for z in self._watchlist.active() if z.state is WatchState.TAGGED]
        if tagged and not self._stop_event.is_set():
            symbols = sorted({z.symbol for z in tagged})
            ltf_frames = self.fetch_ltf(symbols)
            for zone in tagged:
                frame = ltf_frames.get(zone.symbol)
                if frame is None:
                    continue
                trigger = self._try_confirm(zone, frame)
                if trigger is not None:
                    triggers.append(trigger)
                    funnel["confirmed"] += 1
                else:
                    funnel["awaiting_ltf"] += 1

        self._log_funnel(funnel)
        self.save_watchlist()
        return triggers

    def _admit(self, signal_result: TradeSignal, frame: pd.DataFrame) -> bool:
        """Add a validated setup to the watchlist. False if already tracked."""
        zone = WatchedZone.from_order_block(
            signal_result.symbol,
            signal_result.timeframe,
            signal_result.block,
            entry=signal_result.plan.entry,
            stop_loss=signal_result.plan.stop_loss,
            take_profit=signal_result.plan.take_profit,
            quantity=signal_result.plan.quantity,
            created_ms=int(frame["timestamp"].iloc[-1]),
        )
        return self._watchlist.add(zone)

    def _try_confirm(self, zone: WatchedZone, ltf: pd.DataFrame) -> LtfTrigger | None:
        """Look for an LTF trigger on a tagged zone and dispatch if found."""
        trigger, rejection = confirm_entry(
            ltf,
            zone,
            timeframe=self._settings.ltf_timeframe,
            strength=self._settings.swing_strength,
            window=self._settings.confirm_window,
            min_fvg_pct=self._settings.ltf_min_fvg_pct,
        )
        if trigger is None:
            if rejection is not None:
                logger.debug("%s: %s", zone.symbol, rejection.reason)
            return None

        self._watchlist.mark_triggered(zone, when_ms=trigger.fvg_timestamp)
        self._dispatch(zone, trigger)
        return trigger

    def _log_funnel(self, funnel: Counter[str]) -> None:
        """Report where candidates dropped out, so filtering is observable.

        Stages are counted from the typed :class:`FilterStage`, never by matching
        the reason text — several rejection messages share wording, and substring
        matching would silently conflate them.
        """
        if not funnel:
            return
        ordered = [stage.value for stage in FilterStage] + [
            "zone_added",
            "zone_tagged",
            "zone_invalidated",
            "awaiting_ltf",
            "confirmed",
            "error",
        ]
        logger.info(
            "Filter funnel: %s",
            ", ".join(f"{name}={funnel[name]}" for name in ordered if funnel[name])
            or "nothing evaluated",
        )
        counts = self._watchlist.counts()
        logger.info(
            "Watchlist: %d pending, %d tagged, %d triggered, %d invalidated",
            counts.get(WatchState.PENDING.value, 0),
            counts.get(WatchState.TAGGED.value, 0),
            counts.get(WatchState.TRIGGERED.value, 0),
            counts.get(WatchState.INVALIDATED.value, 0),
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def build_order(self, zone: WatchedZone) -> ExecutionOrder | None:
        """Render a confirmed zone as a routable Binance order.

        Failing to format an order must not suppress the alert — a human can act
        on the levels either way — so the error is logged and the alert still
        goes out without the routing block.
        """
        try:
            return build_execution_order_from_zone(
                zone,
                price_to_precision=self._market_data.price_to_precision,
                amount_to_precision=self._market_data.amount_to_precision,
            )
        except (ValueError, TypeError) as exc:
            logger.error("Could not build an execution order for %s: %s", zone.symbol, exc)
            return None

    def _dispatch(self, zone: WatchedZone, trigger: LtfTrigger) -> None:
        order = self.build_order(zone)
        logger.info(
            "%s %s ENTRY on %s | zone [%g, %g] | entry %s | SL %s | TP %s | "
            "LTF %s CHoCH + %.2f%% FVG",
            zone.direction.emoji,
            zone.direction.value,
            zone.symbol,
            zone.zone_low,
            zone.zone_high,
            order.entry if order else f"{zone.entry:g}",
            order.stop_loss if order else f"{zone.stop_loss:g}",
            order.take_profit if order else f"{zone.take_profit:g}",
            trigger.timeframe,
            trigger.fvg_pct,
        )

        def to_precision(value: float) -> str | None:
            return self._market_data.price_to_precision(zone.symbol, value)

        if not self._notifier.send_signal(
            zone, trigger=trigger, order=order, to_precision=to_precision
        ):
            logger.error("Failed to deliver alert for %s.", zone.symbol)

    # ------------------------------------------------------------------
    # Historical simulation
    # ------------------------------------------------------------------
    def simulate(self, *, candle_limit: int | None = None) -> SimulationReport:
        """Replay the pipeline over stored candles and report the funnel."""
        htf_limit = candle_limit or self._settings.candle_limit

        # Both timeframes must cover the SAME wall-clock span. A 15m frame needs
        # four times as many candles as a 1h one to do that, which is past every
        # venue's per-request cap — so the history fetch pages. Getting this
        # wrong makes every older zone look unconfirmable for want of data.
        htf_seconds = self._market_data.timeframe_seconds(self._settings.timeframe)
        ltf_seconds = self._market_data.timeframe_seconds(self._settings.ltf_timeframe)
        ltf_bars = int(htf_limit * htf_seconds / ltf_seconds) + self._settings.confirm_window

        logger.info(
            "Fetching %d %s and %d %s candles for %d symbols (%.1f days)...",
            htf_limit,
            self._settings.timeframe,
            ltf_bars,
            self._settings.ltf_timeframe,
            len(self._symbols),
            htf_limit * htf_seconds / 86_400,
        )

        htf: dict[str, pd.DataFrame] = {}
        ltf: dict[str, pd.DataFrame] = {}
        for symbol in self._symbols:
            if self._stop_event.is_set():
                break
            try:
                htf_frame = self._market_data.fetch_ohlcv_history(
                    symbol, self._settings.timeframe, bars=htf_limit
                )
                ltf_frame = self._market_data.fetch_ohlcv_history(
                    symbol, self._settings.ltf_timeframe, bars=ltf_bars
                )
            except MarketDataError as exc:
                logger.error("Skipping %s: %s", symbol, exc)
                continue
            if not htf_frame.empty:
                htf[symbol] = htf_frame
            if not ltf_frame.empty:
                ltf[symbol] = ltf_frame

        for symbol, frame in htf.items():
            partner = ltf.get(symbol)
            if partner is None or partner.empty:
                logger.warning("%s has no LTF data; entries cannot be confirmed.", symbol)
                continue
            htf_start = int(frame["timestamp"].iloc[0])
            ltf_start = int(partner["timestamp"].iloc[0])
            if ltf_start > htf_start:
                logger.warning(
                    "%s LTF history starts %.1f days after the HTF window; zones "
                    "before that cannot be confirmed and will understate entries.",
                    symbol,
                    (ltf_start - htf_start) / 86_400_000,
                )

        logger.info("Replaying %d HTF frames...", len(htf))
        return simulate(
            htf,
            ltf,
            self._strategy,
            timeframe=self._settings.timeframe,
            ltf_timeframe=self._settings.ltf_timeframe,
            swing_strength=self._settings.swing_strength,
            confirm_window=self._settings.confirm_window,
            ltf_min_fvg_pct=self._settings.ltf_min_fvg_pct,
            max_zone_age_ms=self._max_zone_age_ms,
        )

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------
    def seconds_until_next_scan(self, now: float | None = None) -> float:
        """Seconds to wait so the next pass lands just after the next candle close.

        Anchors to the newest candle timestamp actually returned by the exchange
        when one is known; otherwise falls back to epoch-aligned boundaries.
        """
        now = time.time() if now is None else now
        tf = float(self._timeframe_seconds)

        if self._latest_close_epoch is not None:
            next_close = self._latest_close_epoch
            if next_close <= now:
                periods = math.floor((now - next_close) / tf) + 1
                next_close += periods * tf
        else:
            next_close = (math.floor(now / tf) + 1) * tf

        return max(next_close - now + self._settings.poll_buffer_seconds, _MIN_SLEEP_SECONDS)

    def run_forever(self) -> None:
        """Scan on every candle close until interrupted. Never raises on scan errors."""
        logger.info("Scanner started — %s", self._settings.describe())
        logger.info(
            "Structure is read on %s: a block at [-3] before an impulse at [-2], "
            "validated by a fair value gap of at least %g%% and only taken from the "
            "%s half of the %d-bar dealing range.",
            self._settings.timeframe,
            self._settings.min_fvg_pct,
            "discount (long) / premium (short)",
            self._settings.range_lookback,
        )
        logger.info(
            "Entries are NOT blind: a validated zone is watched until price returns "
            "to it and %s prints a change of character with its own fair value gap. "
            "Zones die if the 1:%g target is reached first, if an %s candle closes "
            "beyond the distal edge, or after %g hours.",
            self._settings.ltf_timeframe,
            self._settings.reward_ratio,
            self._settings.timeframe,
            self._settings.max_zone_age_hours,
        )

        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                triggers = self.scan_once()
            except Exception:
                logger.exception("Scan pass failed; continuing after backoff.")
                if self._stop_event.wait(self._settings.retry_backoff_seconds):
                    break
                continue

            elapsed = time.monotonic() - started
            logger.info(
                "Pass complete: %d symbol(s) in %.1fs, %d confirmed entr%s.",
                len(self._symbols),
                elapsed,
                len(triggers),
                "y" if len(triggers) == 1 else "ies",
            )

            if self._stop_event.is_set():
                break

            delay = self.seconds_until_next_scan()
            logger.info("Next scan in %.0fs.", delay)
            if self._stop_event.wait(delay):
                break

        logger.info("Scanner stopped.")


def build_execution_order_from_zone(
    zone: WatchedZone,
    *,
    price_to_precision=None,
    amount_to_precision=None,
) -> ExecutionOrder:
    """Adapt a watched zone to the execution-order builder.

    The zone carries the levels that were computed when it was admitted, so an
    entry uses the price structure that was validated rather than one recomputed
    from whatever the market looks like at confirmation time.
    """
    from scanner.execution import ExecutionOrder as _Order
    from scanner.execution import to_binance_symbol

    def price(value: float) -> str:
        if price_to_precision is not None:
            rendered = price_to_precision(zone.symbol, value)
            if rendered is not None:
                return rendered
        return f"{value:.8f}".rstrip("0").rstrip(".") or "0"

    def amount(value: float) -> str:
        if amount_to_precision is not None:
            rendered = amount_to_precision(zone.symbol, value)
            if rendered is not None:
                return rendered
        return f"{value:.8f}".rstrip("0").rstrip(".") or "0"

    risk_per_unit = abs(zone.entry - zone.stop_loss)
    reward_ratio = (
        abs(zone.take_profit - zone.entry) / risk_per_unit if risk_per_unit > 0 else 0.0
    )
    return _Order(
        symbol=to_binance_symbol(zone.symbol),
        side=zone.direction.binance_side,
        quantity=amount(zone.quantity),
        entry=price(zone.entry),
        stop_loss=price(zone.stop_loss),
        take_profit=price(zone.take_profit),
        reward_ratio=round(reward_ratio, 4),
        risk_pct=0.0,
        risk_amount=zone.quantity * risk_per_unit,
    )
