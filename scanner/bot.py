"""Scanner orchestration.

Owns the scan/sleep cycle, per-candle deduplication and graceful shutdown. The
loop is designed so that a failure scanning one symbol never aborts the pass,
and a failure of an entire pass never terminates the process.
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

from scanner.config import Settings
from scanner.exchange import MarketDataClient, MarketDataError
from scanner.execution import ExecutionOrder, build_execution_order
from scanner.notifier import ConsoleNotifier, Notifier, TelegramNotifier
from scanner.strategy import (
    FilterStage,
    OrderBlockStrategy,
    StrategyResult,
    TradeSignal,
)

logger: Final[logging.Logger] = logging.getLogger(__name__)

#: Never sleep less than this between passes, even if clocks drift.
_MIN_SLEEP_SECONDS: Final[float] = 5.0


class ScannerBot:
    """Polls a set of symbols for validated order blocks and dispatches alerts."""

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
            stop_buffer_pct=settings.stop_buffer_pct,
            reward_ratio=settings.reward_ratio,
            account_equity=settings.account_equity,
            risk_per_trade_pct=settings.risk_per_trade_pct,
        )

        self._timeframe_seconds = self._market_data.timeframe_seconds(settings.timeframe)
        self._symbols: tuple[str, ...] = settings.symbols

        # symbol -> open timestamp (ms) of the last candle we alerted on.
        self._last_alerted: dict[str, int] = {}
        # Close time (epoch seconds) of the newest closed candle we have seen.
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
                # Not on the main thread, or unsupported on this platform.
                logger.debug("Could not install handler for signal %s.", sig)

    def startup_checks(self) -> None:
        """Validate exchange, timeframe, symbols and notifier before looping."""
        self._market_data.validate_timeframe(self._settings.timeframe)

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
        for resource in (self._notifier, self._market_data):
            closer = getattr(resource, "close", None)
            if callable(closer):
                closer()

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------
    def scan_symbol(self, symbol: str) -> StrategyResult | None:
        """Evaluate one symbol's most recently closed candle.

        Returns the full :class:`StrategyResult` — including the stage at which
        a rejection occurred — or ``None`` when market data was unavailable.
        Deduplication and dispatch are handled by the caller.
        """
        try:
            df = self._market_data.fetch_ohlcv(
                symbol, self._settings.timeframe, self._settings.candle_limit
            )
        except MarketDataError as exc:
            logger.error("Skipping %s: %s", symbol, exc)
            return None

        if df.empty:
            return None

        # Track the newest closed candle so sleeps stay anchored to the venue's grid.
        close_epoch = (int(df["timestamp"].iloc[-1]) / 1000.0) + self._timeframe_seconds
        if self._latest_close_epoch is None or close_epoch > self._latest_close_epoch:
            self._latest_close_epoch = close_epoch

        # The frame holds only closed candles, so the gap that validates a block
        # cannot appear and vanish while the newest bar is still developing.
        return self._strategy.evaluate(df, symbol, self._settings.timeframe)

    def scan_once(self) -> list[TradeSignal]:
        """Run one full pass over all symbols, dispatching alerts as they are found."""
        found: list[TradeSignal] = []
        funnel: Counter[str] = Counter()

        for symbol in self._symbols:
            if self._stop_event.is_set():
                break
            try:
                result = self.scan_symbol(symbol)
            except Exception:  # a bad symbol must never kill the pass
                logger.exception("Unexpected error while scanning %s.", symbol)
                funnel["error"] += 1
                continue

            if result is None:
                funnel["error"] += 1
            else:
                funnel[result.stage.value] += 1
                signal = self._accept(symbol, result)
                if signal is not None:
                    found.append(signal)
                    self._dispatch(signal)

            # Gentle spacing on top of CCXT's own rate limiter.
            if self._settings.request_delay_seconds > 0 and self._stop_event.wait(
                self._settings.request_delay_seconds
            ):
                break

        self._log_funnel(funnel)
        return found

    def _accept(self, symbol: str, result: StrategyResult) -> TradeSignal | None:
        """Return the signal to dispatch, or ``None`` if rejected or already sent."""
        if not result.matched:
            logger.debug("%s: %s", symbol, result.reason)
            return None

        signal = result.signal
        assert signal is not None  # guaranteed by result.matched

        # Keyed on the order block candle, not the confirmation candle: the same
        # block stays the newest structure for as long as price has not returned
        # to it, and it must only be alerted once.
        block_timestamp = signal.block.candle.timestamp
        if self._last_alerted.get(symbol) == block_timestamp:
            logger.debug("%s: order block already reported.", symbol)
            return None

        self._last_alerted[symbol] = block_timestamp
        return signal

    def _log_funnel(self, funnel: Counter[str]) -> None:
        """Report where symbols dropped out, so filter tuning is observable.

        Stages are counted from the typed :class:`FilterStage`, not by matching
        the reason text — the trend and volume messages both read "is not
        above", so substring matching would silently conflate them.
        """
        if not funnel:
            return
        # Derived from the enum's declaration order, so a new stage appears here
        # automatically instead of being silently dropped from the report.
        ordered = [stage.value for stage in FilterStage] + ["error"]
        logger.info(
            "Filter funnel: %s",
            ", ".join(f"{stage}={funnel[stage]}" for stage in ordered if funnel[stage])
            or "nothing evaluated",
        )

    def build_order(self, signal: TradeSignal) -> ExecutionOrder | None:
        """Render a signal as a routable Binance order, or ``None`` if it cannot be.

        Failing to format an order must not suppress the alert — a human can act
        on the levels either way — so the error is logged and the alert still
        goes out without the routing block.
        """
        try:
            return build_execution_order(
                signal.symbol,
                signal.plan,
                signal.block,
                price_to_precision=self._market_data.price_to_precision,
                amount_to_precision=self._market_data.amount_to_precision,
            )
        except (ValueError, TypeError) as exc:
            logger.error("Could not build an execution order for %s: %s", signal.symbol, exc)
            return None

    def _dispatch(self, result: TradeSignal) -> None:
        plan = result.plan
        order = self.build_order(result)

        logger.info(
            "%s %s order block on %s %s | entry %s | SL %s (risk %.2f%%) | "
            "TP 1:%g %s | qty %s | FVG %.2f%%",
            result.direction.emoji,
            result.direction.value,
            result.symbol,
            result.timeframe,
            order.entry if order else f"{plan.entry:g}",
            order.stop_loss if order else f"{plan.stop_loss:g}",
            plan.risk_pct_of_entry,
            plan.reward_ratio,
            order.take_profit if order else f"{plan.take_profit:g}",
            order.quantity if order else f"{plan.quantity:g}",
            result.block.fvg.size_pct(plan.entry),
        )

        def to_precision(value: float) -> str | None:
            return self._market_data.price_to_precision(result.symbol, value)

        if not self._notifier.send_signal(result, order=order, to_precision=to_precision):
            logger.error("Failed to deliver alert for %s.", result.symbol)
            # Clear the dedup marker so a pass that still sees this structure as
            # the newest one (an error-backoff retry, or a restart before the next
            # close) can re-send it. Once the candle rolls over the signal is
            # intentionally abandoned rather than delivered late.
            self._last_alerted.pop(result.symbol, None)

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
                # Advance in whole timeframes to the first close still ahead of us.
                periods = math.floor((now - next_close) / tf) + 1
                next_close += periods * tf
        else:
            next_close = (math.floor(now / tf) + 1) * tf

        return max(next_close - now + self._settings.poll_buffer_seconds, _MIN_SLEEP_SECONDS)

    def run_forever(self) -> None:
        """Scan on every candle close until interrupted. Never raises on scan errors."""
        logger.info("Scanner started — %s", self._settings.describe())
        logger.info(
            "Candle duration %ds; structures are read from the most recently CLOSED "
            "candles. LONG needs a bearish block at [-3] before a bullish impulse at "
            "[-2], validated by low[-1] > high[-3]; SHORT is the mirror image.",
            self._timeframe_seconds,
        )
        logger.info(
            "Entry rests at the block's proximal edge; stop sits %g%% beyond the "
            "distal edge; target at 1:%g. Size risks %g%% of %.2f equity.",
            self._settings.stop_buffer_pct,
            self._settings.reward_ratio,
            self._settings.risk_per_trade_pct,
            self._settings.account_equity,
        )

        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                signals = self.scan_once()
            except Exception:  # last line of defence — the loop must survive
                logger.exception("Scan pass failed; continuing after backoff.")
                if self._stop_event.wait(self._settings.retry_backoff_seconds):
                    break
                continue

            elapsed = time.monotonic() - started
            logger.info(
                "Pass complete: %d symbol(s) in %.1fs, %d signal(s).",
                len(self._symbols),
                elapsed,
                len(signals),
            )

            if self._stop_event.is_set():
                break

            delay = self.seconds_until_next_scan()
            logger.info("Next scan in %.0fs.", delay)
            if self._stop_event.wait(delay):
                break

        logger.info("Scanner stopped.")
