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
from scanner.indicators import add_indicators, pandas_ta_backend
from scanner.notifier import (
    ConsoleNotifier,
    Notifier,
    TelegramNotifier,
    align_decimals,
    humanize_price,
)
from scanner.strategy import (
    EngulfingTrendStrategy,
    FilterStage,
    StrategyResult,
    TradeSignal,
)

logger: Final[logging.Logger] = logging.getLogger(__name__)

#: Never sleep less than this between passes, even if clocks drift.
_MIN_SLEEP_SECONDS: Final[float] = 5.0


class ScannerBot:
    """Polls a set of symbols for engulfing patterns and dispatches alerts."""

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

        self._strategy = EngulfingTrendStrategy(
            sma_period=settings.sma_period,
            volume_sma_period=settings.volume_sma_period,
            min_body_ratio=settings.min_body_ratio,
            require_volume_expansion=settings.require_volume_expansion,
            structural_lookback=settings.structural_lookback,
            stop_buffer_pct=settings.stop_buffer_pct,
            rr_targets=settings.rr_targets,
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

        # Indicators are computed only after the still-forming candle has been
        # dropped, so no moving average can repaint as that bar develops.
        enriched = add_indicators(
            df,
            sma_period=self._settings.sma_period,
            volume_sma_period=self._settings.volume_sma_period,
        )

        return self._strategy.evaluate(enriched, symbol, self._settings.timeframe)

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

        if self._last_alerted.get(symbol) == signal.candle.timestamp:
            logger.debug("%s: signal already reported for this candle.", symbol)
            return None

        self._last_alerted[symbol] = signal.candle.timestamp
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

    def _dispatch(self, result: TradeSignal) -> None:
        price_text = self._market_data.price_to_precision(result.symbol, result.price)
        sma_text = self._market_data.price_to_precision(result.symbol, result.trend_sma)
        price = humanize_price(result.price, price_text)
        logger.info(
            "%s %s on %s %s @ %s | %s %s (%+.2f%%) | volume %.2fx average | engulf %.2fx",
            result.direction.emoji,
            result.direction.value,
            result.symbol,
            result.timeframe,
            price,
            f"SMA{result.trend_sma_period}",
            align_decimals(result.trend_sma, sma_text, price),
            result.sma_distance_pct,
            result.volume_ratio,
            result.engulf_ratio,
        )
        def to_precision(value: float) -> str | None:
            return self._market_data.price_to_precision(result.symbol, value)

        if not self._notifier.send_signal(
            result,
            price_text=price_text,
            sma_text=sma_text,
            to_precision=to_precision,
        ):
            logger.error("Failed to deliver alert for %s.", result.symbol)
            # Clear the dedup marker so a pass that still sees this candle as the
            # latest closed one (an error-backoff retry, or a restart before the
            # next close) can re-send it. Once the candle rolls over the signal is
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
        logger.info("Indicator backend: %s", pandas_ta_backend())
        logger.info(
            "Candle duration %ds; alerts fire on the most recently CLOSED candle. "
            "LONG needs close > SMA%d and volume > VOL_SMA_%d with a bullish engulfing; "
            "SHORT is the mirror image.%s",
            self._timeframe_seconds,
            self._settings.sma_period,
            self._settings.volume_sma_period,
            " Volume must also expand over the engulfed candle (VSA)."
            if self._settings.require_volume_expansion
            else " VSA expansion check is DISABLED.",
        )
        logger.info(
            "Stops anchor to the %d-bar structural %s with a %g%% buffer; targets at %s.",
            self._settings.structural_lookback,
            "low (long) / high (short)",
            self._settings.stop_buffer_pct,
            ", ".join(f"1:{r:g}" for r in self._settings.rr_targets),
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
