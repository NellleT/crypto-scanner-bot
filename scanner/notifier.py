"""Telegram notification delivery.

Uses the Bot API over plain ``requests`` — synchronous, dependency-light, and a
natural fit for a polling scanner that has no other async work to interleave.
Messages use HTML parse mode because escaping is trivial and unambiguous
compared with MarkdownV2.
"""

from __future__ import annotations

import html
import logging
import threading
from typing import Any, Callable, Final, Protocol, Sequence

import requests

from scanner.strategy import TradeSignal

logger: Final[logging.Logger] = logging.getLogger(__name__)

_API_BASE: Final[str] = "https://api.telegram.org"
_MAX_MESSAGE_LENGTH: Final[int] = 4096


def format_price(price: float) -> str:
    """Render a price with a sensible number of decimals for its magnitude.

    Fallback for when the exchange has not supplied tick-size metadata.
    """
    if price >= 1_000:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:,.4f}"
    if price >= 0.01:
        return f"{price:.6f}"
    return f"{price:.8f}"


def humanize_price(price: float, price_text: str | None = None) -> str:
    """Format a price for display, preferring the venue's own tick precision.

    ``price_text`` is the exchange-formatted string (see
    :meth:`scanner.exchange.MarketDataClient.price_to_precision`). It carries
    the correct number of decimals for the instrument — which matters a great
    deal for sub-cent alts — but no digit grouping, so a BTC print arrives as
    ``118700``. We keep its decimal count and add thousands separators.
    """
    if price_text is None:
        return format_price(price)
    try:
        value = float(price_text)
    except ValueError:
        return price_text
    return f"{value:,.{_decimals(price_text)}f}"


def _decimals(text: str) -> int:
    """Number of digits after the decimal point in a formatted number."""
    return len(text.partition(".")[2])


#: Quote currencies that are pegged to the dollar closely enough to print "$".
_USD_QUOTES: Final[frozenset[str]] = frozenset(
    {"USDT", "USDC", "USD", "BUSD", "DAI", "FDUSD", "TUSD", "USDP"}
)


def quote_prefix(symbol: str) -> str:
    """Currency marker for prices in ``symbol``.

    ``$`` for dollar-pegged quotes, otherwise the quote code — a BTC-quoted pair
    priced in "$" would be simply wrong.
    """
    _, _, quote = symbol.partition("/")
    quote = quote.split(":")[0].upper()  # strip any settlement suffix
    if quote in _USD_QUOTES:
        return "$"
    return f"{quote} " if quote else ""


def format_price_group(
    values: Sequence[float],
    texts: Sequence[str | None] | None = None,
) -> list[str]:
    """Format related prices with one shared decimal count.

    Entry, stop and targets are read as a set and compared against each other,
    so they must line up. The venue's ``price_to_precision`` strips trailing
    zeros independently per value, which would otherwise render a stop of
    1898.10 as ``1898.1`` beside an entry of ``1925.91``.
    """
    texts = list(texts or [None] * len(values))
    decimals = 0
    for value, text in zip(values, texts):
        source = text if text is not None else format_price(value).replace(",", "")
        decimals = max(decimals, _decimals(source))

    formatted: list[str] = []
    for value, text in zip(values, texts):
        numeric = value
        if text is not None:
            try:
                numeric = float(text)
            except ValueError:
                formatted.append(text)
                continue
        formatted.append(f"{numeric:,.{decimals}f}")
    return formatted


def align_decimals(value: float, value_text: str | None, reference: str) -> str:
    """Format ``value`` with at least as many decimals as ``reference``.

    The venue's ``price_to_precision`` strips trailing zeros, so an SMA of
    56.00 renders as ``56`` next to a price of ``58.11``. Both are prices of the
    same instrument and the reader compares them directly, so a ragged pair of
    decimal counts just looks like an error.
    """
    numeric = value
    decimals = _decimals(reference)
    if value_text is not None:
        try:
            numeric = float(value_text)
            decimals = max(decimals, _decimals(value_text))
        except ValueError:
            return value_text
    return f"{numeric:,.{decimals}f}"


def format_volume(volume: float) -> str:
    """Compact, readable volume.

    Base-asset volumes span an enormous range: a few hundred LTC against
    trillions of units of a sub-cent meme coin. The trillions tier is not
    hypothetical — without it those pairs render as ``9,100.00B``.
    """
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(volume) >= threshold:
            return f"{volume / threshold:,.2f}{suffix}"
    return f"{volume:,.2f}"


def build_risk_section(
    signal: TradeSignal,
    *,
    to_precision: Callable[[float], str | None] | None = None,
) -> list[str]:
    """Render the risk-management block.

    ``to_precision`` maps a price to the venue's tick-size string; when absent,
    magnitude-based formatting is used instead.
    """
    plan = signal.risk
    prices = plan.prices()
    texts = [to_precision(p) for p in prices] if to_precision else None
    entry, stop, *targets = format_price_group(prices, texts)
    unit = quote_prefix(signal.symbol)

    lines = [
        "🛡 <b>Risk Management</b>",
        f"• <b>Entry:</b> <code>{unit}{html.escape(entry)}</code>",
        f"• <b>Stop-Loss</b> ({html.escape(plan.stop_description)}): "
        f"<code>{unit}{html.escape(stop)}</code>"
        f"  <i>(Risk: {plan.risk_pct:.2f}%)</i>",
    ]
    for index, (take_profit, text) in enumerate(zip(plan.take_profits, targets), start=1):
        lines.append(
            # Signed price move, not the gain: a short's target is a fall.
            f"• <b>Take-Profit {index}</b> ({html.escape(take_profit.label)}): "
            f"<code>{unit}{html.escape(text)}</code>"
            f"  <i>({plan.price_move_pct(take_profit):+.2f}%)</i>"
        )
    return lines


def build_message(
    signal: TradeSignal,
    *,
    price_text: str | None = None,
    sma_text: str | None = None,
    to_precision: Callable[[float], str | None] | None = None,
) -> str:
    """Compose the HTML message body for a confirmed multi-factor signal."""
    price = humanize_price(signal.price, price_text)
    sma = align_decimals(signal.trend_sma, sma_text, price)
    closed_at = signal.candle_open_time.strftime("%Y-%m-%d %H:%M UTC")
    side = "above" if signal.sma_distance_pct >= 0 else "below"

    lines = [
        f"{signal.direction.emoji} <b>{html.escape(signal.direction.value)} SIGNAL</b>"
        f" · {html.escape(signal.symbol)}",
        "",
        f"<b>Pair:</b> <code>{html.escape(signal.symbol)}</code>",
        f"<b>Timeframe:</b> {html.escape(signal.timeframe)}",
        f"<b>Signal:</b> {html.escape(signal.direction.value)}"
        f" ({html.escape(signal.pattern.value)})",
        f"<b>Price:</b> <code>{html.escape(price)}</code>",
        "",
        f"<b>SMA {signal.trend_sma_period}:</b> <code>{html.escape(sma)}</code>"
        f"  <i>({abs(signal.sma_distance_pct):.2f}% {side})</i>",
        f"<b>Volume:</b> <code>{html.escape(format_volume(signal.volume))}</code>"
        f"  <i>({signal.volume_ratio:.2f}x the {signal.volume_sma_period}-period average"
        f" of {html.escape(format_volume(signal.volume_sma))})</i>",
        f"<b>VSA:</b> {signal.volume_expansion_ratio:.2f}x the engulfed candle's"
        f" {html.escape(format_volume(signal.previous_volume))}"
        f"  <i>(follow-through confirmed)</i>",
        "",
        *build_risk_section(signal, to_precision=to_precision),
        "",
        f"<i>Engulf ratio: {signal.engulf_ratio:.2f}x · Candle open {html.escape(closed_at)}</i>",
    ]
    message = "\n".join(lines)
    return message[:_MAX_MESSAGE_LENGTH]


class Notifier(Protocol):
    """Anything that can deliver a rendered alert."""

    def send_signal(
        self,
        signal: TradeSignal,
        *,
        price_text: str | None = ...,
        sma_text: str | None = ...,
        to_precision: Callable[[float], str | None] | None = ...,
    ) -> bool: ...

    def send_text(self, text: str) -> bool: ...


class TelegramNotifier:
    """Sends alerts to a single Telegram chat.

    Delivery failures are logged and reported via the return value rather than
    raised: a scanner should keep scanning even when notifications are down.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        *,
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        stop_event: threading.Event | None = None,
    ) -> None:
        if not bot_token or not chat_id:
            raise ValueError("TelegramNotifier requires both a bot token and a chat id.")
        self._chat_id = chat_id
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._stop_event = stop_event or threading.Event()
        self._endpoint = f"{_API_BASE}/bot{bot_token}/sendMessage"
        self._session = requests.Session()

    def send_signal(
        self,
        signal: TradeSignal,
        *,
        price_text: str | None = None,
        sma_text: str | None = None,
        to_precision: Callable[[float], str | None] | None = None,
    ) -> bool:
        return self.send_text(
            build_message(
                signal,
                price_text=price_text,
                sma_text=sma_text,
                to_precision=to_precision,
            )
        )

    def send_text(self, text: str) -> bool:
        """POST ``text`` to the chat. Returns ``True`` on confirmed delivery."""
        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        for attempt in range(1, self._max_retries + 2):
            if self._stop_event.is_set():
                return False
            try:
                response = self._session.post(
                    self._endpoint, json=payload, timeout=self._timeout_seconds
                )
            except requests.RequestException as exc:
                if not self._backoff(attempt, f"network error: {exc}"):
                    return False
                continue

            if response.status_code == 200:
                return True

            # 429: honour Telegram's own retry hint when present.
            if response.status_code == 429:
                retry_after = self._retry_after(response)
                logger.warning("Telegram rate limited; waiting %.1fs.", retry_after)
                if self._stop_event.wait(retry_after):
                    return False
                if attempt > self._max_retries:
                    return False
                continue

            if 500 <= response.status_code < 600:
                if not self._backoff(attempt, f"HTTP {response.status_code}"):
                    return False
                continue

            # 4xx other than 429 are permanent (bad token, bad chat id, bad HTML).
            logger.error(
                "Telegram rejected the message (HTTP %d): %s",
                response.status_code,
                response.text[:400],
            )
            return False

        return False

    def verify_credentials(self) -> bool:
        """Call ``getMe`` so misconfiguration surfaces at startup, not first signal."""
        url = self._endpoint.replace("/sendMessage", "/getMe")
        try:
            response = self._session.get(url, timeout=self._timeout_seconds)
        except requests.RequestException as exc:
            logger.error("Could not reach Telegram to verify credentials: %s", exc)
            return False

        if response.status_code != 200:
            logger.error(
                "Telegram credential check failed (HTTP %d): %s",
                response.status_code,
                response.text[:400],
            )
            return False

        username = (response.json().get("result") or {}).get("username", "unknown")
        logger.info("Telegram bot authenticated as @%s.", username)
        return True

    def close(self) -> None:
        self._session.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _backoff(self, attempt: int, reason: str) -> bool:
        """Sleep before the next attempt. Returns ``False`` when retries are done."""
        if attempt > self._max_retries:
            logger.error("Telegram delivery failed after %d attempt(s): %s", attempt, reason)
            return False
        delay = self._retry_backoff_seconds * (2 ** (attempt - 1))
        logger.warning(
            "Telegram delivery problem (attempt %d/%d): %s — retrying in %.1fs",
            attempt,
            self._max_retries + 1,
            reason,
            delay,
        )
        return not self._stop_event.wait(delay)

    def _retry_after(self, response: requests.Response) -> float:
        try:
            parameters = response.json().get("parameters") or {}
            return float(parameters.get("retry_after", self._retry_backoff_seconds))
        except (ValueError, AttributeError, TypeError):
            return self._retry_backoff_seconds


class ConsoleNotifier:
    """Dry-run notifier that logs alerts instead of sending them."""

    def send_signal(
        self,
        signal: TradeSignal,
        *,
        price_text: str | None = None,
        sma_text: str | None = None,
        to_precision: Callable[[float], str | None] | None = None,
    ) -> bool:
        price = humanize_price(signal.price, price_text)
        plan = signal.risk
        entry, stop, *targets = format_price_group(
            plan.prices(),
            [to_precision(p) for p in plan.prices()] if to_precision else None,
        )
        logger.info(
            "[DRY RUN] %s %s %s @ %s | SMA%d %s (%+.2f%%) | vol %.2fx avg, %.2fx prior bar "
            "| engulf %.2fx | SL %s (%s, risk %.2f%%) | TP %s",
            signal.direction.emoji,
            signal.symbol,
            signal.direction.value,
            price,
            signal.trend_sma_period,
            align_decimals(signal.trend_sma, sma_text, price),
            signal.sma_distance_pct,
            signal.volume_ratio,
            signal.volume_expansion_ratio,
            signal.engulf_ratio,
            stop,
            plan.stop_description,
            plan.risk_pct,
            " / ".join(
                f"{tp.label} {text}" for tp, text in zip(plan.take_profits, targets)
            ),
        )
        return True

    def send_text(self, text: str) -> bool:
        logger.info("[DRY RUN] %s", text.replace("\n", " | "))
        return True

    def verify_credentials(self) -> bool:
        return True

    def close(self) -> None:
        return None
