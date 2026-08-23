"""Telegram notification delivery.

Uses the Bot API over plain ``requests`` — synchronous, dependency-light, and a
natural fit for a polling scanner with no other async work to interleave.
Messages use HTML parse mode because escaping is trivial and unambiguous
compared with MarkdownV2.
"""

from __future__ import annotations

import html
import logging
import threading
from typing import Any, Callable, Final, Protocol, Sequence

import requests

from scanner.execution import ExecutionOrder
from scanner.strategy import TradeSignal

logger: Final[logging.Logger] = logging.getLogger(__name__)

_API_BASE: Final[str] = "https://api.telegram.org"
_MAX_MESSAGE_LENGTH: Final[int] = 4096

#: Quote currencies pegged closely enough to the dollar to print "$".
_USD_QUOTES: Final[frozenset[str]] = frozenset(
    {"USDT", "USDC", "USD", "BUSD", "DAI", "FDUSD", "TUSD", "USDP"}
)


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


def format_money(amount: float) -> str:
    """Render a quote-currency amount — equity, risk budget, notional.

    Distinct from :func:`format_price`: an instrument price may need eight
    decimals for a sub-cent coin, but a cash amount is always money, and
    ``$100.0000`` reads like a bug.
    """
    return f"{amount:,.2f}"


def format_quantity(quantity: float) -> str:
    """Render an order quantity, which spans BTC fractions to millions of DOGE."""
    if quantity >= 1_000:
        return f"{quantity:,.2f}"
    if quantity >= 1:
        return f"{quantity:,.4f}"
    return f"{quantity:.8f}".rstrip("0").rstrip(".") or "0"


def _decimals(text: str) -> int:
    """Number of digits after the decimal point in a formatted number."""
    return len(text.partition(".")[2])


def quote_prefix(symbol: str) -> str:
    """Currency marker for prices in ``symbol``.

    ``$`` for dollar-pegged quotes, otherwise the quote code — a BTC-quoted pair
    priced in "$" would be simply wrong.
    """
    _, _, quote = symbol.partition("/")
    quote = quote.split(":")[0].upper()
    if quote in _USD_QUOTES:
        return "$"
    return f"{quote} " if quote else ""


def format_price_group(
    values: Sequence[float],
    texts: Sequence[str | None] | None = None,
) -> list[str]:
    """Format related prices with one shared decimal count.

    Entry, stop and target are read as a set and compared against each other, so
    they must line up. ``price_to_precision`` strips trailing zeros per value,
    which would otherwise render a stop of 1898.10 as ``1898.1`` beside an entry
    of ``1925.91``.
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


def build_message(
    signal: TradeSignal,
    *,
    order: ExecutionOrder | None = None,
    to_precision: Callable[[float], str | None] | None = None,
) -> str:
    """Compose the HTML alert for a validated order block.

    Every price is formatted as one group so entry, stop and target carry the
    same decimals — they are read against each other.
    """
    plan = signal.plan
    block = signal.block
    values = (*plan.prices(), block.proximal, block.distal)
    texts = [to_precision(v) for v in values] if to_precision else None
    entry, stop, target, proximal, distal = format_price_group(values, texts)
    unit = quote_prefix(signal.symbol)
    direction = signal.direction

    zone_low, zone_high = (distal, proximal) if direction.is_long else (proximal, distal)
    gap_pct = block.fvg.size_pct(plan.entry)
    confirmed = signal.confirmed_at.strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"{direction.emoji} <b>{html.escape(direction.value)} · ORDER BLOCK</b>"
        f" · {html.escape(signal.symbol)}",
        "",
        f"<b>Pair:</b> <code>{html.escape(signal.symbol)}</code>",
        f"<b>Timeframe:</b> {html.escape(signal.timeframe)}",
        f"<b>Setup:</b> {html.escape(direction.value)} order block, FVG validated",
        "",
        f"<b>OB zone:</b> <code>{unit}{html.escape(zone_low)}</code>"
        f" – <code>{unit}{html.escape(zone_high)}</code>",
        f"<b>Fair Value Gap:</b> {gap_pct:.2f}% of entry"
        f"  <i>(displacement confirmed)</i>",
        "",
        "📋 <b>Pending Limit Order</b>",
        f"• <b>Entry:</b> <code>{unit}{html.escape(entry)}</code>"
        f"  <i>(limit, at proximal edge)</i>",
        f"• <b>Stop-Loss:</b> <code>{unit}{html.escape(stop)}</code>"
        f"  <i>(distal ∓{plan.buffer_pct:g}%, risk {plan.risk_pct_of_entry:.2f}%)</i>",
        f"• <b>Take-Profit</b> (1:{plan.reward_ratio:g}): "
        f"<code>{unit}{html.escape(target)}</code>"
        f"  <i>({plan.take_profit_move_pct:+.2f}%)</i>",
        "",
        "🛡 <b>Position Sizing</b>",
        f"• <b>Quantity:</b> <code>{html.escape(format_quantity(plan.quantity))}</code>",
        f"• <b>Risk:</b> {plan.risk_pct:g}% of {unit}"
        f"{html.escape(format_money(plan.equity))}"
        f" = <code>{unit}{html.escape(format_money(plan.risk_amount))}</code>",
        f"• <b>Reward at target:</b> <code>{unit}"
        f"{html.escape(format_money(plan.reward_amount))}</code>",
        f"• <b>Notional:</b> {unit}{html.escape(format_money(plan.notional))}"
        f"  <i>({plan.leverage_required:.2f}x equity)</i>",
    ]

    if order is not None:
        lines += [
            "",
            f"<b>Route:</b> <code>{html.escape(order.side)} "
            f"{html.escape(order.quantity)} {html.escape(order.symbol)} "
            f"@ {html.escape(order.entry)}</code>",
        ]

    lines += [
        "",
        f"<i>Block opened {html.escape(block.open_time.strftime('%Y-%m-%d %H:%M UTC'))}"
        f" · confirmed {html.escape(confirmed)}</i>",
    ]

    return "\n".join(lines)[:_MAX_MESSAGE_LENGTH]


class Notifier(Protocol):
    """Anything that can deliver a rendered alert."""

    def send_signal(
        self,
        signal: TradeSignal,
        *,
        order: ExecutionOrder | None = ...,
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
        order: ExecutionOrder | None = None,
        to_precision: Callable[[float], str | None] | None = None,
    ) -> bool:
        return self.send_text(
            build_message(signal, order=order, to_precision=to_precision)
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
        order: ExecutionOrder | None = None,
        to_precision: Callable[[float], str | None] | None = None,
    ) -> bool:
        plan = signal.plan
        entry, stop, target = format_price_group(
            plan.prices(),
            [to_precision(v) for v in plan.prices()] if to_precision else None,
        )
        logger.info(
            "[DRY RUN] %s %s %s OB | entry %s | SL %s (risk %.2f%%) | TP 1:%g %s "
            "| qty %s | risk %.2f%% of equity | FVG %.2f%%",
            signal.direction.emoji,
            signal.symbol,
            signal.direction.value,
            entry,
            stop,
            plan.risk_pct_of_entry,
            plan.reward_ratio,
            target,
            format_quantity(plan.quantity),
            plan.risk_pct,
            signal.block.fvg.size_pct(plan.entry),
        )
        if order is not None:
            logger.info("[DRY RUN] route: %s", order.to_json())
        return True

    def send_text(self, text: str) -> bool:
        logger.info("[DRY RUN] %s", text.replace("\n", " | "))
        return True

    def verify_credentials(self) -> bool:
        return True

    def close(self) -> None:
        return None
