"""Unit tests for V5.0 paper trading.

The hand-traced fixture from test_trend drives the account day by day: the
day-5 close breaks out (BUY queued), day 6's open fills it, the day-8 close
breaks the exit channel (SELL queued), and day 9's open sells.
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import pytest

from scanner.exchange import MarketDataError
from scanner.paper import alerts, markdown, replay, save, status
from tests.test_trend import BARS, P, _BASE_MS, _DAY_MS, frame
from tests.test_trend_backtest import NET_R

SYM = "TEST/USDT"


def state_at(bars: int, *, start_day: int = 0):
    """The paper account as a run would see it once day ``bars - 1`` closed."""
    return replay({SYM: frame(BARS[:bars])}, start_ms=_BASE_MS + start_day * _DAY_MS,
                  equity=10_000.0, params=P)


def test_a_breakout_close_queues_a_buy_for_the_next_open() -> None:
    state = state_at(6)
    assert state.as_of_ms == _BASE_MS + 5 * _DAY_MS and state.positions == ()
    (order,) = state.orders
    assert (order.side, order.close, order.level, order.stop) == ("BUY", 11.5, 11.0, 10.2)
    assert order.risk_amount == pytest.approx(100.0)
    (text,) = alerts(state)
    assert text.startswith("[PAPER_TRADE] BUY TEST/USDT")
    assert "3-day high 11.0000" in text and "Trailing stop 10.2000" in text


def test_the_next_run_holds_the_position_with_its_trailing_stop() -> None:
    state = state_at(8)
    (held,) = state.positions
    assert held.entry_ms == _BASE_MS + 6 * _DAY_MS
    assert held.entry_price == pytest.approx(11.6 * 1.001)
    assert held.quantity == pytest.approx(100.0 / 1.3)
    assert (held.close, held.exit_below) == (12.3, 11.4)   # lows of days 6-7
    assert state.orders == () and alerts(state) == []
    assert "sells on a close below 11.4000" in status(state)


def test_a_channel_break_queues_a_trailing_stop_sell() -> None:
    state = state_at(9)
    (order,) = state.orders
    assert order.side == "SELL"
    (text,) = alerts(state)
    assert text.startswith("[PAPER_TRADE] SELL TEST/USDT (Trailing Stop Hit)")
    assert "since 2020-09-" in text or "Held" in text


def test_the_sell_is_booked_at_the_next_open() -> None:
    state = state_at(10)
    assert state.positions == () and len(state.closed) == 1
    assert state.equity == pytest.approx(10_000.0 + NET_R * 100.0)


def test_the_account_starts_flat_at_its_start_date() -> None:
    """A breakout before PAPER_START is history, not a position: starting on
    day 6, the day-5 breakout is ignored, and the first entry comes from the
    first signal on or after the start — day 6's close, 12.0 over the 11.6
    channel — filled at day 7's open."""
    state = state_at(10, start_day=6)
    (trade,) = state.closed
    assert trade["entry_ms"] == _BASE_MS + 7 * _DAY_MS
    assert trade["exit_ms"] == _BASE_MS + 9 * _DAY_MS

    before_any_signal = state_at(10, start_day=9)
    assert before_any_signal.closed == () and before_any_signal.positions == ()
    assert before_any_signal.equity == 10_000.0


def test_before_its_first_close_the_account_says_so() -> None:
    state = replay({SYM: frame(BARS[:6])}, start_ms=_BASE_MS + 30 * _DAY_MS,
                   equity=10_000.0, params=P)
    assert state.as_of_ms is None
    assert "no close to evaluate yet" in status(state)


def test_consecutive_runs_agree_on_everything_already_decided() -> None:
    """Replay is append-only: tomorrow's run rebuilds today's position exactly."""
    today, tomorrow = state_at(8), state_at(9)
    assert today.positions[0] == dataclasses.replace(tomorrow.positions[0],
                                                     close=today.positions[0].close,
                                                     exit_below=today.positions[0].exit_below)


def test_the_record_round_trips_as_json(tmp_path) -> None:
    path = tmp_path / "state" / "paper.json"
    save(state_at(9), path)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["as_of"] == pd.Timestamp(_BASE_MS + 8 * _DAY_MS, unit="ms").strftime("%Y-%m-%d")
    assert record["orders_for_next_open"][0]["side"] == "SELL"
    assert record["orders_for_next_open"][0]["initial_stop"] is None      # NaN -> null
    # The stop for the NEXT close: lows of days 7-8. (11.4 was the level the
    # day-8 close itself broke.)
    assert record["positions"][0]["exit_below"] == 11.0
    assert record["account"]["start_equity"] == 10_000.0


def test_markdown_lists_positions_and_orders() -> None:
    text = markdown(state_at(9))
    assert "| TEST/USDT |" in text
    assert "SELL TEST/USDT (Trailing Stop Hit)" in text
    assert "DRY_RUN=true" in text


# ---------------------------------------------------------------------------
# The bot's daily pass, end to end, with no network
# ---------------------------------------------------------------------------
class FakeMarket:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def timeframe_seconds(self, timeframe: str) -> int:
        return {"1h": 3600, "15m": 900, "1d": 86_400}[timeframe]

    def fetch_ohlcv_history(self, symbol: str, timeframe: str, *, bars: int) -> pd.DataFrame:
        return self.frames.get(symbol, pd.DataFrame())


class Capture:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def send_text(self, text: str) -> bool:
        self.texts.append(text)
        return True

    def send_signal(self, *args, **kwargs) -> bool:
        return True


def paper_bot(tmp_path, monkeypatch, frames, symbols):
    from scanner.bot import ScannerBot
    from scanner.config import Settings

    monkeypatch.setenv("DRY_RUN", "true")
    settings = dataclasses.replace(
        Settings.from_env(), symbols=symbols, paper_start_ms=_BASE_MS,
        paper_state_file=tmp_path / "paper_state.json",
        watchlist_file=tmp_path / "watchlist.json",
    )
    notifier = Capture()
    return ScannerBot(settings, market_data=FakeMarket(frames), notifier=notifier), notifier


def test_the_daily_pass_alerts_records_and_summarises(tmp_path, monkeypatch) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    bot, notifier = paper_bot(tmp_path, monkeypatch, {SYM: frame(BARS[:9])}, (SYM,))
    state = bot.run_paper(params=P, now_ms=_BASE_MS + 9 * _DAY_MS)

    assert notifier.texts[0].startswith("[PAPER_TRADE] SELL TEST/USDT (Trailing Stop Hit)")
    assert notifier.texts[-1].startswith("[PAPER_TRADE] V5 daily status")
    assert json.loads((tmp_path / "paper_state.json").read_text())["as_of"] is not None
    assert "V5 paper trading" in summary.read_text(encoding="utf-8")
    assert len(state.positions) == 1


def test_a_missing_symbol_fails_the_run_rather_than_rewriting_history(tmp_path, monkeypatch) -> None:
    bot, _ = paper_bot(tmp_path, monkeypatch, {SYM: frame(BARS[:9])}, (SYM, "GONE/USDT"))
    with pytest.raises(MarketDataError, match="GONE/USDT"):
        bot.run_paper(params=P, now_ms=_BASE_MS + 9 * _DAY_MS)
    assert not (tmp_path / "paper_state.json").exists()


# ---------------------------------------------------------------------------
# Telegram delivery (PAPER_TELEGRAM) — paper messages only
# ---------------------------------------------------------------------------
def test_telegram_rendering_bolds_the_headline_and_escapes_the_rest() -> None:
    from scanner.paper import telegram_html

    rendered = telegram_html("[PAPER_TRADE] SELL A/USDT (Trailing Stop Hit)\nP&L <1%")
    assert rendered == "<b>[PAPER_TRADE] SELL A/USDT (Trailing Stop Hit)</b>\nP&amp;L &lt;1%"


class Telegram(Capture):
    def __init__(self, *, accepts: bool = True, token_ok: bool = True) -> None:
        super().__init__()
        self.accepts, self.token_ok = accepts, token_ok

    def verify_credentials(self) -> bool:
        return self.token_ok

    def send_text(self, text: str) -> bool:
        super().send_text(text)
        return self.accepts


def telegram_bot(tmp_path, monkeypatch, telegram):
    from scanner.bot import ScannerBot
    from scanner.config import Settings

    monkeypatch.setenv("DRY_RUN", "true")
    settings = dataclasses.replace(
        Settings.from_env(), symbols=(SYM,), paper_start_ms=_BASE_MS, paper_telegram=True,
        paper_state_file=tmp_path / "paper_state.json", watchlist_file=tmp_path / "w.json",
    )
    console = Capture()
    bot = ScannerBot(settings, market_data=FakeMarket({SYM: frame(BARS[:9])}),
                     notifier=console, paper_notifier=telegram)
    return bot, console


def test_paper_messages_go_to_telegram_even_under_dry_run(tmp_path, monkeypatch) -> None:
    telegram = Telegram()
    bot, console = telegram_bot(tmp_path, monkeypatch, telegram)
    bot.run_paper(params=P, now_ms=_BASE_MS + 9 * _DAY_MS)
    assert telegram.texts[0].startswith("<b>[PAPER_TRADE] SELL TEST/USDT (Trailing Stop Hit)</b>")
    assert telegram.texts[-1].startswith("<b>[PAPER_TRADE] V5 daily status")
    assert console.texts == []            # nothing else was sent anywhere


def test_a_rejected_token_stops_the_run_before_anything_is_sent(tmp_path, monkeypatch) -> None:
    from scanner.paper import PaperDeliveryError

    telegram = Telegram(token_ok=False)
    bot, _ = telegram_bot(tmp_path, monkeypatch, telegram)
    with pytest.raises(PaperDeliveryError, match="TELEGRAM_BOT_TOKEN"):
        bot.run_paper(params=P, now_ms=_BASE_MS + 9 * _DAY_MS)
    assert telegram.texts == []


def test_undelivered_messages_fail_the_run_after_the_record_is_saved(tmp_path, monkeypatch) -> None:
    from scanner.paper import PaperDeliveryError

    bot, _ = telegram_bot(tmp_path, monkeypatch, Telegram(accepts=False))
    with pytest.raises(PaperDeliveryError, match="TELEGRAM_CHAT_ID"):
        bot.run_paper(params=P, now_ms=_BASE_MS + 9 * _DAY_MS)
    assert (tmp_path / "paper_state.json").exists()


def test_paper_telegram_refuses_to_start_without_both_keys(tmp_path, monkeypatch) -> None:
    from scanner.config import ConfigError, Settings

    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("PAPER_TELEGRAM", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    with pytest.raises(ConfigError, match="repository"):
        Settings.from_env(env_file=tmp_path / "absent.env")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    assert Settings.from_env(env_file=tmp_path / "absent.env").paper_telegram
