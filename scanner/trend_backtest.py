"""V5.0 evaluation: trade-level results and a spot portfolio, net of costs.

* **Fills.** Entries and exits are market orders at the next day's open, paying
  the taker fee and slippage on each side, slippage always against us.
* **Trades.** Every signal on every symbol, one position per symbol at a time,
  measured in R against the planned risk (signal close minus initial stop). The
  exit is a close through the channel filled at the next open, so a gap can
  lose more than 1R — reported as it happens, not capped.
* **Portfolio.** One spot account: 1% risk on mark-to-market equity, and no
  leverage — an entry that needs more cash than the account holds is shrunk to
  the cash available, and counted.
* **Delistings.** A symbol whose data ends is sold at its last close.

Trades store raw open prices; costs are applied on read, so one simulation can
be priced under several cost assumptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final, Mapping

import numpy as np
import pandas as pd

from scanner.trend import TrendParams, trend_frame

logger: Final[logging.Logger] = logging.getLogger(__name__)

_DAY_MS: Final[int] = 86_400_000


@dataclass(frozen=True, slots=True)
class Costs:
    """Per-side trading costs, as fractions."""

    fee: float
    slippage: float


#: Binance spot taker fee, plus a conservative slippage allowance for a
#: market order at the daily open.
SPOT_COSTS: Final[Costs] = Costs(fee=0.001, slippage=0.001)


@dataclass(frozen=True, slots=True)
class TrendTrade:
    """One round trip, with raw prices; costs are applied by :meth:`r`."""

    symbol: str
    signal_ms: int        # open of the day whose close signalled the entry
    entry_ms: int         # open of the fill day
    entry_open: float     # raw open the entry filled at
    reference: float      # signal close the position was sized on
    initial_stop: float
    exit_ms: int          # open of the exit fill day (or the last day held)
    exit_price: float     # raw open (or last close for delisted / still open)
    exit_kind: str        # channel | delisted | open

    @property
    def planned_risk(self) -> float:
        return self.reference - self.initial_stop

    @property
    def days_held(self) -> int:
        return int((self.exit_ms - self.entry_ms) // _DAY_MS)

    @property
    def gross_r(self) -> float:
        return (self.exit_price - self.entry_open) / self.planned_risk

    def r(self, costs: Costs) -> float:
        """Net R: slippage against us on both fills, fee on both notionals."""
        buy = self.entry_open * (1 + costs.slippage)
        sell = self.exit_price * (1 - costs.slippage)
        return (sell - buy - costs.fee * (buy + sell)) / self.planned_risk

    @property
    def entry_at(self) -> datetime:
        return datetime.fromtimestamp(self.entry_ms / 1000, tz=timezone.utc)


def symbol_trades(
    symbol: str,
    df: pd.DataFrame,
    params: TrendParams | None = None,
    *,
    data_end_ms: int | None = None,
) -> list[TrendTrade]:
    """Every trade the rules take on one symbol, one position at a time.

    ``data_end_ms`` is the last day of the whole dataset: a position still open
    when *this* symbol's data stops earlier was delisted, and is sold at the
    last close; one open at the very end is marked there and flagged ``open``.
    """
    params = params or TrendParams()
    if df.empty:
        return []
    f = trend_frame(df, params)
    ts = f["timestamp"].to_numpy(dtype="int64")
    opens, closes = f["open"].to_numpy(float), f["close"].to_numpy(float)
    entry_sig, exit_sig = f["entry_signal"].to_numpy(bool), f["exit_signal"].to_numpy(bool)
    stops = f["next_stop"].to_numpy(float)
    last_day = int(ts[-1]) if data_end_ms is None else data_end_ms

    trades: list[TrendTrade] = []
    held: tuple[int, int] | None = None   # (signal index, entry index)
    for t in range(len(f)):
        if held is None:
            if entry_sig[t] and t + 1 < len(f) and closes[t] > stops[t]:
                held = (t, t + 1)
            continue
        signal, entry = held
        if t >= entry and exit_sig[t] and t + 1 < len(f):
            trades.append(TrendTrade(symbol, int(ts[signal]), int(ts[entry]), float(opens[entry]),
                                     float(closes[signal]), float(stops[signal]),
                                     int(ts[t + 1]), float(opens[t + 1]), "channel"))
            held = None
    if held is not None:
        signal, entry = held
        kind = "delisted" if int(ts[-1]) < last_day else "open"
        trades.append(TrendTrade(symbol, int(ts[signal]), int(ts[entry]), float(opens[entry]),
                                 float(closes[signal]), float(stops[signal]),
                                 int(ts[-1]), float(closes[-1]), kind))
    return trades


def all_trades(
    frames: Mapping[str, pd.DataFrame], params: TrendParams | None = None
) -> list[TrendTrade]:
    end = max((int(df["timestamp"].iloc[-1]) for df in frames.values() if len(df)), default=None)
    out: list[TrendTrade] = []
    for symbol, df in frames.items():
        out.extend(symbol_trades(symbol, df, params, data_end_ms=end))
    return sorted(out, key=lambda t: t.entry_ms)


def random_entries(
    symbol: str,
    df: pd.DataFrame,
    params: TrendParams,
    *,
    count: int,
    rng: np.random.Generator,
) -> list[TrendTrade]:
    """Control: enter on random days *inside the regime* (close above the SMA),
    with the same initial stop and the same channel exit. If these do as well
    as the breakout entries, the edge is the regime filter and the trailing
    exit, not the breakout."""
    f = trend_frame(df, params)
    ts = f["timestamp"].to_numpy(dtype="int64")
    opens, closes = f["open"].to_numpy(float), f["close"].to_numpy(float)
    exit_sig, stops = f["exit_signal"].to_numpy(bool), f["next_stop"].to_numpy(float)
    eligible = np.flatnonzero(f["regime"].to_numpy(bool) & (closes > stops))
    eligible = eligible[eligible + 1 < len(f)]
    if not len(eligible) or count <= 0:
        return []
    out = []
    for t in rng.choice(eligible, size=count):
        t = int(t)
        exits = np.flatnonzero(exit_sig[t + 1:])
        if len(exits) and t + 1 + exits[0] + 1 < len(f):
            u = t + 1 + int(exits[0])
            exit_ms, exit_price, kind = int(ts[u + 1]), float(opens[u + 1]), "channel"
        else:   # still open at the end: marked at the last close, as real trades are
            exit_ms, exit_price, kind = int(ts[-1]), float(closes[-1]), "open"
        out.append(TrendTrade(symbol, int(ts[t]), int(ts[t + 1]), float(opens[t + 1]),
                              float(closes[t]), float(stops[t]), exit_ms, exit_price, kind))
    return out


# ---------------------------------------------------------------------------
# Trade statistics
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TradeStats:
    trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    worst: float
    expectancy: float          # net R per trade
    total: float               # net R
    gross_total: float
    profit_factor: float
    ci_low: float              # 95% CI of net R per trade, resampling entry months
    ci_high: float
    top5_share: float          # % of total net R from the five best trades
    median_days: float


def trade_stats(
    trades: list[TrendTrade], costs: Costs, *, draws: int = 20_000, seed: int = 2026
) -> TradeStats:
    """Headline trade statistics.

    The interval resamples whole *entry months*: trend entries cluster — one
    breakout month can open five correlated longs — so treating trades as
    independent would claim more certainty than the sample holds. With fewer
    than three months the interval is not claimed at all.
    """
    if not trades:
        return TradeStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan"),
                          float("nan"), 0.0, 0.0)
    net = np.array([t.r(costs) for t in trades])
    wins, losses = net[net > 0], net[net <= 0]
    months = pd.Series(net).groupby([t.entry_at.strftime("%Y-%m") for t in trades])
    sums, counts = months.sum().to_numpy(), months.count().to_numpy()
    if len(sums) >= 3:
        rng = np.random.default_rng(seed)
        pick = rng.integers(0, len(sums), size=(draws, len(sums)))
        boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    else:
        ci = (float("nan"), float("nan"))
    top5 = np.sort(net)[-5:].sum()
    return TradeStats(
        trades=len(net),
        win_rate=len(wins) / len(net) * 100.0,
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        worst=float(net.min()),
        expectancy=float(net.mean()),
        total=float(net.sum()),
        gross_total=float(sum(t.gross_r for t in trades)),
        profit_factor=float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        ci_low=ci[0],
        ci_high=ci[1],
        top5_share=float(top5 / net.sum() * 100.0) if net.sum() > 0 else float("nan"),
        median_days=float(np.median([t.days_held for t in trades])),
    )


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Holding:
    """An open spot position."""

    symbol: str
    quantity: float
    signal_ms: int        # open of the day whose close signalled the entry
    entry_ms: int         # open of the fill day
    entry_price: float    # fill price, slippage included
    cost: float           # cash spent, fee included
    initial_stop: float


@dataclass(frozen=True, slots=True)
class Order:
    """An order queued at a daily close, to fill at the next open."""

    symbol: str
    side: str             # BUY | SELL
    signal_ms: int        # open of the day whose close decided it
    close: float          # that close
    level: float          # the channel it broke
    stop: float           # BUY: initial trailing stop; SELL: NaN
    risk_amount: float    # BUY: equity at the close x risk; SELL: 0


class TrendPortfolio:
    """One spot account run through the V5 rules, one daily candle at a time.

    The backtest and the paper trader drive this same object, so a paper record
    is the backtest continued forward rather than a re-implementation of it.

    Each :meth:`step`: fill the queued orders at the open — exits first, freeing
    cash, then entries in symbol order, each shrunk to the cash available (no
    leverage) — sell any holding whose data has ended (delisted) at its last
    close, mark to market at the close, then queue the close's signals.
    """

    def __init__(
        self,
        frames: Mapping[str, pd.DataFrame],
        params: TrendParams | None = None,
        costs: Costs = SPOT_COSTS,
        *,
        equity: float = 10_000.0,
        risk_pct: float = 1.0,
        min_notional: float = 10.0,
    ) -> None:
        self.params = params or TrendParams()
        self.costs = costs
        self.risk_pct = risk_pct
        self.min_notional = min_notional
        self.data = {
            s: trend_frame(df, self.params).set_index("timestamp")
            for s, df in frames.items() if len(df)
        }
        self.cash = equity
        self.equity = equity
        self.invested = 0.0
        self.holdings: dict[str, Holding] = {}
        self.orders: dict[str, Order] = {}     # queued at the last close
        self.fills: list[dict] = []
        self.capped = 0
        self.skipped = 0
        self.last_day: int | None = None

    def days(self, start_ms: int | None = None) -> list[int]:
        """Every daily candle open in the data, from ``start_ms`` on."""
        days = sorted(set().union(*(d.index for d in self.data.values())))
        return days if start_ms is None else [d for d in days if d >= start_ms]

    def step(self, day: int) -> None:
        self._fill(day)
        self._sell_delisted(day)
        self._mark(day)
        self._queue(day)
        self.last_day = day

    def _sell(self, symbol: str, raw: float, day: int, kind: str) -> None:
        pos = self.holdings.pop(symbol)
        price = raw * (1 - self.costs.slippage)
        proceeds = pos.quantity * price * (1 - self.costs.fee)
        self.cash += proceeds
        self.fills.append({
            "symbol": symbol, "entry_ms": pos.entry_ms, "exit_ms": day,
            "quantity": pos.quantity, "entry_price": pos.entry_price, "exit_price": price,
            "cost": pos.cost, "proceeds": proceeds, "kind": kind,
        })

    def _fill(self, day: int) -> None:
        queued, self.orders = self.orders, {}
        for symbol in sorted(s for s, o in queued.items() if o.side == "SELL"):
            if symbol in self.holdings and day in self.data[symbol].index:
                self._sell(symbol, float(self.data[symbol].at[day, "open"]), day, "channel")
        for symbol in sorted(s for s, o in queued.items() if o.side == "BUY"):
            order = queued[symbol]
            if day not in self.data[symbol].index or symbol in self.holdings:
                continue
            price = float(self.data[symbol].at[day, "open"]) * (1 + self.costs.slippage)
            quantity = order.risk_amount / (order.close - order.stop)
            if quantity * price * (1 + self.costs.fee) > self.cash:
                quantity = self.cash / (price * (1 + self.costs.fee))
                self.capped += 1
            if quantity * price < self.min_notional:
                self.skipped += 1
                continue
            cost = quantity * price * (1 + self.costs.fee)
            self.cash -= cost
            self.holdings[symbol] = Holding(symbol, quantity, order.signal_ms, day,
                                            price, cost, order.stop)

    def _sell_delisted(self, day: int) -> None:
        for symbol in [s for s in self.holdings
                       if day not in self.data[s].index and day > self.data[s].index[-1]]:
            self._sell(symbol, float(self.data[symbol]["close"].iloc[-1]), day, "delisted")

    def _mark(self, day: int) -> None:
        value = sum(pos.quantity * float(self.data[s].at[day, "close"])
                    for s, pos in self.holdings.items() if day in self.data[s].index)
        self.equity = self.cash + value
        self.invested = value / self.equity if self.equity > 0 else 0.0

    def _queue(self, day: int) -> None:
        for symbol, frame in self.data.items():
            if day not in frame.index:
                continue
            row = frame.loc[day]
            if symbol in self.holdings:
                if bool(row["exit_signal"]):
                    self.orders[symbol] = Order(symbol, "SELL", day, float(row["close"]),
                                                float(row["exit_level"]), float("nan"), 0.0)
            elif bool(row["entry_signal"]) and float(row["close"]) > float(row["next_stop"]):
                self.orders[symbol] = Order(
                    symbol, "BUY", day, float(row["close"]), float(row["entry_level"]),
                    float(row["next_stop"]), self.equity * self.risk_pct / 100.0,
                )


@dataclass(slots=True)
class PortfolioResult:
    equity: pd.Series                 # mark-to-market equity at each daily close
    exposure: pd.Series               # invested fraction of equity at each close
    fills: list[dict] = field(default_factory=list)
    capped: int = 0                   # entries shrunk to the cash available
    skipped: int = 0                  # entries too small to place


def simulate_portfolio(
    frames: Mapping[str, pd.DataFrame],
    params: TrendParams | None = None,
    costs: Costs = SPOT_COSTS,
    *,
    equity: float = 10_000.0,
    risk_pct: float = 1.0,
    start_ms: int | None = None,
    min_notional: float = 10.0,
) -> PortfolioResult:
    """Run the rules as one spot account over history (see :class:`TrendPortfolio`)."""
    book = TrendPortfolio(frames, params, costs, equity=equity, risk_pct=risk_pct,
                          min_notional=min_notional)
    days = book.days(start_ms)
    curve, invested = [], []
    for day in days:
        book.step(day)
        curve.append(book.equity)
        invested.append(book.invested)
    index = pd.to_datetime(days, unit="ms", utc=True)
    return PortfolioResult(pd.Series(curve, index=index), pd.Series(invested, index=index),
                           book.fills, book.capped, book.skipped)


def buy_and_hold(
    df: pd.DataFrame, costs: Costs = SPOT_COSTS, *, equity: float = 10_000.0,
    start_ms: int | None = None,
) -> pd.Series:
    """All-in at the first open, held to the end — the benchmark to beat."""
    frame = df if start_ms is None else df[df["timestamp"] >= start_ms]
    price = float(frame["open"].iloc[0]) * (1 + costs.slippage)
    qty = equity / (price * (1 + costs.fee))
    return pd.Series(qty * frame["close"].to_numpy(float),
                     index=pd.to_datetime(frame["timestamp"], unit="ms", utc=True))


@dataclass(frozen=True, slots=True)
class CurveStats:
    start: datetime
    end: datetime
    final: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    longest_underwater_days: int


def curve_stats(equity: pd.Series) -> CurveStats:
    """Return, drawdown and risk-adjusted return of a daily equity curve."""
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    returns = equity.pct_change().dropna()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    underwater, longest = 0, 0
    for below in (equity < peak).to_numpy():
        underwater = underwater + 1 if below else 0
        longest = max(longest, underwater)
    return CurveStats(
        start=equity.index[0].to_pydatetime(),
        end=equity.index[-1].to_pydatetime(),
        final=float(equity.iloc[-1]),
        total_return_pct=float(equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0,
        cagr_pct=float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0) * 100.0,
        max_drawdown_pct=float(drawdown.min()) * 100.0,
        sharpe=float(returns.mean() / returns.std() * np.sqrt(365)) if returns.std() > 0 else 0.0,
        longest_underwater_days=longest,
    )


def render(
    trades: list[TrendTrade],
    portfolio: PortfolioResult,
    benchmark: pd.Series,
    *,
    params: TrendParams,
    costs: Costs = SPOT_COSTS,
    benchmark_name: str = "BTC",
) -> str:
    """Human-readable result of one V5 backtest."""
    ts = trade_stats(trades, costs)
    ps, bs = curve_stats(portfolio.equity), curve_stats(benchmark)
    ci = ("n/a" if np.isnan(ts.ci_low) else f"[{ts.ci_low:+.2f}, {ts.ci_high:+.2f}]")
    lines = [
        "=" * 72,
        "V5.0 DAILY TREND-FOLLOWING BACKTEST (research — not live)",
        "=" * 72,
        f"Rules    : close > SMA{params.sma}, close > prior {params.entry_channel}-day high "
        f"-> buy next open; close < prior {params.exit_channel}-day low -> sell next open",
        f"Costs    : fee {costs.fee * 100:g}% + slippage {costs.slippage * 100:g}% per side",
        f"Period   : {ps.start:%Y-%m-%d} to {ps.end:%Y-%m-%d}",
        "",
        f"{'Trades (win rate)':<34}{ts.trades:>8}  ({ts.win_rate:.1f}%)",
        f"{'Average win / loss':<34}{ts.avg_win:>+8.2f}R / {ts.avg_loss:+.2f}R",
        f"{'Net R (gross)':<34}{ts.total:>+8.1f}  ({ts.gross_total:+.1f})",
        f"{'Net R per trade, 95% CI':<34}{ts.expectancy:>+8.3f}  {ci}",
        f"{'Profit factor':<34}{ts.profit_factor:>8.2f}",
        "",
        f"{'Portfolio vs ' + benchmark_name + ' buy-and-hold':<34}{'V5':>10}{benchmark_name:>12}",
        f"{'  final equity':<34}{ps.final:>10,.0f}{bs.final:>12,.0f}",
        f"{'  CAGR':<34}{ps.cagr_pct:>9.1f}%{bs.cagr_pct:>11.1f}%",
        f"{'  max drawdown':<34}{ps.max_drawdown_pct:>9.1f}%{bs.max_drawdown_pct:>11.1f}%",
        f"{'  Sharpe':<34}{ps.sharpe:>10.2f}{bs.sharpe:>12.2f}",
        f"{'  longest time under water':<34}{ps.longest_underwater_days:>9}d"
        f"{bs.longest_underwater_days:>11}d",
        f"{'  time invested (avg exposure)':<34}"
        f"{(portfolio.exposure > 0).mean() * 100:>9.0f}%  ({portfolio.exposure.mean() * 100:.0f}%)",
        f"{'  entries shrunk to cash / skipped':<34}{portfolio.capped:>10} / {portfolio.skipped}",
        "=" * 72,
    ]
    return "\n".join(lines)


__all__ = [
    "SPOT_COSTS",
    "Costs",
    "CurveStats",
    "Holding",
    "Order",
    "PortfolioResult",
    "TradeStats",
    "TrendTrade",
    "all_trades",
    "buy_and_hold",
    "curve_stats",
    "random_entries",
    "render",
    "simulate_portfolio",
    "symbol_trades",
    "trade_stats",
    "TrendPortfolio",
]
