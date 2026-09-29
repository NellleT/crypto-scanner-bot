"""V4.0 evaluation: deviation trades walked forward on history, with controls.

The discipline V3.1 taught, applied from the start:

* **Entry** fills at the confirming close — a market order — paying taker.
* **Exit** is a bracket: a stop-market (taker) beyond the sweep and a resting
  limit (maker) at the target. When one candle spans both levels the stop is
  taken: the order of events inside a candle is unknowable, and assuming the
  good one is how backtests lie.
* **One position per symbol.** A deviation that confirms while a trade is still
  open on that symbol is skipped, as a single-account bot would have to.
* **Slippage is not modelled**, which flatters every result.

Two controls ask whether the *direction* call carries information, because V3.1
was profitable-looking on paper and no better than a coin in these terms:

* **flipped** — the same fill, the same stop and target distances, the opposite
  direction: a breakout trade taken at the same moment;
* **random** — the same symbol, direction, stop width and reward ratio, entered
  at uniformly random candles.

A mean-reversion edge has to beat both, after fees.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final, Mapping

import numpy as np
import pandas as pd

from scanner.deviation import Deviation, DeviationParams, plan_deviation, scan_deviations
from scanner.risk import DEFAULT_ACCOUNT_EQUITY, DEFAULT_RISK_PER_TRADE_PCT, TradePlan
from scanner.smc import Direction

logger: Final[logging.Logger] = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Fees:
    """Commission as fractions of notional."""

    maker: float
    taker: float


FUTURES_FEES: Final[Fees] = Fees(maker=0.0002, taker=0.0005)   # Binance USDT-M, VIP0
SPOT_FEES: Final[Fees] = Fees(maker=0.0010, taker=0.0010)      # Binance spot, VIP0


def first_hit(mask: np.ndarray) -> int:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else -1


def walk_bracket(
    highs: np.ndarray,
    lows: np.ndarray,
    start: int,
    *,
    long: bool,
    stop: float,
    target: float,
) -> tuple[str, int]:
    """First exit at or after candle ``start``: ``("tp" | "sl", index)``, or
    ``("open", -1)`` if neither level is reached. Stop first on a shared candle."""
    h, l = highs[start:], lows[start:]
    sl = first_hit(l <= stop) if long else first_hit(h >= stop)
    tp = first_hit(h >= target) if long else first_hit(l <= target)
    if sl < 0 and tp < 0:
        return "open", -1
    if sl >= 0 and (tp < 0 or sl <= tp):
        return "sl", start + sl
    return "tp", start + tp


@dataclass(frozen=True, slots=True)
class DeviationTrade:
    """One deviation, sized and walked to its exit."""

    symbol: str
    deviation: Deviation
    plan: TradePlan
    status: str                # tp | sl | open
    entry_index: int           # candle whose close is the fill
    exit_index: int            # candle the exit happened in, -1 while open
    exit_ms: int | None

    @property
    def direction(self) -> Direction:
        return self.plan.direction

    @property
    def risk(self) -> float:
        return self.plan.risk_per_unit

    @property
    def r(self) -> float:
        """Gross result in R."""
        return {"tp": self.plan.reward_ratio, "sl": -1.0}.get(self.status, 0.0)

    def fee_r(self, fees: Fees) -> float:
        """Round-trip commission in R: taker in, taker out on a stop, maker at the
        target — each on the notional at that price."""
        if self.status not in ("tp", "sl"):
            return 0.0
        exit_price = self.plan.take_profit if self.status == "tp" else self.plan.stop_loss
        exit_rate = fees.maker if self.status == "tp" else fees.taker
        return (fees.taker * self.plan.entry + exit_rate * exit_price) / self.risk

    def net_r(self, fees: Fees) -> float:
        return self.r - self.fee_r(fees)

    @property
    def exit_at(self) -> datetime | None:
        if self.exit_ms is None:
            return None
        return datetime.fromtimestamp(self.exit_ms / 1000, tz=timezone.utc)


@dataclass(slots=True)
class DeviationReport:
    """Every trade a backtest took, and why the rest were not taken."""

    params: DeviationParams
    symbols: tuple[str, ...]
    trades: list[DeviationTrade] = field(default_factory=list)
    deviations: int = 0
    skipped_busy: int = 0
    rejected: Counter[str] = field(default_factory=Counter)
    start: datetime | None = None
    end: datetime | None = None

    def closed(self) -> list[DeviationTrade]:
        return sorted((t for t in self.trades if t.status in ("tp", "sl")),
                      key=lambda t: t.exit_ms or 0)


def backtest(
    frames: Mapping[str, pd.DataFrame],
    params: DeviationParams | None = None,
    *,
    equity: float = DEFAULT_ACCOUNT_EQUITY,
    risk_pct: float = DEFAULT_RISK_PER_TRADE_PCT,
) -> DeviationReport:
    """Scan, size and walk every deviation in ``frames``, one symbol at a time."""
    params = params or DeviationParams()
    report = DeviationReport(params=params, symbols=tuple(frames))
    starts, ends = [], []
    for symbol, df in frames.items():
        if df.empty:
            continue
        timestamps = df["timestamp"].to_numpy(dtype="int64")
        highs = df["high"].to_numpy(dtype="float64")
        lows = df["low"].to_numpy(dtype="float64")
        starts.append(int(timestamps[0]))
        ends.append(int(timestamps[-1]))

        busy_until = -1   # candle in which the open position exits
        for deviation in scan_deviations(df, params):
            report.deviations += 1
            index = int(np.searchsorted(timestamps, deviation.confirm_ms))
            if index < busy_until:
                report.skipped_busy += 1
                continue
            plan, rejection = plan_deviation(deviation, params, equity=equity, risk_pct=risk_pct)
            if plan is None:
                assert rejection is not None
                report.rejected[rejection.stage] += 1
                continue
            status, exit_index = walk_bracket(
                highs, lows, index + 1,
                long=plan.direction is Direction.LONG,
                stop=plan.stop_loss, target=plan.take_profit,
            )
            report.trades.append(
                DeviationTrade(
                    symbol=symbol, deviation=deviation, plan=plan, status=status,
                    entry_index=index, exit_index=exit_index,
                    exit_ms=int(timestamps[exit_index]) if exit_index >= 0 else None,
                )
            )
            busy_until = exit_index if exit_index >= 0 else len(df)

    if starts:
        report.start = datetime.fromtimestamp(min(starts) / 1000, tz=timezone.utc)
        report.end = datetime.fromtimestamp(max(ends) / 1000, tz=timezone.utc)
    return report


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
def flipped_r(trade: DeviationTrade, frame: pd.DataFrame) -> float | None:
    """The same fill with the direction reversed at the same distances — the
    breakout trade at the moment V4 fades. ``None`` if it never resolves."""
    highs = frame["high"].to_numpy(dtype="float64")
    lows = frame["low"].to_numpy(dtype="float64")
    entry, risk = trade.plan.entry, trade.risk
    reward = risk * trade.plan.reward_ratio
    long = trade.direction is Direction.SHORT     # reversed
    stop = entry - risk if long else entry + risk
    target = entry + reward if long else entry - reward
    status, _ = walk_bracket(highs, lows, trade.entry_index + 1, long=long, stop=stop, target=target)
    return {"tp": trade.plan.reward_ratio, "sl": -1.0}.get(status)


def random_r(
    trade: DeviationTrade,
    frame: pd.DataFrame,
    *,
    draws: int,
    rng: np.random.Generator,
    tail_margin: int = 500,
) -> list[float]:
    """The trade's bracket — same direction, stop width and reward ratio — entered
    at random candles of the same symbol. Unresolved draws are dropped."""
    highs = frame["high"].to_numpy(dtype="float64")
    lows = frame["low"].to_numpy(dtype="float64")
    closes = frame["close"].to_numpy(dtype="float64")
    stop_frac = trade.risk / trade.plan.entry
    ratio = trade.plan.reward_ratio
    long = trade.direction is Direction.LONG
    upper = max(len(frame) - tail_margin, 2)
    out: list[float] = []
    for j in rng.integers(1, upper, size=draws):
        px = closes[j]
        stop = px * (1 - stop_frac) if long else px * (1 + stop_frac)
        target = px * (1 + stop_frac * ratio) if long else px * (1 - stop_frac * ratio)
        status, _ = walk_bracket(highs, lows, int(j) + 1, long=long, stop=stop, target=target)
        if status != "open":
            out.append(ratio if status == "tp" else -1.0)
    return out


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Summary:
    trades: int
    wins: int
    win_rate: float
    breakeven_win_rate: float    # fair-bet rate: mean of 1 / (1 + reward ratio)
    avg_ratio: float
    gross_r: float
    net_r: float
    expectancy: float            # net R per trade
    ci_low: float                # 95% CI of net R per trade, resampling months
    ci_high: float
    max_drawdown: float          # net R, peak to trough, in exit order
    losing_run: int
    account_pct: float           # compounded at the report's risk per trade


def summarize(
    trades: list[DeviationTrade],
    fees: Fees,
    *,
    risk_pct: float = DEFAULT_RISK_PER_TRADE_PCT,
    draws: int = 20_000,
    seed: int = 2026,
) -> Summary:
    """Headline statistics over closed trades, in exit order.

    The confidence interval resamples whole calendar months rather than single
    trades: trades inside a month are correlated (one market move can stop out
    four pairs in an afternoon), and treating them as independent would claim
    more certainty than the sample holds.
    """
    closed = sorted((t for t in trades if t.status in ("tp", "sl")), key=lambda t: t.exit_ms or 0)
    if not closed:
        return Summary(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
    net = np.array([t.net_r(fees) for t in closed])
    gross = np.array([t.r for t in closed])
    ratios = np.array([t.plan.reward_ratio for t in closed])
    wins = int((gross > 0).sum())

    equity = peak = drawdown = 0.0
    run = worst = 0
    for x in net:
        equity += x
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)
        run = run + 1 if x < 0 else 0
        worst = max(worst, run)

    months = pd.Series(net).groupby(
        [datetime.fromtimestamp((t.exit_ms or 0) / 1000, tz=timezone.utc).strftime("%Y-%m")
         for t in closed]
    )
    sums, counts = months.sum().to_numpy(), months.count().to_numpy()
    if len(sums) >= 3:
        rng = np.random.default_rng(seed)
        pick = rng.integers(0, len(sums), size=(draws, len(sums)))
        boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
        ci_low, ci_high = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    else:
        # Resampling one or two months only reshuffles themselves: the interval
        # would collapse to a point and read as certainty it does not have.
        ci_low = ci_high = float("nan")

    return Summary(
        trades=len(closed),
        wins=wins,
        win_rate=wins / len(closed) * 100.0,
        breakeven_win_rate=float(np.mean(1.0 / (1.0 + ratios))) * 100.0,
        avg_ratio=float(ratios.mean()),
        gross_r=float(gross.sum()),
        net_r=float(net.sum()),
        expectancy=float(net.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        max_drawdown=drawdown,
        losing_run=worst,
        account_pct=(float(np.prod(1.0 + net * risk_pct / 100.0)) - 1.0) * 100.0,
    )


def render(report: DeviationReport, *, fees: Fees = FUTURES_FEES) -> str:
    """Human-readable result of one backtest."""
    p = report.params
    s = summarize(report.trades, fees)
    span = (f"{report.start:%Y-%m-%d} to {report.end:%Y-%m-%d}"
            if report.start and report.end else "unknown span")
    ci = ("n/a — needs 3+ months of trades" if np.isnan(s.ci_low)
          else f"[{s.ci_low:+.3f}, {s.ci_high:+.3f}]")
    lines = [
        "=" * 72,
        "V4.0 DEVIATION BACKTEST (research — not live)",
        "=" * 72,
        f"Assets   : {', '.join(report.symbols)}",
        f"Period   : {span}",
        f"Model    : pools within {p.lookback} candles, pivot strength {p.strength}, "
        f"reclaim within {p.reclaim_bars}, stop {p.stop_buffer_pct:g}% beyond the sweep, "
        f"target {p.target.value}",
        f"Costs    : maker {fees.maker * 100:g}% / taker {fees.taker * 100:g}%, "
        "no slippage (flatters results)",
        "",
        f"{'Deviations confirmed':<36}{report.deviations:>8}",
        f"{'  skipped: position already open':<36}{report.skipped_busy:>8}",
    ]
    for stage, count in report.rejected.most_common():
        lines.append(f"{'  rejected: ' + stage:<36}{count:>8}")
    lines += [
        f"{'Trades closed':<36}{s.trades:>8}",
        f"{'Win rate (fair-bet rate)':<36}{s.win_rate:>7.1f}%  ({s.breakeven_win_rate:.1f}%)",
        f"{'Average reward ratio':<36}{s.avg_ratio:>8.2f}",
        f"{'Gross R / after fees':<36}{s.gross_r:>+8.1f} / {s.net_r:+.1f}",
        f"{'Net R per trade (95% CI)':<36}{s.expectancy:>+8.3f}  {ci}",
        f"{'Max drawdown / losing run':<36}{s.max_drawdown:>+8.1f}R / {s.losing_run}",
        f"{'Account at 1% risk':<36}{s.account_pct:>+7.1f}%",
        "=" * 72,
    ]
    return "\n".join(lines)


__all__ = [
    "FUTURES_FEES",
    "SPOT_FEES",
    "DeviationReport",
    "DeviationTrade",
    "Fees",
    "Summary",
    "backtest",
    "flipped_r",
    "random_r",
    "render",
    "summarize",
    "walk_bracket",
]
