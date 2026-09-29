# Crypto Scanner Bot v3.1.1 — Institutional MTF SMC Engine

> **Deprecated — do not trade this strategy.** A live-faithful three-year
> backtest (2023-09 → 2026-09, 7 pairs, realistic orders and fees) lost money in
> every variant tested. At 1:3 it won **20.1%** of 239 filled trades against the
> 25% it needs to break even: **−52R after futures fees, −43% of the account at
> 1% risk**. Keep `DRY_RUN=true`. The numbers are under
> [Three-year backtest](#three-year-backtest). Its successor, V4.0, is in
> research.

Finds higher-timeframe **Order Blocks** with genuine displacement at the
**extremes** of the dealing range, watches them, and only builds an order once
the **lower timeframe confirms** with a change of character.

Public market data only — **no exchange API keys are required or accepted.** The
bot builds and displays orders; it never sends them.

---

## What v3.1 changed

v3.0 alerted on every validated order block and rested a limit order there
immediately. That captured noise, inducements and traps. v3.1 adds three gates
and a lifecycle.

| | v3.0 | v3.1 |
| --- | --- | --- |
| Gap rule | Any gap > 0 | Gap ≥ **`MIN_FVG_PCT`** (0.30%) |
| Location | Anywhere | **Discount only** for longs, **premium only** for shorts |
| Entry | Blind GTC limit on detection | **Watched**, then LTF **CHoCH + FVG** |
| Invalidation | — | TP-before-tag, structure break, age |
| Reporting | — | `--simulate` funnel and frequency report |
| Data | 1h only | **1h + 15m**, fetched concurrently |

Nothing is required to upgrade a v3.0 `.env` — every new key has a default.

## What v3.1.1 changed

* **Dead-on-arrival guard.** LTF confirmation now also requires price to sit
  strictly between the stop and the target, at both the trigger candle's close
  and the newest close. Otherwise the zone is retired (`stop_breached` /
  `target_reached`) instead of alerted. Over three years, 20 of 364 live alerts
  (5.5%) would have gone out with price already past the target.
* **The replay is faithful to the live scanner.** The scanner wakes once per 1h
  close, retires dead zones *before* it looks for a trigger, and keeps a tagged
  zone until it breaks or ages out. The old replay instead searched every 15m bar
  for a fixed ten hours with no HTF checks after the tag: it confirmed 135 of 481
  three-year entries after the live scanner would already have killed the zone,
  and dated alerts up to 45 minutes before the scanner could send them. Each
  entry now carries the pass that actually sends it (`alert_utc` in the CSV).
* **Scheduled GitHub runs are dry-run.** A manual run can still opt out.

---

## The pipeline

### 1. Structure (HTF, 1h)

Read from the three most recently **closed** candles:

| Index | Role | Requirement (bullish) |
| --- | --- | --- |
| `[-3]` | **Order Block** | Bearish — last down-close before the impulse |
| `[-2]` | **Impulse** | Bullish displacement |
| `[-1]` | **Confirmation** | Its low defines the gap |

### 2. Displacement threshold

```
bullish:  ((low[-1]  - high[-3]) / high[-3]) * 100  >=  MIN_FVG_PCT
bearish:  ((low[-3]  - high[-1]) / low[-3])  * 100  >=  MIN_FVG_PCT
```

A gap that exists but spans a few ticks is spread and noise. Measured over 5,250
hourly evaluations on the 7-pair watchlist, this rejects **221 blocks (8.9%)**
that had a valid gap but no real displacement behind it.

### 3. Spatial filter — premium / discount

The dealing range is the high and low of the last `RANGE_LOOKBACK` candles;
equilibrium is the 0.5 Fibonacci level.

* **Longs** are only taken when the block sits entirely **below** equilibrium.
* **Shorts** only when it sits entirely **above** it.

"Entirely" is enforced by testing the edge nearest equilibrium — the high of a
bullish block, the low of a bearish one — so a block straddling the midpoint is
rejected rather than counted by its far edge.

### 4. Stop-width sanity

A block that survives the filters above still has to produce a stop worth
taking. `MAX_STOP_PCT` rejects setups whose stop sits more than 3.5% from entry.

This is **not** an account-risk control — fixed-fraction sizing already holds the
loss at `RISK_PER_TRADE_PCT` however wide the stop is. It is a precision
control: a zone that thick has not located anything, and the position it implies
is too small to be worth the fees. On the measured sample the one rejected entry
was a 6.5-point SOL block (96.18–102.74, a 7.03% stop) sizing down to a $1,422
notional.

### 5. Watchlist, not a blind entry

A block that survives all of the above becomes a **watched zone**, not an order:

```
PENDING ──price enters zone──▶ TAGGED ──LTF CHoCH + FVG──▶ TRIGGERED
   │                             │
   ├── take-profit reached first ┤
   ├── HTF close beyond distal ──┤
   └── max age exceeded ─────────┴──▶ INVALIDATED
```

* **TP before tag** — price ran to the 1:4 target without us. The liquidity is
  gone; the setup is discarded.
* **Structure break** — an HTF candle *closes* beyond the distal edge. Wicks do
  not invalidate; closes do.
* **Age** — retired after `MAX_ZONE_AGE_HOURS`.

### 6. LTF confirmation (15m)

A tagged zone is a *location*, not a trade. Entry requires, in order:

1. price trading inside the zone on the LTF;
2. a **Change of Character** — swing highs were descending and price closes
   above the most recent one (mirrored for shorts). Requiring the prior sequence
   to be trending is what separates a genuine turn from a continuation break;
3. an **LTF fair value gap** at or after the CHoCH, evidencing displacement out
   of the turn.

Only then is an order built.

---

## Regime filter (kill switch) — available, off by default

`REGIME_FILTER` can halt trend-following entries while the 1h market reads as
ranging, by ADX, by structural containment, or by both. It is **off by default
because three years of backtesting say it should be.**

### How it decides

* **ADX** — Wilder's ADX(14) below `ADX_THRESHOLD` (20) reads as ranging.
  Verified against `pandas-ta-classic` to within 1e-4.
* **Structure** — the last confirmed swing high and low bound a range; the
  market is ranging until a candle closes with its *whole body* beyond a bound.
  Wicks through a bound are sweeps and do not count.
* Both are causal: a pivot is only treated as known `REGIME_SWING_STRENGTH`
  bars after it prints, and a reading at time *t* uses only candles closed by
  *t*. The backtest and the live scanner apply the identical rule.
* `REGIME_GATE` chooses where it acts: at zone **admission**, at the LTF
  **entry** trigger, or **both**. An entry-gated zone is not discarded — it
  waits, and a later trigger in a trending reading can still fire.

### What the backtest found

Over three years neither variant rescues the strategy (see the variants table
under [Three-year backtest](#three-year-backtest)). Both trade less, and both lose
*more per trade* than trading unfiltered: −0.29R (ADX) and −0.30R (structure)
against −0.22R, after futures fees.

An earlier one-year study here reported +16R in-sample and credited low-ADX
setups as the strategy's best trades. Those figures came from an outcome model
with a one-bar lookahead and orders that never expired, run on the unfaithful
replay fixed in v3.1.1. They are withdrawn.

---

## Three-year backtest

2023-09-25 → 2026-09-29, all seven pairs, Binance 1h + 15m candles, the defaults
in `.env.example` at 1:3. Alerts come from the live-faithful replay; each is then
walked forward on 15m the way a real order would live:

* the order goes in at the scanner pass that sends the alert;
* a limit that is already through the market fills at once, at market (taker);
* otherwise it rests (maker), and is cancelled if the target prints first or
  after 72 hours;
* the stop is taken first when a candle spans both, and a target inside a
  resting order's own fill candle is not credited — the order of events inside
  one candle is unknowable;
* fees are Binance VIP0: futures 0.02% maker / 0.05% taker, spot 0.10%;
* slippage is not modelled, which flatters every number below.

| 1:3 | Result |
| --- | --- |
| Alerts / filled / cancelled | 344 / 239 / 105 |
| Win rate | **20.1%** (breakeven 25.0%) |
| Gross | −38.2R |
| After futures fees | **−52.3R** (−0.22R per trade, 95% CI −0.48 to +0.04) |
| After spot fees | −78.7R |
| Account at 1% risk, futures fees | **−43%** |
| Worst drawdown / longest losing run | −57.7R / 23 trades |

| Year | Fills | Win rate | Gross | After futures fees |
| --- | --- | --- | --- | --- |
| 2023-09 → 2024-09 | 68 | 13.2% | −29.6R | −33.7R |
| 2024-09 → 2025-09 | 91 | 25.3% | +4.0R | −0.9R |
| 2025-09 → 2026-09 | 80 | 20.0% | −12.5R | −17.7R |

| Variant | Fills | Win rate (needs) | Gross | Futures fees | Per trade |
| --- | --- | --- | --- | --- | --- |
| 1:2 | 166 | 27.1% (33.3%) | −23.6R | −32.8R | −0.20R |
| **1:3** | 239 | 20.1% (25.0%) | −38.2R | −52.3R | −0.22R |
| 1:4 | 316 | 15.5% (20.0%) | −60.4R | −80.3R | −0.25R |
| 1:3, ADX < 20 kill switch at entry | 128 | 18.0% (25.0%) | −30.1R | −37.7R | −0.29R |
| 1:3, structure filter at admission | 84 | 17.9% (25.0%) | −20.7R | −25.4R | −0.30R |
| 1:3, orders never cancelled | 339 | 20.4% (25.0%) | −54.2R | −74.2R | −0.22R |
| 1:3, filled only on a trade through | 238 | 19.7% (25.0%) | −41.2R | −55.3R | −0.23R |

Longs lost far more than shorts (−35.1R against −3.0R gross), and 4 of 12
quarters were positive after fees. The headline's confidence interval still
touches zero, so the sample cannot rule out a sliver of edge — but no year,
variant or rule change was positive after costs, and the best case the
statistics allow is roughly breakeven. That is not a strategy to put money on.

Figures published here before v3.1.1 (+16R over six months, −2R over a year)
are superseded: they came from the replay and outcome model fixed in v3.1.1.

---

## Measured behaviour

The replay behind the backtest above — the same report `--simulate` prints
(`--history 26400` fetches three years):

```
-- HTF funnel ----------------------------------------------------
Order blocks detected                        86624
  rejected: no FVG                           77419       (89.4%)
  rejected: FVG < 0.3%                        5795        (6.7%)
  rejected: premium/discount                  1738        (2.0%)
  rejected: ranging market                       0        (0.0%)
  rejected: stop wider than 3.5%                43        (0.0%)
  rejected: not sizeable                         0        (0.0%)
Converted to watchlist                        1629        (1.9%)

-- Zone lifecycle ------------------------------------------------
Tagged (price returned to zone)                776       (47.6%)
  invalidated: TP hit before tag               770       (47.3%)
  invalidated: HTF structure break              74        (4.5%)
  invalidated: expired                           8        (0.5%)
  still open at end of data                      1        (0.1%)

-- LTF confirmation ----------------------------------------------
CONFIRMED ENTRIES                              344       (21.1%)
  died waiting: HTF structure break            370       (22.7%)
  died waiting: expired                         42        (2.6%)
  dead on arrival: through the stop              0        (0.0%)
  dead on arrival: past the target              20        (1.2%)
  still waiting at end of data                   0        (0.0%)

-- Signal frequency ----------------------------------------------
Entries per day (all assets)                  0.31
Entries per week (all assets)                 2.19
Entries per day per asset                    0.045
```

Every confirmed entry can be listed for manual verification:

```bash
python main.py --simulate --entries full   # per-entry breakdown (default)
python main.py --simulate --entries table  # compact grid
python main.py --simulate --entries csv    # for a spreadsheet
```

Each entry reports four timestamps — block formed, tagged, CHoCH, LTF gap — at
the **open** of the candle concerned, so any signal can be located on a chart
and checked by hand.

Two numbers worth dwelling on:

* **~1.9% of order blocks reach the watchlist.** The pipeline is severe by
  design; most "order blocks" on a 1h chart are not tradable structures.
* **~47% of watched zones die because price hit the target before returning,**
  and nearly half of the zones that are tagged then break structure before the
  lower timeframe turns. Being selective did not make the survivors profitable.

Counts drift by a candle or two between runs as the newest bar closes; the
proportions are stable.

Every live pass logs the same funnel:

```
Filter funnel: order_block=2, fvg=5, zone_added=1, confirmed=0
Watchlist: 3 pending, 1 tagged, 0 triggered, 12 invalidated
```

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

cp .env.example .env            # add Telegram credentials + ACCOUNT_EQUITY

python main.py --simulate                    # replay history, print the funnel
python main.py --once --dry-run              # one live pass, sends nothing
python main.py --once --dry-run --log-level DEBUG   # per-symbol rejections
python main.py                               # run continuously
```

`--simulate` implies `--dry-run` and needs no credentials. Use `--history N` to
set the HTF depth; the LTF frame is paged automatically to cover the same span.

---

## Configuration reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | — | Required unless `DRY_RUN=true`. |
| `SYMBOLS` | 7 USDT pairs | Accepts `BTCUSDT` **or** `BTC/USDT`. |
| `TIMEFRAME` | `1h` | Higher timeframe — structure. |
| `LTF_TIMEFRAME` | `15m` | Lower timeframe — entries. Must be faster. |
| `EXCHANGE_ID` | `binance` | Must match the execution venue. |
| `CANDLE_LIMIT` | `200` | HTF candles; must exceed `RANGE_LOOKBACK`. |
| `LTF_CANDLE_LIMIT` | `120` | LTF candles per live pass. |
| `MIN_FVG_PCT` | `0.30` | Displacement threshold, in **percent**. |
| `RANGE_LOOKBACK` | `50` | Candles forming the dealing range. |
| `REQUIRE_EXTREME_OB` | `true` | Enforce premium/discount. |
| `MIN_BODY_RATIO` | `0.05` | Doji guard on block and impulse. |
| `SWING_STRENGTH` | `2` | Bars each side to confirm a pivot. |
| `LTF_CONFIRM_WINDOW` | `30` | LTF candles searched for CHoCH + FVG. |
| `LTF_MIN_FVG_PCT` | `0.0` | Minimum LTF gap, in percent. |
| `MAX_ZONE_AGE_HOURS` | `72` | Age at which a zone is retired. |
| `WATCHLIST_FILE` | `watchlist.json` | Persisted zone state. |
| `MAX_WORKERS` | `4` | Concurrent market-data fetches. |
| `STOP_BUFFER_PCT` | `0.2` | Offset beyond the distal edge, in percent. |
| `MAX_STOP_PCT` | `3.5` | Widest tradable stop, in percent. `0.0` disables. |
| `REWARD_RATIO` | `4` | Take-profit R-multiple. |
| `ACCOUNT_EQUITY` | `10000` | **Stale values mis-size every order.** |
| `RISK_PER_TRADE_PCT` | `1` | Percent of equity risked per trade. |
| `REGIME_FILTER` | `off` | Kill switch: `off`, `adx`, `structure`, `confluence`. |
| `REGIME_GATE` | `entry` | Where it acts: `admission`, `entry`, `both`. |
| `ADX_PERIOD` / `ADX_THRESHOLD` | `14` / `20` | ADX regime settings. |
| `REGIME_SWING_STRENGTH` | `5` | Pivot strength for structural containment. |
| `REGIME_LOOKBACK` | `200` | HTF candles the regime is read over (≤ `CANDLE_LIMIT`). |
| `DRY_RUN` | `false` | Log alerts instead of sending them. **Keep `true`: v3.1 is deprecated.** |

---

## Project layout

```
Trading Bot/
├── main.py                  entrypoint, CLI, --simulate
├── scanner/
│   ├── config.py            env loading + validation (fails fast)
│   ├── exchange.py          CCXT wrapper, retries, concurrency, paged history
│   ├── candles.py           bar geometry — no I/O
│   ├── smc.py               order blocks, FVG, premium/discount, CHoCH — no I/O
│   ├── risk.py              entry/stop/target + position sizing — no I/O
│   ├── watchlist.py         zone lifecycle + persistence
│   ├── mtf.py               lower-timeframe confirmation — no I/O
│   ├── execution.py         Binance order payloads — no I/O
│   ├── analytics.py         historical replay + funnel report
│   ├── regime.py            ADX + structural regime, kill switch — no I/O
│   ├── strategy.py          HTF filter chain, typed rejection stages
│   ├── notifier.py          Telegram delivery + dry-run console notifier
│   ├── bot.py               MTF scan loop, scheduling, shutdown
│   └── logging_setup.py     console + rotating file handlers
└── tests/                   182 tests, no network required
    ├── test_smc.py          displacement, premium/discount, CHoCH
    ├── test_watchlist.py    lifecycle + persistence
    ├── test_mtf.py          confirmation stages
    ├── test_regime.py       ADX correctness, causality, both gates
    ├── test_replay.py       replay fidelity to the live scanner's passes
    ├── test_risk.py  test_strategy.py  test_execution.py  test_notifier.py
```

Everything except `exchange`, `notifier` and `bot` is I/O-free:

```bash
python -m pytest tests -q
```

---

## Implementation notes

**Vectorised maths.** Gaps are two shifted subtractions (`fvg_frame`); the
dealing range and the premium/discount array are rolling extremes
(`premium_discount_frame`); swing pivots are a centred rolling extreme
(`swing_points`). `order_block_mask` classifies a whole frame at once and is
tested for exact agreement with the scalar detector.

**No look-ahead by construction.** A centred rolling window leaves the newest
`strength` bars NaN, so an unconfirmed pivot can never be read as structure. The
historical replay evaluates each bar against only the candles up to itself, and
searches for LTF confirmation only in candles that closed *after* the tag.

**Concurrency.** MTF doubles the request count and each request is almost
entirely network wait, so fetches run in a thread pool — measured 3.4× faster
for 14 frames. Each worker gets its **own** CCXT instance: the sync client keeps
a `requests.Session` and a rate-limit clock on the instance, neither of which is
thread-safe. The venue therefore sees up to `MAX_WORKERS` times the request
rate, so `REQUEST_DELAY_SECONDS` still matters.

**Paged history.** Venues cap one OHLCV response (1000 on Binance). A backtest
needs the same wall-clock span on both timeframes, and 15m needs four times as
many candles as 1h to cover it. `fetch_ohlcv_history` pages with `since`.
Without it the LTF frame silently covers a fraction of the HTF period and every
older zone looks unconfirmable — which is exactly what the first simulation run
showed before it was fixed.

**Persistence.** The zone lifecycle spans many candles, but a scheduled run is a
fresh process. Held only in memory, no zone could ever reach TRIGGERED. The
watchlist is written atomically after each pass and reloaded on start; a corrupt
or future-schema file starts empty rather than crashing.

---

## Notes and limitations

- **It places no orders.** It builds payloads and displays them.
- **`ACCOUNT_EQUITY` is static config**, not your live balance.
- **The watchlist file is state.** Delete it and every in-flight zone is lost;
  point two bots at the same file and they will fight over it.
- **Entries are still pending orders.** Confirmation says the LTF turned, not
  that the trade will work. There is no fill tracking or post-entry management.
- **One structure per symbol per pass.** Only the newest three closed HTF
  candles are examined; older unmitigated blocks are not rediscovered.
- **CHoCH is a simplification.** It uses a two-pivot lower-high / higher-low
  test, not a full market-structure model with BOS/liquidity labelling.
- **GitHub Actions cannot run this properly** — Binance restricts its runner IP
  ranges, and v3.1 needs both timeframes from the execution venue. Scheduled
  runs are dry-run.
