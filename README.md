# Crypto Scanner Bot v3.1 — Institutional MTF SMC Engine

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

## Measured behaviour

`python main.py --simulate --history 800` over 33 days of 1h candles on all
seven pairs:

```
-- HTF funnel ----------------------------------------------------
Order blocks detected                         2468
  rejected: no FVG                            2169       (87.9%)
  rejected: FVG < 0.3%                         221        (9.0%)
  rejected: premium/discount                    36        (1.5%)
  rejected: stop wider than 3.5%                 3        (0.1%)
  rejected: not sizeable                         0        (0.0%)
Converted to watchlist                          39        (1.6%)

-- Zone lifecycle ------------------------------------------------
Tagged (price returned to zone)                 25       (64.1%)
  invalidated: TP hit before tag                10       (25.6%)
  invalidated: HTF structure break               2        (5.1%)
  invalidated: expired                           0        (0.0%)
  still open at end of data                      2        (5.1%)

-- LTF confirmation ----------------------------------------------
CONFIRMED ENTRIES                               18       (46.2%)
  unconfirmed: choch                             4
  unconfirmed: in_zone                           2
  unconfirmed: ltf_fvg                           1

-- Signal frequency ----------------------------------------------
Entries per day (all assets)                  0.54
Entries per week (all assets)                 3.79
Entries per day per asset                    0.077
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

* **~1.6% of order blocks reach the watchlist.** The pipeline is severe by
  design; most "order blocks" on a 1h chart are not tradable structures.
* **~26% of watched zones die because price hit the target before returning.**
  That is the inducement case the MTF design exists to avoid — v3.0 would have
  rested a limit order into a move that never came back.

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
| `DRY_RUN` | `false` | Log alerts instead of sending them. |

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
│   ├── strategy.py          HTF filter chain, typed rejection stages
│   ├── notifier.py          Telegram delivery + dry-run console notifier
│   ├── bot.py               MTF scan loop, scheduling, shutdown
│   └── logging_setup.py     console + rotating file handlers
└── tests/                   128 tests, no network required
    ├── test_smc.py          displacement, premium/discount, CHoCH
    ├── test_watchlist.py    lifecycle + persistence
    ├── test_mtf.py          confirmation stages
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
  ranges, and v3.1 needs both timeframes from the execution venue.
