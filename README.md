# Crypto Scanner Bot v3.0 — Smart Money Concepts

Scans crypto pairs for **Order Blocks validated by Fair Value Gaps**, and emits
a fully sized pending limit order ready to route to a Binance execution engine.

Public market data only — **no exchange API keys are required or accepted.** The
bot builds and displays orders; it never sends them.

---

## What v3.0 changed

The v2.x retail chain (engulfing + SMA-200 + volume + VSA) is **retired**. The
`indicators` and `patterns` modules are gone; `smc.py` and `execution.py` are
new.

| | v2.2 | v3.0 |
| --- | --- | --- |
| Signal | Engulfing + SMA + volume | Order Block + FVG |
| Entry | Market, at candle close | **Resting limit** at block's proximal edge |
| Stop | 10-bar structural low ±0.1% | Block's distal edge ±0.2% |
| Target | 1:2 and 1:3 | Fixed **1:4** |
| Sizing | — | **1% of account equity** |
| Output | Alert only | Alert **+ Binance order payload** |
| Timeframe | 4h | **1h** |
| Watchlist | 3 pairs | **7 pairs** |

Config keys `SMA_PERIOD`, `VOLUME_SMA_PERIOD`, `REQUIRE_VOLUME_EXPANSION`,
`STRUCTURAL_LOOKBACK` and `RR_TARGETS` no longer do anything and can be deleted.

---

## The setup

Read from the three most recently **closed** candles:

| Index | Role | Requirement (bullish) |
| --- | --- | --- |
| `[-3]` | **Order Block** | Bearish candle — last down-close before the impulse |
| `[-2]` | **Impulse** | Bullish displacement |
| `[-1]` | **Confirmation** | Its low defines the gap |

### The FVG is the validation

An order block is only tradable if the displacement left an unfilled
inefficiency behind it:

```
bullish:  low[-1]  > high[-3]
bearish:  high[-1] < low[-3]
```

Both **strict**. Equality means price traded through the entire range leaving
nothing behind — no gap, no displacement, no trade. Measured over 3,479 hourly
windows on the 7-pair watchlist:

| Stage | Count | Share |
| --- | --- | --- |
| Structural order blocks | 1,617 | — |
| **Rejected: no FVG** | **1,385** | **86% of blocks** |
| Confirmed signals | 232 | 14% of blocks |

The gap rule is doing the overwhelming majority of the filtering. A structurally
perfect order block with no gap is rejected outright.

### Gap detection is vectorised

`fvg_frame()` computes both gap series for a whole frame with two shifted
subtractions, so a 500-candle frame costs one pass rather than 500 Python-level
comparisons:

```python
pre_high = df["high"].shift(2)
pre_low  = df["low"].shift(2)
df["fvg_bull_gap"] = df["low"]  - pre_high   # > 0 → bullish FVG
df["fvg_bear_gap"] = pre_low    - df["high"] # > 0 → bearish FVG
```

`order_block_mask()` builds on it to classify an entire history at once. It is
tested for exact agreement with the scalar detector — **3,479 live windows, zero
mismatches** — so the fast path and the live path can never diverge silently.

---

## Order construction

```
Entry  = block proximal edge   (high for a long, low for a short)
Stop   = block distal edge, pushed 0.2% AWAY from the zone
Risk   = |entry − stop|
Target = entry ± 4 × Risk
Size   = (equity × 1%) / Risk
```

**On the stop buffer.** The spec reads "distal + buffer". Implemented as *away
from the zone* — below the low for a long, above the high for a short. Adding
0.2% to a long's stop would move it up toward the entry, closer to the sweep it
is meant to survive, which inverts the intent.

**On sizing and leverage.** Size depends only on the entry-to-stop distance, so
being stopped out always costs exactly `RISK_PER_TRADE_PCT` of equity. Leverage
is deliberately ignored — but that means a tight stop implies a large notional.
Across the 232 measured signals the median stop was **0.60%** wide (min 0.22%,
max 7.03%), giving a median notional of **1.66× equity** and a maximum of
**4.49×**. Those are not fundable on spot. `leverage_required` is reported on
every plan rather than clamped, because whether it can be funded is an account
question, not a strategy one.

### Sample alert

```
🟢 LONG · ORDER BLOCK · AVAX/USDT

Pair: AVAX/USDT
Timeframe: 1h
Setup: LONG order block, FVG validated

OB zone: $7.428 – $7.504
Fair Value Gap: 0.59% of entry  (displacement confirmed)

📋 Pending Limit Order
• Entry: $7.504  (limit, at proximal edge)
• Stop-Loss: $7.413  (distal ∓0.2%, risk 1.21%)
• Take-Profit (1:4): $7.867  (+4.84%)

🛡 Position Sizing
• Quantity: 1,100.64
• Risk: 1% of $10,000.00 = $100.00
• Reward at target: $400.00
• Notional: $8,259.22  (0.83x equity)

Route: BUY 1100.64 AVAXUSDT @ 7.504
```

### Execution payload

`build_execution_order()` renders Binance REST parameters. Prices and quantity
are **strings at venue tick/lot precision** — float repr is exactly how an
over-precise value slips past `PRICE_FILTER` or `LOT_SIZE`.

```json
{
  "symbol": "AVAXUSDT",
  "side": "BUY",
  "entry_order": {
    "symbol": "AVAXUSDT", "side": "BUY", "type": "LIMIT",
    "timeInForce": "GTC", "quantity": "1100.64", "price": "7.504"
  },
  "exit_oco": {
    "symbol": "AVAXUSDT", "side": "SELL", "quantity": "1100.64",
    "aboveType": "LIMIT_MAKER", "abovePrice": "7.867",
    "belowType": "STOP_LOSS_LIMIT", "belowPrice": "7.413",
    "belowStopPrice": "7.413", "belowTimeInForce": "GTC"
  },
  "risk": { "reward_ratio": 4.0, "risk_pct_of_equity": 1.0, "risk_amount": 100.0 }
}
```

The OCO is **not** submitted with the entry — it must go in after the entry
fills, or it would bracket a position that does not exist. Both payloads carry
the same quantity, so no residual position can be left behind.

---

## Exchange choice is now load-bearing

v2.x emitted alerts you would eyeball on a chart, so a venue mismatch was
tolerable. **v3.0 emits limit orders at exact price levels, so the levels must
come from the book they will rest in.** Kraken and Binance disagree by a median
0.02–0.18%, but up to **1.32% on AVAX** and 0.70% on BNB — against a median stop
of 0.60%, a level from the wrong venue either fills instantly or never fills.

`EXCHANGE_ID` defaults to `binance` for that reason.

> **This means GitHub Actions cannot run v3.0 properly.** Binance restricts
> GitHub's runner IP ranges. The workflow still runs with `EXCHANGE_ID: kraken`
> so the schedule stays alive, but treat that output as indicative only. For
> routable levels, run on a host Binance serves — a VPS, a self-hosted runner,
> or locally.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

Copy-Item .env.example .env     # fill in Telegram credentials + ACCOUNT_EQUITY

python main.py --once --dry-run              # verify wiring, sends nothing
python main.py --once --dry-run --log-level DEBUG   # see why each pair was rejected
python main.py                               # run continuously
```

---

## Configuration reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | — | BotFather token. Required unless `DRY_RUN=true`. |
| `TELEGRAM_CHAT_ID` | — | Destination chat/channel. |
| `SYMBOLS` | 7 USDT pairs | Accepts `BTCUSDT` **or** `BTC/USDT`. |
| `TIMEFRAME` | `1h` | Candle size. |
| `EXCHANGE_ID` | `binance` | Must match the execution venue — see above. |
| `CANDLE_LIMIT` | `200` | Only 3 are needed; the rest is context. |
| `MIN_BODY_RATIO` | `0.05` | Doji guard on the block and impulse. `0.0` disables. |
| `STOP_BUFFER_PCT` | `0.2` | Offset beyond the distal edge, in **percent**. |
| `REWARD_RATIO` | `4` | Take-profit R-multiple. |
| `ACCOUNT_EQUITY` | `10000` | Quote currency. **Stale values mis-size every order.** |
| `RISK_PER_TRADE_PCT` | `1` | Percent of equity risked per trade. |
| `POLL_BUFFER_SECONDS` | `10` | Grace period after a candle closes. |
| `REQUEST_DELAY_SECONDS` | `0.25` | Pause between symbols. |
| `MAX_RETRIES` | `3` | Retries for transient failures. |
| `RETRY_BACKOFF_SECONDS` | `2.0` | Base delay for exponential backoff. |
| `HTTP_TIMEOUT_SECONDS` | `15.0` | Per-request timeout. |
| `LOG_LEVEL` | `INFO` | `DEBUG` shows per-symbol rejections. |
| `LOG_FILE` | *(unset)* | Optional rotating log file. |
| `DRY_RUN` | `false` | Log alerts instead of sending them. |

---

## Project layout

```
Trading Bot/
├── main.py                  entrypoint, CLI parsing, exit codes
├── scanner/
│   ├── config.py            env loading + validation (fails fast)
│   ├── exchange.py          CCXT wrapper, retries, unclosed-candle removal
│   ├── candles.py           bar geometry — no I/O
│   ├── smc.py               order blocks + vectorised FVG — no I/O
│   ├── risk.py              entry/stop/target + position sizing — no I/O
│   ├── execution.py         Binance order payloads — no I/O
│   ├── strategy.py          filter chain, typed rejection stages
│   ├── notifier.py          Telegram delivery + dry-run console notifier
│   ├── bot.py               scan loop, dedup, scheduling, shutdown
│   └── logging_setup.py     console + rotating file handlers
└── tests/                   72 tests, no network required
    ├── test_smc.py
    ├── test_risk.py
    ├── test_execution.py
    ├── test_strategy.py
    └── test_notifier.py
```

Everything except `exchange`, `notifier` and `bot` is I/O-free, so the trading
logic is testable without touching the network:

```bash
python -m pytest tests -q
```

---

## Filter funnel

Every pass logs where candidates dropped out:

```
Filter funnel: order_block=3, fvg=4
```

Stages are `warmup`, `order_block`, `fvg`, `risk`, `confirmed`, plus `error`. A
stage names where evaluation *stopped*, so `fvg=4` means four pairs formed a
valid structural block and failed only on the gap. Depth is derived from the
enum's declaration order, so inserting a stage cannot desync the helpers.

---

## Error handling

| Condition | Behaviour |
| --- | --- |
| `RateLimitExceeded`, `DDoSProtection` | Exponential backoff, then retry. |
| `NetworkError`, `RequestTimeout` | Exponential backoff, then retry. |
| `BadSymbol` | Permanent — symbol skipped, no retries consumed. |
| Unlisted or inactive symbol | Dropped at startup with a warning. |
| Venue returns short history | Warned once per symbol. |
| Structure not sizeable | Rejected at the `risk` stage with a warning. |
| Order payload cannot be built | Logged; the alert still goes out without routing. |
| Telegram `429` / `5xx` | Honours `retry_after`; retried with backoff. |
| Unexpected error in one symbol | Logged with traceback; the pass continues. |
| `SIGINT` / `SIGTERM` | Graceful shutdown. |

Configuration errors fail fast at startup with exit code `2`.

---

## Notes and limitations

- **It places no orders.** It builds payloads and displays them. Routing,
  signing and keys belong to the execution engine.
- **Entries are pending, not immediate.** A signal means "rest a limit order
  here", and price may never return to the zone. There is no fill tracking, no
  expiry, and no invalidation logic if price runs away.
- **`ACCOUNT_EQUITY` is static.** It does not read your balance. Every order is
  sized against whatever number is configured, so a stale value silently
  mis-sizes everything.
- **Notional regularly exceeds equity.** Median 1.66×, max 4.49× on the measured
  sample. Sizing ignores leverage by design; funding it is your call.
- **One structure per symbol per scan.** Only the newest three closed candles
  are examined; older unmitigated blocks are not tracked.
