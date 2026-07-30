# Crypto Scanner Bot v2.2

Monitors a watchlist of crypto pairs (Kraken by default) and sends a Telegram alert when an
engulfing reversal closes **in agreement with the trend regime, on above-average
volume that is also expanding bar-on-bar** — with a structural stop-loss and R:R
targets attached.

Public market data only — **no exchange API keys are required or accepted.**

---

## What's new in v2.2

A second, independent volume test: the signal candle's volume must be **strictly
greater than the volume of the candle it engulfed**.

This closes a real gap. A bar can sit comfortably above its 20-period volume
average and *still* print less volume than the bar it swallowed. Price reversed
on **fading** participation — a move drifting rather than being driven. Measured
over 7,990 closed 4h candles on 10 pairs, this occurred in **11 of 79** otherwise
valid signals (**14%**):

| | vs `VOL_SMA_20` | vs prior bar |
| --- | --- | --- |
| Median of rejected signals | **1.11×** (passes) | **0.82×** (fails) |
| Worst case (XRP/USDT) | 1.06× (passes) | **0.39×** (fails) |

That XRP signal cleared the liquidity filter while trading **61% less** volume
than the candle it engulfed. The rejected cases cluster just above the average
(median 1.11×), so the rule mostly catches marginal-volume signals — exactly
where a false positive is most likely. Surviving signals expand by a median of
**2.23×**.

Set `REQUIRE_VOLUME_EXPANSION=false` to measure the effect; the default is on.

---

## What v2.1 changed

Every confirmed signal now carries a risk plan:

- **Structural stop-loss** anchored to the extreme of the last
  `STRUCTURAL_LOOKBACK` bars, not to the signal candle — see
  [below](#why-a-structural-stop).
- **Take-profit ladder** at configurable R multiples (default 1:2 and 1:3).
- **Risk percentage** so position size can be derived from account risk.

Nothing is required to upgrade from v2.0: all three new settings
(`STRUCTURAL_LOOKBACK`, `STOP_BUFFER_PCT`, `RR_TARGETS`) have working defaults.

### Earlier: what v2.0 changed

| | v1 | v2.0 |
| --- | --- | --- |
| Signal | Engulfing pattern alone | Pattern **+** trend **+** volume |
| Trend filter | — | `close` vs `SMA_200` |
| Volume filter | — | `volume` vs `VOL_SMA_20` |
| Default timeframe | `15m` | `4h` |
| Default candles | 50 | 300 (SMA-200 needs 200+) |

Upgrading a v1 `.env` requires `TIMEFRAME=4h`, `CANDLE_LIMIT=300`,
`SMA_PERIOD=200` and `VOLUME_SMA_PERIOD=20`. The bot refuses to start if
`CANDLE_LIMIT` cannot warm up the configured SMA, rather than running and
silently emitting nothing.

---

## Quick start

### 1. Install dependencies

Python 3.10 or newer is required.

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Create a Telegram bot

1. Open Telegram and message [@BotFather](https://t.me/BotFather).
2. Send `/newbot` and follow the prompts.
3. Copy the token it returns — it looks like `123456789:AAF-xxxxxxxxxxxxxxxxxxxx`.

### 3. Find your chat ID

1. Send any message to your new bot (required — bots cannot message you first).
2. Open `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser.
3. Read the numeric id from `"chat":{"id":987654321,...}`.

> For a **channel**, add the bot as an administrator and use the channel id,
> which begins with `-100`.

### 4. Configure `.env`

```bash
# Windows (PowerShell)
Copy-Item .env.example .env
# macOS / Linux
cp .env.example .env
```

Minimum viable `.env`:

```ini
TELEGRAM_BOT_TOKEN=123456789:AAF-xxxxxxxxxxxxxxxxxxxx
TELEGRAM_CHAT_ID=987654321
SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT
TIMEFRAME=4h
CANDLE_LIMIT=300
```

`.env` is git-ignored, along with `.env.*` backups. **Never commit it.**

### 5. Run

```bash
# Verify wiring without sending anything (no credentials needed)
python main.py --once --dry-run

# See why each symbol was accepted or rejected
python main.py --once --dry-run --log-level DEBUG

# Run continuously
python main.py
```

Stop with `Ctrl+C` — shutdown is graceful and interrupts any pending sleep.

---

## Signal logic

A candle must satisfy **all four** conditions. The trend and volume comparisons
are strict; see [below](#engulfing-definition) for the one inclusive comparison
in the pattern rule.

| Filter | LONG | SHORT |
| --- | --- | --- |
| Trend | `close > SMA_200` | `close < SMA_200` |
| Volume | `volume > VOL_SMA_20` | `volume > VOL_SMA_20` |
| VSA | `volume > previous volume` | `volume > previous volume` |
| Pattern | Bullish Engulfing | Bearish Engulfing |

Both volume conditions are identical in either direction: conviction is shown by
participation, regardless of which side is winning.

**The two volume filters are independent and neither implies the other.** The
first asks whether participation is high against the recent norm; the second
whether it *expanded* over the very bar being engulfed. See
[what's new in v2.2](#whats-new-in-v22) for how often they disagree in practice.

Everything is evaluated on the **most recently closed** candle. Bybit and
Binance both return the still-forming candle as the last element of any OHLCV
response, so that bar is discarded *before* indicators are computed — otherwise
both the moving averages and the signal itself would repaint as the bar
developed.

### Engulfing definition

Writing `P` for the previous candle and `C` for the signal candle:

```
Bullish                        Bearish
P.close < P.open   (bearish)   P.close > P.open   (bullish)
C.close > C.open   (bullish)   C.close < C.open   (bearish)
C.open <= P.close              C.open >= P.close
C.close > P.open               C.close < P.open
```

**The open comparison is inclusive, the close comparison is strict.** The
textbook form demands a *gap* past the previous close (`C.open < P.close`),
which is an assumption inherited from session-based markets. In 24/7 crypto it
makes the rule depend on how a venue stitches candles rather than on price
action:

| Venue | `open == prev close` | Patterns, strict `<` | Patterns, `<=` |
| --- | --- | --- | --- |
| Bybit | **100%** | **0** | 216–240 |
| Binance | ~50% | 34–58 | 151–177 |

Bybit publishes a continuous series where every candle opens exactly at the
previous close, so `C.open < P.close` is unsatisfiable there and the strict form
detects **nothing at all**. The inclusive form gives comparable counts on both
venues. Containment still holds with equality: if the open sits at the previous
close and the close exceeds the previous open, the signal body spans the entire
previous body.

Both candles must also have a real body of at least `MIN_BODY_RATIO` of their
own high-low range. The four inequalities are trivially satisfied when the
previous body is a tick or two wide, and "engulfing" a doji says nothing about
who won the bar. Measured over ~6,000 15m candle pairs, 5.5% of raw patterns
engulfed a body narrower than 5% of its range; the `0.05` default removes those
while keeping ~94% of patterns. Set `MIN_BODY_RATIO=0.0` for the unfiltered
definition.

### How much the filters actually remove

Measured on **Kraken** over 519 closed 4h candles on each of 10 USD pairs
(5,190 evaluations):

| Stage | Count | Share |
| --- | --- | --- |
| Raw engulfing patterns | 714 | — |
| Rejected by trend filter | 337 | 47% of patterns |
| Rejected by volume filter | 230 | 61% of what remained |
| Rejected by VSA filter | 33 | 23% of what remained |
| **Confirmed signals** | **114** | **16% of patterns** |

The `VOL_SMA_20` filter is the most selective single stage. **16–18% of raw
patterns survive the chain on every venue tested** — a useful invariant, since
the raw pattern count itself varies a lot with a venue's candle conventions.

Expect about one signal per pair per 45 4h candles (~7 days). Ten pairs on `4h`
is therefore roughly one alert a day. Shorter timeframes scale that up roughly
linearly; raise `MIN_BODY_RATIO` or `VOLUME_SMA_PERIOD` if you want it quieter.

Every pass logs this funnel, so you can see where candidates are dropping out:

```
Filter funnel: pattern=2, trend=1, confirmed=0
```

The stages are `warmup`, `pattern`, `trend`, `volume`, `vsa`, `risk`,
`confirmed`, plus `error`. A stage naming where evaluation *stopped*, so
`vsa=1` means one candidate passed the trend and liquidity filters and failed
only on follow-through.

---

## Choosing an exchange

`EXCHANGE_ID` accepts any CCXT venue with public OHLCV, but venues differ in
ways that silently change what the scanner sees. Measured on 4h candles:

| Venue | Works from GitHub Actions | Candle history | `open == prev close` | Notes |
| --- | --- | --- | --- | --- |
| **Kraken** *(default)* | ✅ | ~720 | 7% | Fiat-primary — **use USD pairs** |
| Bybit | ❌ restricted IPs | 1000 | **100%** | Gapless feed, see below |
| Binance | ❌ HTTP 451 | 1000 | ~50% | Best liquidity if reachable |

Two traps worth knowing about:

**Quote currency matters on Kraken.** It is a fiat-primary venue: its USD books
carry a median **19.2×** the turnover of the matching USDT pair. The thin USDT
books print candles with no volume at all and candles where `high == low`
(DOT/USDT: ~$1.9k per 4h, 18 flat bars in 200). Two of the four filters are
volume-based, so on a thin book they measure noise rather than participation.
The two feeds track within **0.07%** at the median and produce comparable signal
counts, so USD pairs are strictly the better feed here. On a USDT-primary venue
(Binance, Bybit) use USDT pairs instead.

**Gapless feeds break gap-based rules.** Bybit stitches every candle open to the
previous close. That is why the engulfing rule compares bodies rather than
requiring a gap — see [engulfing definition](#engulfing-definition).

The scanner warns when a venue returns materially fewer candles than requested,
since a silent history cap starves the moving averages and the only symptom
would be a scanner that never alerts.

---

## Risk management

### Why a structural stop

A stop placed just under the signal candle's own low sits exactly where every
other reader of the same pattern puts theirs. That cluster is visible resting
liquidity, and price routinely wicks through it before continuing in the
original direction — the "stop hunt".

The bot instead anchors the stop to the extreme of the last
`STRUCTURAL_LOOKBACK` closed candles (including the signal candle), then pushes
it `STOP_BUFFER_PCT` beyond that level so it is not resting on the exact tick
where orders pile up.

```
LONG :  stop = min(low  of last N bars) × (1 − buffer)
SHORT:  stop = max(high of last N bars) × (1 + buffer)
```

**The trade-off is real.** Measured over the same 79 confirmed 4h signals:

| Stop style | Median risk | Mean risk |
| --- | --- | --- |
| Naive (signal candle ± 0.1%) | 2.12% | 2.28% |
| **Structural (10-bar ± 0.1%)** | **4.01%** | **4.30%** |

The structural stop is **1.89× wider** at the median. In 89% of signals it sits
strictly further out; in the remaining 11% the engulfing bar *is* the 10-bar
extreme, so the two coincide and there is no anti-sweep benefit. A wider stop
means a **smaller position** for the same account risk — that is the intended
exchange: a worse entry ratio for a materially lower chance of being swept out
of a correct call.

Observed risk across those signals ranged from 0.98% (LTC/USDT) to 14.32%
(AVAX/USDT), so position sizing genuinely has to be per-signal.

### Targets

Risk per unit is `|entry − stop|`. Each target in `RR_TARGETS` is that multiple
of risk from entry: 1:2 is twice the risk in profit, 1:3 three times.

A setup whose stop and targets cannot be expressed as real prices is **rejected**
at the `risk` stage rather than alerted without them. This is not hypothetical —
a short whose risk exceeds 1/3 of entry produces a negative 1:3 target, which no
venue can express. Those rejections appear in the funnel log.

### Sample alert

```
🟢 LONG SIGNAL · LTC/USDT

Pair: LTC/USDT
Timeframe: 4h
Signal: LONG (Bullish Engulfing)
Price: 58.11

SMA 200: 56.00  (3.77% above)
Volume: 68.39K  (1.31x the 20-period average of 52.40K)
VSA: 2.19x the engulfed candle's 31.26K  (follow-through confirmed)

🛡 Risk Management
• Entry: $58.11
• Stop-Loss (10-bar low): $56.22  (Risk: 3.25%)
• Take-Profit 1 (1:2): $61.88  (+6.49%)
• Take-Profit 2 (1:3): $63.77  (+9.74%)

Engulf ratio: 7.37x · Candle open 2026-05-14 12:00 UTC
```

The percentage beside each target is the **signed price move** required to reach
it, not the gain. A short's 1:2 target reads `-5.74%`: the gain is 5.74% but
price has to *fall* to get there, and printing `+5.74%` under a target below the
entry invites the opposite read.

---

## Command-line flags

Flags override `.env` values.

| Flag | Effect |
| --- | --- |
| `--once` | Run a single pass and exit (suitable for cron). |
| `--dry-run` | Log alerts instead of sending them; no credentials needed. |
| `--timeframe 1h` | Override `TIMEFRAME`. |
| `--symbols BTC/USDT,ETH/USDT` | Override `SYMBOLS`. |
| `--log-level DEBUG` | Override `LOG_LEVEL`; shows per-symbol filter decisions. |

---

## Configuration reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | — | BotFather token. Required unless `DRY_RUN=true`. |
| `TELEGRAM_CHAT_ID` | — | Destination chat/channel. Required unless `DRY_RUN=true`. |
| `SYMBOLS` | `BTC/USD,ETH/USD,SOL/USD` | Watchlist in CCXT unified format. |
| `TIMEFRAME` | `4h` | Candle size: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`… |
| `EXCHANGE_ID` | `kraken` | Any CCXT exchange id with public OHLCV. |
| `CANDLE_LIMIT` | `300` | Candles per request. Must be ≥ `SMA_PERIOD + 2`; venue max 1000. |
| `SMA_PERIOD` | `200` | Trend filter period. |
| `VOLUME_SMA_PERIOD` | `20` | Volume filter period. |
| `MIN_BODY_RATIO` | `0.05` | Doji filter — see above. `0.0` disables it. |
| `REQUIRE_VOLUME_EXPANSION` | `true` | VSA rule: volume must exceed the engulfed bar's. |
| `STRUCTURAL_LOOKBACK` | `10` | Bars scanned for the stop's structural anchor. |
| `STOP_BUFFER_PCT` | `0.1` | Buffer beyond that extreme, in **percent**. |
| `RR_TARGETS` | `2,3` | Reward:risk ladder, comma separated. |
| `POLL_BUFFER_SECONDS` | `10` | Grace period after a close before scanning. |
| `REQUEST_DELAY_SECONDS` | `0.25` | Pause between symbols, on top of CCXT throttling. |
| `MAX_RETRIES` | `3` | Retries for transient network/rate-limit failures. |
| `RETRY_BACKOFF_SECONDS` | `2.0` | Base delay for exponential backoff. |
| `HTTP_TIMEOUT_SECONDS` | `15.0` | Per-request timeout. |
| `LOG_LEVEL` | `INFO` | `DEBUG` shows why each symbol was rejected. |
| `LOG_FILE` | *(unset)* | Optional rotating log file (5 MB × 3 backups). |
| `DRY_RUN` | `false` | Log alerts instead of sending them. |

---

## A note on `pandas-ta`

`requirements.txt` declares **`pandas-ta-classic`**, the maintained fork of the
classic pandas-ta 0.3.x API, rather than `pandas-ta` itself. The original no
longer installs on current interpreters: 0.3.14b0 has been withdrawn from PyPI,
and the remaining 0.4.x releases require `numba`, which supports only Python
< 3.14.

[indicators.py](scanner/indicators.py) imports either distribution
transparently, so if you are pinned to Python ≤ 3.13 and prefer the original,
swapping the requirement for `pandas-ta==0.3.14b0` needs no code change. If
neither is installed it falls back to `series.rolling(n).mean()` — which is the
definition of an SMA. All three paths were verified to agree exactly (maximum
absolute difference 0.0, identical warm-up), and the test suite asserts this
whenever the library is importable.

---

## Project layout

```
Trading Bot/
├── main.py                  entrypoint, CLI parsing, exit codes
├── requirements.txt
├── .env.example             copy to .env
├── scanner/
│   ├── config.py            env loading + validation (fails fast)
│   ├── exchange.py          CCXT wrapper, retries, unclosed-candle removal
│   ├── indicators.py        SMA calculation via pandas-ta (+ fallback)
│   ├── patterns.py          bar geometry and engulfing rules — no I/O
│   ├── risk.py              structural stops and R:R targets — no I/O
│   ├── strategy.py          multi-factor filter chain, typed rejection stages
│   ├── notifier.py          Telegram delivery + dry-run console notifier
│   ├── bot.py               scan loop, dedup, scheduling, shutdown
│   └── logging_setup.py     console + rotating file handlers
└── tests/                   100 tests, no network required
    ├── test_patterns.py
    ├── test_indicators.py
    ├── test_risk.py
    ├── test_strategy.py
    └── test_notifier.py
```

The pattern, indicator, risk and strategy layers are all I/O-free, so the
trading rules can be tested without touching the network:

```bash
pip install pytest
python -m pytest tests -q
```

---

## Error handling

| Condition | Behaviour |
| --- | --- |
| `RateLimitExceeded`, `DDoSProtection` | Exponential backoff, then retry. |
| `NetworkError`, `RequestTimeout`, `ExchangeNotAvailable` | Exponential backoff, then retry. |
| `BadSymbol` | Permanent — symbol skipped, no retries consumed. |
| Unlisted or inactive symbol | Dropped at startup with a warning. |
| Indicators not yet warmed up | Symbol skipped (a new listing has no SMA-200). |
| No expressible stop/target | Signal rejected at the `risk` stage, with a warning. |
| pandas-ta raising internally | Logged, falls back to the pandas rolling mean. |
| Telegram `429` | Honours the API's own `retry_after` hint. |
| Telegram `5xx` | Retried with backoff. |
| Telegram `4xx` (bad token/chat) | Logged as permanent; scanning continues. |
| Unexpected error in one symbol | Logged with traceback; the pass continues. |
| Unexpected error in a whole pass | Logged with traceback; the loop continues. |
| `SIGINT` / `SIGTERM` | Graceful shutdown; pending sleeps interrupt immediately. |

The process never exits on a market-data or delivery failure. Configuration
errors are the exception — those fail fast at startup with exit code `2`.

---

## Scheduling behaviour

Rather than sleeping a fixed interval, the bot anchors to the exchange's candle
grid: after each pass it computes the next candle close and wakes
`POLL_BUFFER_SECONDS` afterwards. Exactly one scan per candle, no drift, no
skipped bars. After a long stall it re-synchronises to the current grid instead
of replaying the backlog. Alerts are de-duplicated per `(symbol, candle)`.

---

## Running as a background service

**Linux (systemd)** — `/etc/systemd/system/crypto-scanner.service`:

```ini
[Unit]
Description=Crypto Scanner Bot
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/crypto-scanner
ExecStart=/opt/crypto-scanner/.venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now crypto-scanner
journalctl -u crypto-scanner -f
```

**Windows** — Task Scheduler with "Run whether user is logged on or not", action
set to your venv's `python.exe` with argument `main.py`, and "Start in" set to
the project directory.

### GitHub Actions

[`.github/workflows/bot.yml`](.github/workflows/bot.yml) runs the scanner on a
cron schedule. Three things matter:

1. **Use `--once`, never bare `main.py`.** The default mode loops forever; on a
   schedule it would pin a job until the timeout and overlap the next run,
   burning Actions minutes for no benefit.
2. **Match the cron interval to `TIMEFRAME`.** Each run is a fresh process and
   the "already alerted" state is in memory, so a candle scanned by two runs is
   alerted twice. Hourly cron pairs with `TIMEFRAME=1h`; for `4h` use
   `'5 0,4,8,12,16,20 * * *'`.
3. **Exchange choice is not free.** Both Binance and Bybit restrict the IP
   ranges GitHub's runners use, so neither can serve market data from a
   workflow. The workflow pins `EXCHANGE_ID: kraken`, which is US-regulated and
   does not geo-block them. See [choosing an exchange](#choosing-an-exchange).

Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` under *Settings → Secrets and
variables → Actions*. Run it manually first from the *Actions* tab — the
`workflow_dispatch` trigger has a **dry run** checkbox that logs alerts instead
of sending them.

GitHub's scheduler is best-effort and can lag several minutes under load. A
missed run means that candle is skipped, never a duplicate alert.

---

## Notes and limitations

This is a scanner, not a trading system.

- **It places no orders.** It only observes and notifies. The stop and targets
  are suggestions to be entered manually, not instructions sent anywhere.
- **It does not size positions for you.** The risk percentage is the input to
  that calculation; the account-risk half is yours.
- **A structural stop is not an un-sweepable stop.** It sits beyond the *recent*
  swing, which is a better place than under the signal candle — not a guarantee.
- **Trend-filtered engulfing is still a weak edge.** The filters remove the
  worst counter-trend and low-conviction noise; they do not make the pattern
  predictive. Treat alerts as a prompt to look at a chart.
- **SMA-200 on a short timeframe is not a trend filter.** On `15m` it spans
  about two days. The `4h` default spans roughly 33 days, which is the regime
  the strategy is designed around. If you shorten the timeframe, consider
  whether the SMA period still means what you intend.
- **Restarts skip history.** Only the latest closed candle is ever evaluated;
  patterns missed while the bot was down are not backfilled.
- **New listings produce nothing** until they have `SMA_PERIOD` closed candles.
