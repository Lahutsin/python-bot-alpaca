# Day Trader (Index Bot)

This bot now uses a multi-timeframe long-only trend strategy with explicit risk management instead of a single-bar pseudo-SuperTrend rule.

## What Changed

- Proper SuperTrend, ATR, ADX, RSI and EMA calculations based on historical OHLCV data.
- Market regime filter using SPY and QQQ, with an optional volatility proxy.
- Relative strength ranking versus QQQ before any order is sent.
- Entry logic that requires daily trend alignment, hourly pullback and 15-minute breakout confirmation.
- Position sizing from adaptive risk-per-trade, not from raw cash allocation.
- Exit logic with ATR-based stop, partial take profit at 1R and trailing stop management.
- Persistent state in `bot_state.json` and a trade journal in `trade_journal.csv` with signal score, relative strength and MFE/MAE in R.
- A `backtest.py` runner with true multi-timeframe simulation and walk-forward slices.

## Strategy Summary

Long entries are allowed only when:

- SPY is above EMA 200.
- QQQ is above EMA 200.
- The symbol is above EMA 200.
- EMA 50 is above EMA 200.
- Daily SuperTrend is long.
- ADX is above the configured threshold.
- Average daily dollar volume passes the liquidity threshold.
- Relative strength against the benchmark is above the configured floor.
- The price is not too extended from EMA 20 and RSI is not overheated.
- The hourly chart has pulled back into EMA 20 or EMA 50.
- The 15-minute chart confirms with a breakout above the prior high on stronger volume.
- The final setup score is above the minimum ranking threshold.

Open positions are managed with:

- ATR-based initial stop.
- Partial exit at 1R.
- Trailing stop that follows price and SuperTrend.
- Daily and weekly portfolio loss caps.
- Adaptive risk reduction after losing streaks or drawdown.
- Cooldown after too many consecutive losses.
- Correlation and max-position limits.

## Configuration

Copy `config.example` to `config.py` and set your Alpaca credentials.

Key parameters:

- `ALPACA_RISK_PER_TRADE_PCT`: risk budget per trade.
- `ALPACA_MAX_POSITION_PCT`: cap on gross position size.
- `ALPACA_MAX_OPEN_POSITIONS`: portfolio concentration cap.
- `ALPACA_DAILY_LOSS_LIMIT_PCT`: no new entries after daily drawdown exceeds this value.
- `ALPACA_WEEKLY_LOSS_LIMIT_PCT`: no new entries after weekly drawdown exceeds this value.
- `ALPACA_VOLATILITY_SYMBOL`: optional volatility proxy, for example `VIXY`.
- `ALPACA_MIN_ENTRY_SCORE`: minimum ranked setup quality required for a trade.
- `ALPACA_MIN_RELATIVE_STRENGTH`: required outperformance versus the benchmark.
- `ALPACA_MAX_CONSECUTIVE_LOSSES`: number of losing trades before a cooldown starts.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the live bot:

```bash
python main.py
```

Run in Docker Compose with persistent logs and runtime files stored on a mounted volume:

```bash
docker compose up -d --build
```

Compose mounts `./data` into the container at `/app/data` and writes these files there:

- `debug.log`
- `bot_state.json`
- `trade_journal.csv`
- `bot_heartbeat.json`

The container uses environment overrides for runtime paths, so your broker credentials can remain in `config.py`.
The service also includes a Docker healthcheck that validates a fresh heartbeat file from the bot loop.

Inspect current SuperTrend values:

```bash
python supertrend.py
```

Run the historical backtest and walk-forward report:

```bash
python backtest.py
```

By default, the backtest now prints a terminal summary with per-symbol scorecards, ASCII bars, walk-forward stats and a simple readiness assessment.

If you still need the raw machine-readable report:

```bash
python backtest.py --json
```

## Output Files

- `debug.log`: bot runtime log.
- `trade_journal.csv`: entries, partial exits and full exits with risk metrics.
- `trade_journal.csv`: entries, partial exits and full exits with signal score, relative strength, effective risk and MFE/MAE in R.
- `bot_state.json`: persistent open-trade state used for trailing and partial exits.
- `bot_heartbeat.json`: heartbeat timestamp used by Docker healthcheck.

## Docker Files

- `Dockerfile`: builds the bot image.
- `docker-compose.yml`: starts the bot and mounts `./data` for persistent runtime files.