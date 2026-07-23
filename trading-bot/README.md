# PPO Swing-Trading Bot (Alpaca Paper Trading)

A reinforcement-learning swing-trading bot that holds trades for roughly 1–3
days. It follows the design described in the original write-up:

- **Brain**: PPO (Proximal Policy Optimization) from `stable-baselines3`,
  trained for **350,000 timesteps** (the sweet spot between under- and
  over-training).
- **Memory**: **30 months** of historical **hourly** candles from Alpaca's
  IEX historical servers.
- **Basket**: trained on 8 diverse stocks (`AMD, GOOGL, AMZN, AAPL, NVDA,
  TSLA, MSFT, META`) but executes trades only on the 4 best performers:
  `AMD, AAPL, MSFT, GOOGL`.
- **Vision** (relative % distances on the 1-hour chart, never absolute
  prices): daily VWAP, 8 EMA, previous-day high/low (PDH/PDL), and a
  **SPY market-bias signal** fed to the model.
- **Drawdown control — daily 200-day regime filter** (on by default): while
  SPY closes below its 200-day SMA the bot blocks long entries and flattens
  existing longs. This is a *slow* daily signal that flips only a few times a
  year, so unlike the hourly VWAP shield it dodges sustained bears without
  whipsaw. Backtested on the model it cut max drawdown from ~-33% to ~-20%
  while improving return and Sharpe. Disable with `REGIME_FILTER=0`.
- The hourly "Red Light" VWAP shield (force longs to cash while SPY < its
  *session VWAP*) is **off by default** — over 30 months it gave up ~90% of
  returns to hourly whipsaw with no drawdown benefit. Enable with
  `SPY_SHIELD=1` only if you specifically want that hard override.
- **Reward engine**: linear recency-weighted reward
  `weight = 0.5 + 0.5 * (bar_index / total_bars)` — old data (~2.5 years
  back) counts 50% (survival instincts), today's data counts 100% (profit
  extraction). This is what lets the agent hold multi-day runners instead of
  exiting early.
- **Retraining**: automatic every Saturday via `retrain.py` + cron, with a
  champion/challenger gate so a worse retrain never replaces a better model.

## Setup

```bash
cd trading-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create your **paper** API keys at https://app.alpaca.markets (top-left
account switcher → Paper account → API keys), then:

```bash
cp .env.example .env       # fill in the two keys
export $(grep -v '^#' .env | xargs)
```

The bot only ever connects with `paper=True`, so it cannot touch a live
account.

## Usage

```bash
# 1. Fetch 30 months of hourly data and train the champion (350k steps).
#    Takes a while on CPU — roughly 1-3 hours depending on the machine.
python train.py

# 2. Out-of-sample backtest on the most recent 5 months (held out of training)
python backtest.py

# 3. Start live paper trading (acts once per hourly bar during market hours)
python live_trade.py
```

Run `live_trade.py` under `tmux`, `systemd`, or `supervisor` on your server so
it survives disconnects.

## Daily Telegram close report

`daily_report.py` sends a summary of every configured paper account (equity,
day P&L, open positions, fills) to your Telegram bot after the close. Add
`TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` and any extra account keys to
`.env` (see `.env.example`), test with `python daily_report.py --print`,
then schedule it 10 minutes after the close:

```cron
10 16 * * 1-5 cd /path/to/trading-bot && .venv/bin/python daily_report.py >> report.log 2>&1
```

(Cron times are in the machine's local timezone — the line above assumes
US/Eastern; adjust if your server runs on something else.)

## Automatic Saturday retraining

```cron
0 8 * * 6 cd /path/to/trading-bot && .venv/bin/python retrain.py >> retrain.log 2>&1
```

`retrain.py` refetches the full 30-month window, trains a challenger for 350k
steps, backtests both the challenger and the current champion on the same
out-of-sample window, and only promotes the challenger if it wins. Every
trained model is also archived under `models/` with a timestamp for rollback.

## Files

| File | Purpose |
| --- | --- |
| `config.py` | Tickers, timesteps, weights, costs, paths, credentials |
| `data.py` | Alpaca IEX hourly bars + VWAP / EMA8 / PDH / PDL / SPY-bias features |
| `trading_env.py` | Gymnasium env (cash/long/short) with recency-weighted reward |
| `train.py` | Trains PPO 350k steps, saves `models/champion.zip` |
| `backtest.py` | Deterministic replay on the 5-month holdout, APY / DD / Sharpe |
| `live_trade.py` | Hourly paper-trading loop with the SPY red-light shield |
| `risk.py` | Safety net: portfolio kill switch + per-position stop losses |
| `retrain.py` | Saturday champion/challenger retraining |
| `daily_report.py` | End-of-day Telegram summary across all paper accounts |

## Safety net

Two independent layers protect the account (both checked every hourly bar):

- **Portfolio kill switch** (`KILL_SWITCH_DD`, default **15%**): if account
  equity falls 15% below its high-water mark, every position is liquidated
  and trading halts. The halt is sticky across restarts — inspect with
  `python risk.py status` and clear it deliberately with
  `python risk.py reset` once you've decided it's safe to resume.
- **Per-position stop loss** (`POSITION_STOP_LOSS`, default **10%**): any
  position down more than 10% from entry is closed immediately and that
  ticker is locked out until the next trading day, so the model can't
  instantly re-buy a falling knife.

Both thresholds can be overridden via environment variables. Note the checks
run on the hourly loop, so an overnight gap can exit worse than the stop
level — acceptable for paper trading, but know the limitation.

## Honest caveats

Backtested returns (the original post cites 73% APY, 5.8–6.8% max drawdown,
1.37 Sharpe) are in-sample-adjacent and **do not guarantee** live results —
that's exactly why this runs on a paper account first. Expect slippage,
IEX-vs-SIP data differences, and regime changes to eat into the backtest
numbers. Keep it on paper until it has survived at least a few months.
