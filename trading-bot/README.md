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
  real-time **SPY market-bias shield** — when SPY is below its session VWAP
  the bot goes "Red Light" and may only hold cash or shorts.
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
| `retrain.py` | Saturday champion/challenger retraining |

## Honest caveats

Backtested returns (the original post cites 73% APY, 5.8–6.8% max drawdown,
1.37 Sharpe) are in-sample-adjacent and **do not guarantee** live results —
that's exactly why this runs on a paper account first. Expect slippage,
IEX-vs-SIP data differences, and regime changes to eat into the backtest
numbers. Keep it on paper until it has survived at least a few months.
