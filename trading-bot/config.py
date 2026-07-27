"""Central configuration for the PPO swing-trading bot."""

import os

# --- Universe -------------------------------------------------------------
# Trained on 8 diverse stocks for general learning...
TRAIN_TICKERS = ["AMD", "GOOGL", "AMZN", "AAPL", "NVDA", "TSLA", "MSFT", "META"]
# ...but trades are only executed on the 4 best performers
# (MSFT replaced AMZN to eliminate sideways whipsaw losses).
TRADE_TICKERS = ["AMD", "AAPL", "MSFT", "GOOGL"]
MARKET_PROXY = "SPY"  # real-time market bias shield

# --- Data -----------------------------------------------------------------
HISTORY_MONTHS = 30          # 30 months of hourly candles from Alpaca IEX
BACKTEST_MONTHS = 5          # out-of-sample holdout for the backtest
BAR_TIMEFRAME_HOURS = 1

# --- Training -------------------------------------------------------------
TOTAL_TIMESTEPS = 350_000    # the sweet spot: prevents under/over-training
EPISODE_LENGTH = 512         # bars per training episode (random slices)
SEED = 42

# Linear recency-weighted reward: oldest bar -> 0.5x, newest bar -> 1.0x
RECENCY_WEIGHT_MIN = 0.5
RECENCY_WEIGHT_MAX = 1.0

# --- Trading frictions ----------------------------------------------------
TRANSACTION_COST_PCT = 0.0005  # 5 bps slippage+fees per unit of position change

# --- Live trading ---------------------------------------------------------
MAX_POSITION_PCT = 1.0 / len(TRADE_TICKERS)  # equal-weight slots
POLL_MINUTES = 60                            # act once per hourly bar

# --- Safety net -----------------------------------------------------------
# Portfolio kill switch: liquidate everything and halt trading if account
# equity falls this far below its high-water mark. Sticky until manually
# cleared with `python risk.py reset`.
KILL_SWITCH_DD = float(os.environ.get("KILL_SWITCH_DD", "0.15"))
# Per-position stop loss: close any position whose unrealized loss exceeds
# this, then lock the ticker out until the next trading day.
POSITION_STOP_LOSS = float(os.environ.get("POSITION_STOP_LOSS", "0.10"))

# Daily market-regime overlay: block long entries (and flatten existing longs)
# while SPY closes below its 200-day simple moving average. Unlike the hourly
# VWAP shield, this is a slow daily signal that flips only a few times a year,
# so it dodges sustained bear markets without whipsaw. Backtested on the swing
# model it cut max drawdown from ~-33% to ~-20% while improving return. Default
# ON. Set REGIME_FILTER=0 to disable.
REGIME_FILTER_ENABLED = os.environ.get("REGIME_FILTER", "1") == "1"
REGIME_SMA_DAYS = 200

# SPY red-light shield (force longs to cash while SPY < session VWAP).
# Default OFF: backtested over 30 months it gave up ~90% of returns to
# whipsaw while leaving max drawdown unchanged. SPY bias still reaches the
# model through the observation vector either way. Set SPY_SHIELD=1 to
# re-enable the hard override.
SPY_SHIELD_ENABLED = os.environ.get("SPY_SHIELD", "0") == "1"

# --- Paths ----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHAMPION_MODEL_PATH = os.path.join(MODEL_DIR, "champion.zip")
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

# --- Alpaca credentials (paper account) -----------------------------------
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = True

# --- 9sig bot (nine_sig.py) -----------------------------------------------
# Jason Kelly's quarterly signal strategy on TQQQ + a bond sleeve, plus the
# 200-day regime overlay that (in backtest) cut its max drawdown from ~-65%
# to ~-22% while slightly improving return.
NINE_SIG_GROWTH = "TQQQ"          # growth (leveraged) sleeve
NINE_SIG_BOND = "AGG"             # safe sleeve
NINE_SIG_INIT_GROWTH_PCT = 0.60   # start 60% growth / 40% bond
NINE_SIG_QUARTERLY_TARGET = 0.09  # signal line grows 9% per quarter
# Regime overlay: park the growth sleeve in the bond fund while the market
# proxy (QQQ) closes below its 200-day SMA. Reuses REGIME_SMA_DAYS below.
NINE_SIG_REGIME_ENABLED = os.environ.get("NINE_SIG_REGIME", "1") == "1"
NINE_SIG_REGIME_PROXY = "QQQ"     # underlying for TQQQ (Nasdaq-100)
# Execution mode: "shadow" logs intended orders and places nothing (default,
# for the first quarter of observation); "paper" actually trades the account.
NINE_SIG_MODE = os.environ.get("NINE_SIG_MODE", "shadow")
# Dedicated paper account for 9sig; falls back to the main keys if unset, but
# use a SEPARATE account so it never collides with the swing bot's positions.
NINE_SIG_API_KEY = os.environ.get("NINE_SIG_ALPACA_API_KEY", ALPACA_API_KEY)
NINE_SIG_SECRET_KEY = os.environ.get("NINE_SIG_ALPACA_SECRET_KEY", ALPACA_SECRET_KEY)
NINE_SIG_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "nine_sig_state.json")
