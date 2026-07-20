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

# --- Paths ----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHAMPION_MODEL_PATH = os.path.join(MODEL_DIR, "champion.zip")
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

# --- Alpaca credentials (paper account) -----------------------------------
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = True
