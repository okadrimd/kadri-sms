"""Fetch hourly candles from Alpaca (IEX feed) and compute the indicator set.

The model never sees absolute prices. Every feature is a relative percentage
distance on the 1-hour chart:

  * dist_vwap  -- distance from the session (daily) VWAP
  * dist_ema8  -- distance from the 8-period EMA
  * dist_pdh   -- distance from the previous day's high
  * dist_pdl   -- distance from the previous day's low
  * spy_bias   -- +1 if SPY is above its own session VWAP, else -1
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config

FEATURE_COLUMNS = ["dist_vwap", "dist_ema8", "dist_pdh", "dist_pdl", "spy_bias"]

NY_TZ = "America/New_York"


def _historical_client():
    from alpaca.data.historical import StockHistoricalDataClient

    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        raise RuntimeError(
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in the environment "
            "(see .env.example)."
        )
    return StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)


def fetch_hourly_bars(
    symbols: list[str],
    start: datetime,
    end: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch 1-hour bars for `symbols` from Alpaca's IEX historical servers.

    Returns {symbol: DataFrame} indexed by New York time, restricted to
    regular trading hours.
    """
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = _historical_client()
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(config.BAR_TIMEFRAME_HOURS, TimeFrameUnit.Hour),
        start=start,
        end=end,
        feed=DataFeed.IEX,
        adjustment=Adjustment.SPLIT,
    )
    raw = client.get_stock_bars(request).df

    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if symbol not in raw.index.get_level_values(0):
            raise RuntimeError(f"No IEX bars returned for {symbol}")
        df = raw.xs(symbol, level=0).copy()
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(NY_TZ)
        df = df.sort_index()
        # Keep regular-session bars only so session VWAP / PDH / PDL are clean.
        df = df.between_time("09:30", "15:59")
        out[symbol] = df[["open", "high", "low", "close", "volume"]]
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add session VWAP, 8 EMA, and previous-day high/low to a bar frame."""
    df = df.copy()
    session = df.index.date

    # Daily (session-anchored) VWAP from typical price.
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    grouped_pv = pv.groupby(session).cumsum()
    grouped_vol = df["volume"].groupby(session).cumsum().replace(0, np.nan)
    df["vwap"] = grouped_pv / grouped_vol

    # 8-period EMA of hourly closes (short-term trend speed).
    df["ema8"] = df["close"].ewm(span=8, adjust=False).mean()

    # Previous day high / low (breakout boundary targets).
    daily = df.groupby(session).agg(day_high=("high", "max"), day_low=("low", "min"))
    daily = daily.shift(1)
    df["pdh"] = pd.Series(session, index=df.index).map(daily["day_high"])
    df["pdl"] = pd.Series(session, index=df.index).map(daily["day_low"])
    return df


def build_features(df: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """Combine a ticker frame with the SPY frame into model-ready features."""
    df = add_indicators(df)
    spy = add_indicators(spy)

    close = df["close"]
    feat = pd.DataFrame(index=df.index)
    feat["close"] = close
    feat["dist_vwap"] = close / df["vwap"] - 1.0
    feat["dist_ema8"] = close / df["ema8"] - 1.0
    feat["dist_pdh"] = close / df["pdh"] - 1.0
    feat["dist_pdl"] = close / df["pdl"] - 1.0

    # SPY market bias: above its session VWAP = green light, below = red light.
    spy_bias = np.where(spy["close"] >= spy["vwap"], 1.0, -1.0)
    feat["spy_bias"] = (
        pd.Series(spy_bias, index=spy.index).reindex(df.index).ffill().fillna(0.0)
    )

    feat = feat.dropna()
    return feat


def load_dataset(
    months: int = config.HISTORY_MONTHS,
    tickers: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch and featurize `months` of history for the training basket."""
    tickers = tickers or config.TRAIN_TICKERS
    start = datetime.now(timezone.utc) - timedelta(days=months * 30.44)
    bars = fetch_hourly_bars(tickers + [config.MARKET_PROXY], start=start)
    spy = bars.pop(config.MARKET_PROXY)
    return {symbol: build_features(df, spy) for symbol, df in bars.items()}


def market_regime_risk_on(symbol: str = config.MARKET_PROXY,
                          sma_days: int = config.REGIME_SMA_DAYS) -> bool:
    """True when `symbol` last daily close is at/above its `sma_days` SMA.

    Uses Alpaca daily IEX bars. Slow-moving (daily) so it is safe to call once
    per hourly rebalance; the underlying SMA only changes once a day.
    """
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = _historical_client()
    start = datetime.now(timezone.utc) - timedelta(days=int(sma_days * 2 + 30))
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        feed=DataFeed.IEX,
        adjustment=Adjustment.ALL,
    )
    df = client.get_stock_bars(request).df.xs(symbol, level=0).sort_index()
    if len(df) < sma_days:
        # Not enough history to judge — fail open (treat as risk-on).
        return True
    sma = df["close"].rolling(sma_days).mean().iloc[-1]
    return bool(df["close"].iloc[-1] >= sma)


def cache_path(name: str) -> str:
    os.makedirs(config.DATA_CACHE_DIR, exist_ok=True)
    return os.path.join(config.DATA_CACHE_DIR, f"{name}.parquet")


def load_dataset_cached(months: int = config.HISTORY_MONTHS,
                        refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Parquet-cached variant of load_dataset (used by train/backtest)."""
    data: dict[str, pd.DataFrame] = {}
    missing = []
    for symbol in config.TRAIN_TICKERS:
        path = cache_path(symbol)
        if not refresh and os.path.exists(path):
            data[symbol] = pd.read_parquet(path)
        else:
            missing.append(symbol)
    if missing or refresh:
        fresh = load_dataset(months=months)
        for symbol, df in fresh.items():
            df.to_parquet(cache_path(symbol))
        data = fresh
    return data
