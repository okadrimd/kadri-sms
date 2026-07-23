"""Fetch and cache 1-minute RTH bars for TQQQ + SQQQ (all available history)."""
from datetime import datetime, timezone
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed
import config, os

client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
os.makedirs("data_cache", exist_ok=True)

for sym in ["TQQQ", "SQQQ"]:
    req = StockBarsRequest(
        symbol_or_symbols=sym,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 7, 23, tzinfo=timezone.utc),
        feed=DataFeed.IEX, adjustment=Adjustment.SPLIT,
    )
    df = client.get_stock_bars(req).df.xs(sym, level=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    df = df.sort_index().between_time("09:30", "15:59")
    df[["open", "high", "low", "close", "volume"]].to_parquet(f"data_cache/{sym}_1min.parquet")
    print(f"{sym}: {len(df):,} RTH minute bars  {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
print("DONE")
