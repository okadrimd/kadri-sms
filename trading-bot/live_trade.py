"""Live paper-trading loop for the champion PPO model.

  python live_trade.py

Once per hourly bar during regular market hours it:
  1. pulls the last ~10 sessions of hourly IEX bars for the 4 trade tickers
     and SPY,
  2. builds the same relative-distance features the model was trained on,
  3. asks the model for cash / long / short per ticker,
  4. optionally applies the SPY "Red Light" shield (off by default; enable
     with SPY_SHIELD=1) -- if SPY is below its session VWAP, long signals
     are forced to cash (shorts and cash are still allowed),
  5. reconciles positions on the Alpaca PAPER account with market orders,
     allocating an equal-weight slot of account equity per ticker.

Run it under systemd/supervisor/tmux on the server. It only ever talks to the
paper endpoint (config.ALPACA_PAPER = True).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import numpy as np

import config
from data import FEATURE_COLUMNS, build_features, fetch_hourly_bars
from trading_env import ACTION_TO_POSITION

log = logging.getLogger("live")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def trading_client():
    from alpaca.trading.client import TradingClient

    return TradingClient(
        config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=config.ALPACA_PAPER
    )


def current_positions(client) -> dict[str, float]:
    """Signed share counts per symbol currently held on the paper account."""
    positions = {}
    for pos in client.get_all_positions():
        positions[pos.symbol] = float(pos.qty)
    return positions


def latest_features() -> tuple[dict[str, "np.ndarray"], dict[str, float], bool]:
    """Return (features by ticker, last close by ticker, spy_red_light)."""
    start = datetime.now(timezone.utc) - timedelta(days=15)
    bars = fetch_hourly_bars(
        config.TRADE_TICKERS + [config.MARKET_PROXY], start=start
    )
    spy = bars.pop(config.MARKET_PROXY)

    feats, closes = {}, {}
    red_light = False
    for symbol, df in bars.items():
        feat = build_features(df, spy)
        if feat.empty:
            raise RuntimeError(f"No usable bars for {symbol}")
        last = feat.iloc[-1]
        feats[symbol] = last[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        closes[symbol] = float(last["close"])
        red_light = last["spy_bias"] < 0
    return feats, closes, red_light


def desired_qty(equity: float, price: float, direction: float) -> int:
    slot = equity * config.MAX_POSITION_PCT
    return int(direction * max(int(slot / price), 0))


def submit_delta_order(client, symbol: str, delta: int) -> None:
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    side = OrderSide.BUY if delta > 0 else OrderSide.SELL
    order = MarketOrderRequest(
        symbol=symbol, qty=abs(delta), side=side, time_in_force=TimeInForce.DAY
    )
    client.submit_order(order)
    log.info("order %s %s %d shares", side.value, symbol, abs(delta))


def rebalance_once(model, client) -> None:
    clock = client.get_clock()
    if not clock.is_open:
        log.info("market closed; next open %s", clock.next_open)
        return

    feats, closes, red_light = latest_features()
    if red_light:
        if config.SPY_SHIELD_ENABLED:
            log.info("SPY below VWAP -> RED LIGHT: long entries blocked")
        else:
            log.info("SPY below VWAP (shield disabled; informational only)")

    account = client.get_account()
    equity = float(account.equity)
    held = current_positions(client)

    for symbol in config.TRADE_TICKERS:
        held_qty = held.get(symbol, 0.0)
        position_state = float(np.sign(held_qty))
        obs = np.append(feats[symbol], np.float32(position_state))
        action, _ = model.predict(obs, deterministic=True)
        direction = ACTION_TO_POSITION[int(action)]

        # Optional real-time shield: in red-light mode the bot may only hold
        # cash or shorts, never open/keep longs. Disabled by default — see
        # SPY_SHIELD_ENABLED in config.py for the backtest rationale.
        if config.SPY_SHIELD_ENABLED and red_light and direction > 0:
            direction = 0.0

        target = desired_qty(equity, closes[symbol], direction)
        delta = int(target - held_qty)
        log.info(
            "%s action=%s held=%d target=%d (close=%.2f)",
            symbol, int(action), int(held_qty), target, closes[symbol],
        )
        if delta != 0:
            submit_delta_order(client, symbol, delta)


def sleep_until_next_bar() -> None:
    """Sleep until a couple of minutes past the next full hour."""
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=2, second=0, microsecond=0)
    time.sleep(max((next_hour - now).total_seconds(), 60))


def main() -> None:
    from stable_baselines3 import PPO

    model = PPO.load(config.CHAMPION_MODEL_PATH)
    client = trading_client()
    account = client.get_account()
    log.info(
        "connected to Alpaca PAPER account %s (equity $%s)",
        account.account_number, account.equity,
    )

    while True:
        try:
            rebalance_once(model, client)
        except Exception:
            log.exception("rebalance failed; retrying next bar")
        sleep_until_next_bar()


if __name__ == "__main__":
    main()
