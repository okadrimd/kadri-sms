"""Automated 9sig bot (TQQQ signal strategy) with a 200-day regime overlay.

Strategy
--------
Jason Kelly's "9-Sig": hold a growth sleeve (TQQQ) and a bond sleeve (AGG),
starting 60/40. Each quarter a "signal line" grows 9%; the bot trades TQQQ
back to that line -- selling the excess into bonds in strong quarters, buying
the shortfall from bonds in weak ones.

Regime overlay (the reason this bot exists rather than raw 9sig): while the
market proxy (QQQ) closes below its 200-day SMA, the growth sleeve is parked
in the bond fund instead of TQQQ, and rebuilt when QQQ reclaims the line.
Backtested 2016-2026 this cut max drawdown from ~-65% to ~-22% while keeping
(slightly improving) the return -- see scalper/rotation analysis in git log.

Optional volatility-targeting overlay (NINE_SIG_VOL_TARGET=1, off by default):
scales TQQQ exposure by min(1, TARGET_VOL / realized_vol) so the bot holds
less when volatility spikes. In backtest this raised Sharpe from ~1.43 to
~1.6, and re-levered (100% growth, 12%/qtr) gave higher return at similar
drawdown. Intended for the base-vs-vol A/B test -- run it on a SEPARATE paper
account with its own NINE_SIG_STATE_PATH. Config knobs (env): NINE_SIG_BOND,
NINE_SIG_GROWTH_PCT, NINE_SIG_TARGET, NINE_SIG_TARGET_VOL, NINE_SIG_VOL_WINDOW.

Cadence
-------
Run once per trading day, near the close. Each run it:
  1. bumps the signal line by 9% if a new quarter has started,
  2. reads the 200-day regime,
  3. computes the desired TQQQ / AGG split (TQQQ -> bonds when risk-off),
  4. reconciles the account with fractional (notional) market orders.

Safety
------
* Paper only (TradingClient paper=True, hard-coded).
* Mode "shadow" (default) logs intended orders and places NOTHING -- run it
  like this for the first quarter and eyeball the decisions. Set
  NINE_SIG_MODE=paper to actually trade.
* Uses a SEPARATE paper account (NINE_SIG_ALPACA_API_KEY/SECRET) so it never
  touches the swing bot's positions.
* Signal-line state persists in nine_sig_state.json across restarts.

Schedule (weekday, ~15 min before the close so notional orders fill same day):

  45 15 * * 1-5 cd /path/to/trading-bot && .venv/bin/python nine_sig.py >> nine_sig.log 2>&1
"""

from __future__ import annotations

import json
import logging
import os

import pandas as pd

import config
from data import market_regime_risk_on, realized_vol

log = logging.getLogger("9sig")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

MIN_ORDER_USD = 1.0        # Alpaca notional-order minimum
REBALANCE_DEADBAND = 0.01  # ignore drift smaller than 1% of equity


def trading_client():
    from alpaca.trading.client import TradingClient

    if not config.NINE_SIG_API_KEY or not config.NINE_SIG_SECRET_KEY:
        raise RuntimeError("Set NINE_SIG_ALPACA_API_KEY / _SECRET_KEY (or the "
                           "main ALPACA keys) in the environment.")
    return TradingClient(config.NINE_SIG_API_KEY, config.NINE_SIG_SECRET_KEY,
                         paper=True)


# --- persistent signal-line state ----------------------------------------
def load_state() -> dict:
    if os.path.exists(config.NINE_SIG_STATE_PATH):
        with open(config.NINE_SIG_STATE_PATH) as fh:
            return json.load(fh)
    return {"initialized": False, "signal": None, "last_quarter": None}


def save_state(state: dict) -> None:
    with open(config.NINE_SIG_STATE_PATH, "w") as fh:
        json.dump(state, fh, indent=2)


def current_quarter() -> str:
    now = pd.Timestamp.now(tz="America/New_York")
    return f"{now.year}Q{now.quarter}"


# --- account snapshot -----------------------------------------------------
def position_values(client) -> dict[str, float]:
    return {p.symbol: float(p.market_value) for p in client.get_all_positions()}


def submit(client, symbol: str, side: str, notional: float, mode: str) -> None:
    """Place (or, in shadow mode, just log) a fractional market order."""
    if notional < MIN_ORDER_USD:
        return
    if mode != "paper":
        log.info("[shadow] would %s $%.2f of %s", side, notional, symbol)
        return
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    order = MarketOrderRequest(
        symbol=symbol, notional=round(notional, 2),
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    client.submit_order(order)
    log.info("order %s $%.2f of %s", side, notional, symbol)


def rebalance_once(client) -> None:
    mode = config.NINE_SIG_MODE
    growth, bond = config.NINE_SIG_GROWTH, config.NINE_SIG_BOND

    account = client.get_account()
    equity = float(account.equity)
    held = position_values(client)
    g_val = held.get(growth, 0.0)
    b_val = held.get(bond, 0.0)

    state = load_state()
    quarter = current_quarter()

    # First run: seed the signal line at the initial growth allocation.
    if not state["initialized"] or state["signal"] is None:
        state.update(initialized=True, signal=equity * config.NINE_SIG_INIT_GROWTH_PCT,
                     last_quarter=quarter)
        log.info("initialized: signal line = $%.2f (%.0f%% of $%.2f equity)",
                 state["signal"], config.NINE_SIG_INIT_GROWTH_PCT * 100, equity)
    # New quarter: grow the signal line by the quarterly target.
    elif quarter != state["last_quarter"]:
        state["signal"] *= (1 + config.NINE_SIG_QUARTERLY_TARGET)
        state["last_quarter"] = quarter
        log.info("new quarter %s: signal line grown to $%.2f", quarter, state["signal"])
    else:
        log.info("same quarter %s: no signal-line change (regime check only)", quarter)

    signal = state["signal"]

    # Regime overlay: park the growth sleeve in bonds while risk-off.
    risk_on = True
    if config.NINE_SIG_REGIME_ENABLED:
        risk_on = market_regime_risk_on(config.NINE_SIG_REGIME_PROXY,
                                        config.REGIME_SMA_DAYS)
        log.info("regime: %s vs %d-day SMA -> %s", config.NINE_SIG_REGIME_PROXY,
                 config.REGIME_SMA_DAYS, "RISK-ON" if risk_on else "RISK-OFF (park in bonds)")

    # Optional volatility-targeting overlay: scale the TQQQ exposure down when
    # realized volatility runs above target (hold less in turbulent markets).
    exposure = 1.0
    if config.NINE_SIG_VOL_TARGET_ENABLED and risk_on:
        rv = realized_vol(config.NINE_SIG_GROWTH, config.NINE_SIG_VOL_WINDOW)
        if rv and rv > 0:
            exposure = min(1.0, config.NINE_SIG_TARGET_VOL / rv)
        log.info("vol-target: realized %.0f%% vs target %.0f%% -> exposure %.2f",
                 (rv or 0) * 100, config.NINE_SIG_TARGET_VOL * 100, exposure)

    # Desired split: TQQQ target is the (vol-scaled) signal line when risk-on,
    # else zero. Whatever isn't in TQQQ sits in the bond sleeve.
    desired_growth = 0.0 if not risk_on else exposure * min(signal, equity)
    desired_bond = max(equity - desired_growth, 0.0)

    log.info("equity $%.2f | held %s $%.2f / %s $%.2f | target %s $%.2f / %s $%.2f",
             equity, growth, g_val, bond, b_val,
             growth, desired_growth, bond, desired_bond)

    # Reconcile: sells first (free up cash), then buys.
    orders = []
    for sym, cur, tgt in [(growth, g_val, desired_growth), (bond, b_val, desired_bond)]:
        delta = tgt - cur
        if abs(delta) < max(MIN_ORDER_USD, REBALANCE_DEADBAND * equity):
            continue
        orders.append((sym, "buy" if delta > 0 else "sell", abs(delta)))
    for sym, side, notional in sorted(orders, key=lambda o: o[1] == "buy"):
        submit(client, sym, side, notional, mode)
    if not orders:
        log.info("within deadband; no rebalance needed")

    save_state(state)


def main() -> None:
    client = trading_client()
    acct = client.get_account()
    log.info("9sig bot on PAPER account %s (equity $%s) mode=%s "
             "sleeve=%s/%s split=%.0f%% target=%.0f%%/q regime=%s vol_target=%s",
             acct.account_number, acct.equity, config.NINE_SIG_MODE,
             config.NINE_SIG_GROWTH, config.NINE_SIG_BOND,
             config.NINE_SIG_INIT_GROWTH_PCT * 100,
             config.NINE_SIG_QUARTERLY_TARGET * 100, config.NINE_SIG_REGIME_ENABLED,
             config.NINE_SIG_VOL_TARGET_ENABLED)
    rebalance_once(client)


if __name__ == "__main__":
    main()
