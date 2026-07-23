"""Faithful backtest of the TQQQ Scalper (EMA9/21 + ROC + vol + opening-range).

Reconstructs the strategy from its documented spec and replays it on 1-minute
RTH bars for TQQQ (bullish leg) and SQQQ (bearish leg). Long-only on either
leg, one position at a time, flat overnight.

Spec faithfully modeled:
  entry (all true): EMA9 vs EMA21 direction; ROC(5) > 0; relative_volume >= 1.0;
                    ATR(14) > 0; opening-range counter-trend guard (first 30 min)
  exit (first wins): EMA cross back; force-flatten 15:55 ET
                     (take-profit / stop-loss are null/disabled in default config)
  sizing: $1,000 fixed notional -> floor(1000/price) shares
  gates: max 10 trades/day; no new entries after 15:45; daily loss halt at -$300;
         one concurrent position

Documented assumptions (spec was silent; flagged in output):
  * relative_volume lookback = 20 bars (spec gave no lookback)
  * opening-range guard: once price breaks above the 30-min OR high the market
    is "bullish" so bearish bets (long SQQQ) are blocked, and vice-versa; inside
    the range nothing is blocked. ATR filter is a no-op (min_atr_pct = 0).
  * indicators computed per-session (reset each day) since the bot flattens EOD.

The cost sweep is the whole point: market orders on 3x/inverse ETFs pay spread +
slippage every round trip, and the live account made only +$0.16/round trip gross.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PER_TRADE_DOLLARS = 1000.0
EMA_FAST, EMA_SLOW, ROC_P = 9, 21, 5
RELVOL_LOOKBACK, MIN_RELVOL = 20, 1.0
OR_MINUTES = 30
MAX_TRADES_DAY = 10
DAILY_LOSS_HALT = 300.0
NO_ENTRY_AFTER = pd.Timestamp("15:45").time()
FLATTEN_AT = pd.Timestamp("15:55").time()


def ema(arr: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def session_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].to_numpy(float)
    df = df.copy()
    df["ema_f"] = ema(c, EMA_FAST)
    df["ema_s"] = ema(c, EMA_SLOW)
    df["roc"] = df["close"].pct_change(ROC_P)
    df["relvol"] = df["volume"] / df["volume"].rolling(RELVOL_LOOKBACK, min_periods=1).mean()
    return df


def run(tqqq: pd.DataFrame, sqqq: pd.DataFrame, cost_bps_per_side: float,
        start=None, end=None) -> dict:
    """Replay the strategy; return trade blotter + summary for one cost level."""
    if start:
        tqqq, sqqq = tqqq[tqqq.index >= start], sqqq[sqqq.index >= start]
    if end:
        tqqq, sqqq = tqqq[tqqq.index <= end], sqqq[sqqq.index <= end]

    cost = cost_bps_per_side / 10_000.0
    trades = []
    for day, t_day in tqqq.groupby(tqqq.index.date):
        s_day = sqqq[sqqq.index.date == day]
        if len(t_day) < EMA_SLOW + 1 or len(s_day) < EMA_SLOW + 1:
            continue
        t_day = session_indicators(t_day)
        s_day = session_indicators(s_day)
        # 30-min opening range on TQQQ
        or_end = t_day.index[0] + pd.Timedelta(minutes=OR_MINUTES)
        opening = t_day[t_day.index < or_end]
        or_hi, or_lo = opening["high"].max(), opening["low"].min()

        pos = None            # None | ("TQQQ"/"SQQQ", entry_px, shares, entry_ts)
        trades_today = 0
        realized_today = 0.0

        # iterate by shared timestamps
        idx = t_day.index.intersection(s_day.index)
        for ts in idx:
            tr, sr = t_day.loc[ts], s_day.loc[ts]
            bullish = tr["ema_f"] > tr["ema_s"]

            # --- exits first ---
            if pos is not None:
                leg, epx, sh, ets = pos
                row = tr if leg == "TQQQ" else sr
                px = row["close"]
                ema_cross_against = (leg == "TQQQ" and not bullish) or \
                                    (leg == "SQQQ" and bullish)
                flatten = ts.time() >= FLATTEN_AT
                if ema_cross_against or flatten:
                    gross = (px - epx) * sh
                    fees = (epx + px) * sh * cost
                    pnl = gross - fees
                    realized_today += pnl
                    trades.append({"day": day, "leg": leg, "entry": ets, "exit": ts,
                                   "entry_px": epx, "exit_px": px, "shares": sh,
                                   "gross": gross, "pnl": pnl,
                                   "reason": "flatten" if flatten else "ema_cross"})
                    pos = None
                if ts.time() >= FLATTEN_AT:
                    continue

            # --- entries ---
            if pos is None:
                if trades_today >= MAX_TRADES_DAY: continue
                if ts.time() >= NO_ENTRY_AFTER: continue
                if realized_today <= -DAILY_LOSS_HALT: continue
                row = tr if bullish else sr
                # momentum + volume filters (on the leg being entered)
                if not (row["roc"] > 0.0): continue
                if not (row["relvol"] >= MIN_RELVOL): continue
                # opening-range counter-trend guard (based on TQQQ price vs OR)
                broke_up = tr["close"] > or_hi
                broke_dn = tr["close"] < or_lo
                if bullish and broke_dn: continue      # bullish bet, market broke down
                if (not bullish) and broke_up: continue  # bearish bet, market broke up
                leg = "TQQQ" if bullish else "SQQQ"
                px = row["close"]
                sh = int(PER_TRADE_DOLLARS // px)
                if sh <= 0: continue
                pos = (leg, px, sh, ts)
                trades_today += 1

    bt = pd.DataFrame(trades)
    if bt.empty:
        return {"cost_bps": cost_bps_per_side, "round_trips": 0}
    wins = (bt["pnl"] > 0).mean()
    days = bt["day"].nunique()
    return {
        "cost_bps": cost_bps_per_side,
        "round_trips": len(bt),
        "days": days,
        "rt_per_day": len(bt) / days,
        "win_rate": wins,
        "gross_total": bt["gross"].sum(),
        "net_total": bt["pnl"].sum(),
        "avg_gross_rt": bt["gross"].mean(),
        "avg_net_rt": bt["pnl"].mean(),
        "blotter": bt,
    }


def main():
    tqqq = pd.read_parquet("data_cache/TQQQ_1min.parquet")
    sqqq = pd.read_parquet("data_cache/SQQQ_1min.parquet")
    span = f"{tqqq.index[0]:%Y-%m-%d} -> {tqqq.index[-1]:%Y-%m-%d}"
    print(f"Data: TQQQ {len(tqqq):,} + SQQQ {len(sqqq):,} minute bars ({span})")
    print("Assumptions: relvol lookback=20, per-session indicators, OR guard as documented.\n")

    print("FULL AVAILABLE HISTORY — cost sensitivity (per-side bps):")
    print(f"{'cost/side':>9} {'round-trips':>12} {'rt/day':>7} {'win%':>6} "
          f"{'gross $':>10} {'net $':>10} {'net/rt':>8}")
    base = None
    for bps in [0.0, 1.0, 2.0, 3.0, 5.0]:
        r = run(tqqq, sqqq, bps)
        if bps == 0.0: base = r
        print(f"{bps:>9.0f} {r['round_trips']:>12,} {r['rt_per_day']:>7.1f} "
              f"{r['win_rate']:>6.1%} {r['gross_total']:>10,.0f} "
              f"{r['net_total']:>10,.0f} {r['avg_net_rt']:>8.3f}")

    # Validation vs the live account window
    print("\nVALIDATION — same window as the 192 live fills (Jun 29 -> Jul 22, 2026):")
    for bps in [0.0, 2.0]:
        r = run(tqqq, sqqq, bps, start="2026-06-29", end="2026-07-22")
        if r["round_trips"]:
            print(f"  cost {bps:.0f}bps: {r['round_trips']} round-trips over {r['days']} days "
                  f"({r['rt_per_day']:.1f}/day), win {r['win_rate']:.1%}, "
                  f"net ${r['net_total']:.2f} (${r['avg_net_rt']:.3f}/rt)")
    print("  live actual: 96 round-trips over 15 days (6.4/day), win 44.8%, "
          "net +$15.14 (+$0.16/rt)")


if __name__ == "__main__":
    main()
