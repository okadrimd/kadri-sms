"""Backtest a mechanical support-ladder dip-buy on TQQQ.

Objective proxy for "trade supports/resistances, scale in, keep dry powder":
  * rolling high  = TQQQ 63-day (3-month) max
  * buy ladder    = deploy a tranche each time TQQQ closes a further step below
                    that high (default -5/-10/-15/-20/-25%, 20% of equity each)
  * dry powder    = held in a safe sleeve (AGG or SGOV) until deployed
  * exit          = when TQQQ reclaims to within 5% of the rolling high
                    (resistance), sell all tranches back to dry powder
  * optional 200-day regime filter: only buy while QQQ >= its 200d SMA, and
    flatten to dry powder when it drops below (the crash protection).
"""
from __future__ import annotations
import numpy as np, pandas as pd

def run(tqqq, qqq, safe, levels, tranche, reclaim_gap, regime, cost=0.0005):
    tr = tqqq.pct_change().fillna(0); sr = safe.pct_change().fillna(0)
    roll_hi = tqqq.rolling(63, min_periods=1).max()
    sma200 = qqq.rolling(200).mean()
    cash = 1.0; tranches = {}          # level -> current $ value
    prev_on = True; eq = []
    for i in range(len(tqqq)):
        cash *= (1 + sr.iloc[i])
        for k in tranches: tranches[k] *= (1 + tr.iloc[i])
        on = True if not regime else (not np.isnan(sma200.iloc[i]) and qqq.iloc[i] >= sma200.iloc[i])
        dd = tqqq.iloc[i] / roll_hi.iloc[i] - 1.0
        V = cash + sum(tranches.values())
        # regime flip to risk-off -> flatten all tranches to dry powder
        if regime and prev_on and not on and tranches:
            cash += sum(tranches.values()) * (1 - cost); tranches = {}
        prev_on = on
        # exit: reclaimed resistance -> sell everything
        if tranches and dd >= -reclaim_gap:
            cash += sum(tranches.values()) * (1 - cost); tranches = {}
        # entries: each un-held ladder level that price has broken below
        if on:
            for lvl in levels:
                if lvl not in tranches and dd <= lvl:
                    buy = min(tranche * V, cash)
                    if buy > 0:
                        tranches[lvl] = buy * (1 - cost); cash -= buy
        eq.append(cash + sum(tranches.values()))
    return pd.Series(eq, index=tqqq.index)

def perf(name, eq):
    r = eq.pct_change().fillna(0); yrs = len(eq)/252
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1; dd=(eq/eq.cummax()-1).min()
    shp = r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0
    print(f"  {name:44s} CAGR {cagr:+6.1%}  MaxDD {dd:6.1%}  Sharpe {shp:4.2f}")

def bh(px):
    return (1+px.pct_change().fillna(0)).cumprod()

if __name__ == "__main__":
    reg = pd.read_parquet("data_cache/regime_daily.parquet")
    agg = pd.read_parquet("data_cache/agg_daily.parquet")["agg"]
    sgov = pd.read_parquet("data_cache/sgov_daily.parquet")["sgov"]
    LEVELS = [-0.05,-0.10,-0.15,-0.20,-0.25]; TRANCHE=0.20; RECLAIM=0.05

    for label, safe, start in [("AGG dry powder (full 10y)", agg, None),
                               ("SGOV dry powder (2020+)", sgov, "2020-05-28")]:
        df = pd.DataFrame({"tqqq": reg["tqqq"], "qqq": reg["qqq"], "safe": safe}).dropna()
        if start: df = df[df.index >= start]
        t, q, s = df["tqqq"], df["qqq"], df["safe"]
        print(f"\n=== {label} | {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d} ===")
        perf("Buy & hold TQQQ", bh(t))
        perf("Ladder (no regime filter)", run(t,q,s,LEVELS,TRANCHE,RECLAIM,regime=False))
        perf("Ladder + 200d regime filter", run(t,q,s,LEVELS,TRANCHE,RECLAIM,regime=True))
