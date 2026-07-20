"""Out-of-sample backtest of the champion model on the last 5 months.

  python backtest.py [--model models/champion.zip]

Runs the model deterministically over the holdout window for each of the four
trade tickers and reports per-ticker return plus portfolio APY, max drawdown,
and Sharpe ratio (equal-weight portfolio).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config
from data import FEATURE_COLUMNS
from data import load_dataset_cached
from train import split_train_holdout

HOURS_PER_YEAR = 252 * 6.5  # regular-session hourly bars per year


def run_ticker(model, df: pd.DataFrame) -> pd.Series:
    """Replay the model over one ticker's holdout bars; return hourly returns."""
    from trading_env import ACTION_TO_POSITION

    feats = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    close = df["close"].to_numpy(dtype=np.float64)
    position = 0.0
    strat_returns = np.zeros(len(df))
    for t in range(len(df) - 1):
        obs = np.append(feats[t], np.float32(position))
        action, _ = model.predict(obs, deterministic=True)
        target = ACTION_TO_POSITION[int(action)]
        cost = abs(target - position) * config.TRANSACTION_COST_PCT
        position = target
        bar_return = close[t + 1] / close[t] - 1.0
        strat_returns[t + 1] = position * bar_return - cost
    return pd.Series(strat_returns, index=df.index)


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def main(model_path: str) -> None:
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    datasets = load_dataset_cached()
    _, holdout = split_train_holdout(datasets)

    per_ticker: dict[str, pd.Series] = {}
    print(f"Out-of-sample backtest (last {config.BACKTEST_MONTHS} months):\n")
    for symbol in config.TRADE_TICKERS:
        returns = run_ticker(model, holdout[symbol])
        per_ticker[symbol] = returns
        total = float((1.0 + returns).prod() - 1.0)
        buy_hold = float(
            holdout[symbol]["close"].iloc[-1] / holdout[symbol]["close"].iloc[0] - 1.0
        )
        print(f"  {symbol:6s} strategy {total:+8.2%}   buy&hold {buy_hold:+8.2%}")

    # Equal-weight portfolio of the four trade tickers.
    matrix = pd.concat(per_ticker, axis=1).fillna(0.0)
    portfolio = matrix.mean(axis=1)
    equity = (1.0 + portfolio).cumprod()

    total_return = float(equity.iloc[-1] - 1.0)
    years = len(portfolio) / HOURS_PER_YEAR
    apy = (1.0 + total_return) ** (1.0 / max(years, 1e-9)) - 1.0
    sharpe = 0.0
    if portfolio.std() > 0:
        sharpe = float(portfolio.mean() / portfolio.std() * np.sqrt(HOURS_PER_YEAR))

    print("\nPortfolio (equal-weight across trade tickers):")
    print(f"  Total return : {total_return:+.2%}")
    print(f"  APY          : {apy:+.2%}")
    print(f"  Max drawdown : {max_drawdown(equity):.2%}")
    print(f"  Sharpe ratio : {sharpe:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.CHAMPION_MODEL_PATH)
    args = parser.parse_args()
    main(args.model)
