"""Weekly automatic retraining (run every Saturday).

  python retrain.py

Refetches the full 30-month hourly window, retrains PPO for 350k timesteps,
and promotes the new model to champion only if it beats the current champion
on the out-of-sample backtest (portfolio total return). The previous champion
is kept as a timestamped archive either way, so you can always roll back.

Cron entry (Saturday 08:00, after Friday's close is in the data):

  0 8 * * 6 cd /path/to/trading-bot && /usr/bin/python3 retrain.py >> retrain.log 2>&1
"""

from __future__ import annotations

import os

import pandas as pd

import config
from backtest import run_ticker
from data import load_dataset_cached
from train import split_train_holdout, train


def portfolio_return(model, holdout: dict[str, pd.DataFrame]) -> float:
    per_ticker = {s: run_ticker(model, holdout[s]) for s in config.TRADE_TICKERS}
    matrix = pd.concat(per_ticker, axis=1).fillna(0.0)
    portfolio = matrix.mean(axis=1)
    return float((1.0 + portfolio).prod() - 1.0)


def main() -> None:
    from stable_baselines3 import PPO

    had_champion = os.path.exists(config.CHAMPION_MODEL_PATH)
    old_champion = PPO.load(config.CHAMPION_MODEL_PATH) if had_champion else None

    # train() refreshes the data cache and overwrites champion.zip, but also
    # writes a timestamped archive of every model it produces.
    new_path = train(refresh_data=True)
    new_model = PPO.load(new_path)

    datasets = load_dataset_cached()
    _, holdout = split_train_holdout(datasets)

    new_score = portfolio_return(new_model, holdout)
    print(f"Challenger out-of-sample portfolio return: {new_score:+.2%}")

    if old_champion is not None:
        old_score = portfolio_return(old_champion, holdout)
        print(f"Previous champion on the same window:      {old_score:+.2%}")
        if old_score > new_score:
            print("Previous champion wins -> restoring it as champion.zip")
            old_champion.save(config.CHAMPION_MODEL_PATH)
            return
    print("Challenger promoted to champion.zip")


if __name__ == "__main__":
    main()
