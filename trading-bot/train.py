"""Train the champion PPO model.

  python train.py [--refresh-data]

Fetches 30 months of hourly IEX candles for the 8-ticker training basket,
holds out the most recent 5 months for the out-of-sample backtest, and trains
PPO for 350,000 timesteps (the sweet spot between under- and over-training).
The result is saved as models/champion.zip, plus a timestamped archive copy.
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime

import pandas as pd

import config
from data import load_dataset_cached
from trading_env import SwingTradingEnv


def split_train_holdout(
    datasets: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Reserve the last BACKTEST_MONTHS for out-of-sample evaluation."""
    train, holdout = {}, {}
    for symbol, df in datasets.items():
        cutoff = df.index.max() - pd.DateOffset(months=config.BACKTEST_MONTHS)
        train[symbol] = df[df.index <= cutoff]
        holdout[symbol] = df[df.index > cutoff]
    return train, holdout


def train(refresh_data: bool = False) -> str:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    print(f"Loading {config.HISTORY_MONTHS} months of hourly data "
          f"for {config.TRAIN_TICKERS} ...")
    datasets = load_dataset_cached(refresh=refresh_data)
    train_data, _ = split_train_holdout(datasets)
    for symbol, df in train_data.items():
        print(f"  {symbol}: {len(df)} bars "
              f"({df.index.min():%Y-%m-%d} -> {df.index.max():%Y-%m-%d})")

    env = Monitor(SwingTradingEnv(train_data))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        ent_coef=0.005,
        seed=config.SEED,
        verbose=1,
    )
    print(f"Training PPO for {config.TOTAL_TIMESTEPS:,} timesteps ...")
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, progress_bar=True)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(config.MODEL_DIR, f"ppo_swing_{stamp}.zip")
    model.save(archive_path)
    shutil.copyfile(archive_path, config.CHAMPION_MODEL_PATH)
    print(f"Saved champion model -> {config.CHAMPION_MODEL_PATH}")
    print(f"Archived copy       -> {archive_path}")
    return config.CHAMPION_MODEL_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-data", action="store_true",
                        help="refetch hourly candles instead of using the cache")
    args = parser.parse_args()
    train(refresh_data=args.refresh_data)
