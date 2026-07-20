"""Gymnasium environment for the swing-trading PPO agent.

Actions:  0 = cash, 1 = long, 2 = short
Observation: [dist_vwap, dist_ema8, dist_pdh, dist_pdl, spy_bias, position]

Reward: position * bar-to-bar % change, minus transaction costs, scaled by a
linear recency weight so that the oldest data in the window contributes 50%
(teaching survival / capital preservation) and the newest contributes 100%
(teaching profit extraction on current trends). This is what lets the agent
hold multi-day runners instead of exiting early.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

import config
from data import FEATURE_COLUMNS

ACTION_CASH, ACTION_LONG, ACTION_SHORT = 0, 1, 2
ACTION_TO_POSITION = {ACTION_CASH: 0.0, ACTION_LONG: 1.0, ACTION_SHORT: -1.0}


class SwingTradingEnv(gym.Env):
    """One episode = a random contiguous slice of one random ticker's history."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        datasets: dict[str, pd.DataFrame],
        episode_length: int = config.EPISODE_LENGTH,
        transaction_cost: float = config.TRANSACTION_COST_PCT,
        random_slices: bool = True,
    ):
        super().__init__()
        self.symbols = list(datasets.keys())
        self.features: dict[str, np.ndarray] = {}
        self.returns: dict[str, np.ndarray] = {}
        self.weights: dict[str, np.ndarray] = {}
        for symbol, df in datasets.items():
            feats = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            close = df["close"].to_numpy(dtype=np.float64)
            rets = np.zeros(len(close), dtype=np.float64)
            rets[1:] = close[1:] / close[:-1] - 1.0
            n = len(close)
            # Linear recency weight over the full historical window:
            # w = 0.5 + 0.5 * (bar_index / total_bars)
            idx = np.arange(n, dtype=np.float64)
            span = config.RECENCY_WEIGHT_MAX - config.RECENCY_WEIGHT_MIN
            weights = config.RECENCY_WEIGHT_MIN + span * (idx / max(n - 1, 1))
            self.features[symbol] = feats
            self.returns[symbol] = rets
            self.weights[symbol] = weights

        self.episode_length = episode_length
        self.transaction_cost = transaction_cost
        self.random_slices = random_slices

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(FEATURE_COLUMNS) + 1,),
            dtype=np.float32,
        )

        self._symbol = self.symbols[0]
        self._t = 0
        self._end = 0
        self._position = 0.0

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        row = self.features[self._symbol][self._t]
        return np.append(row, np.float32(self._position))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._symbol = self.symbols[self.np_random.integers(len(self.symbols))]
        n = len(self.returns[self._symbol])
        if self.random_slices and n > self.episode_length + 1:
            start = int(self.np_random.integers(0, n - self.episode_length - 1))
            self._t = start
            self._end = start + self.episode_length
        else:
            self._t = 0
            self._end = n - 1
        self._position = 0.0
        return self._obs(), {"symbol": self._symbol}

    def step(self, action: int):
        target = ACTION_TO_POSITION[int(action)]
        cost = abs(target - self._position) * self.transaction_cost
        self._position = target

        self._t += 1
        bar_return = self.returns[self._symbol][self._t]
        weight = self.weights[self._symbol][self._t]
        reward = float((self._position * bar_return - cost) * weight)

        terminated = False
        truncated = self._t >= self._end
        info = {
            "symbol": self._symbol,
            "position": self._position,
            "bar_return": bar_return,
        }
        return self._obs(), reward, terminated, truncated, info
