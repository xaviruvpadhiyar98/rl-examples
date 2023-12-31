import gymnasium as gym
import numpy as np
from gymnasium import spaces
import polars as pl
from pathlib import Path


class PatternMatchingEnv(gym.Env):
    """"""

    metadata = {}

    def __init__(self):
        super().__init__()

        df = pl.read_excel(Path("data/LabelTradeSBI.NS.xlsx")).drop("Datetime")
        print(df)
        raise

        self.action_space = spaces.Discrete(3)
        low = -1
        high = 1
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(18,), dtype=np.float32
        )

        df = pl.read_excel(Path("data/LabelTradeSBI.NS.xlsx")).drop("Datetime")
        self.correct_actions = df.select("Actions").to_series().to_list()
        self.all_states = df.drop("Actions").to_numpy()
        self.len_of_all_states = len(self.all_states)
        self.action_map = {0: "BUY", 1: "HOLD", 2: "SELL"}

    def step(self, action):
        reward = self.took_correct_actions
        truncated = False

        action_char = self.action_map[action]
        expected_action = self.correct_actions[self.current_step]

        if action_char == expected_action:
            self.took_correct_actions += 1
            reward += 1
        else:
            reward = -1
            self.took_incorrect_actions += 1
            truncated = True

        info = {
            "seed": self.seed,
            "current_step": self.current_step,
            "close_price": self.state[0],
            "correct_actions": self.took_correct_actions,
            "incorrect_actions": self.took_incorrect_actions,
            "expected_action": expected_action,
            "model_predicted_action": action_char,
            "reward": reward,
        }

        done = self.current_step >= self.len_of_all_states - 1
        if done or truncated:
            reward += self.current_step - self.len_of_all_states
            return self.state, reward, done, truncated, info

        self.current_step += 1
        self.state = self.all_states[self.current_step]
        return self.state, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.took_correct_actions = 0
        self.took_incorrect_actions = 0
        self.seed = seed
        self.state = self.all_states[self.current_step]
        return self.state, {}

    def close(self):
        pass
