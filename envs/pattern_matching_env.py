import gymnasium as gym
import numpy as np
from gymnasium import spaces

# from data.actions import correct_actions
# from data.ob_space import obs
import polars as pl
from pathlib import Path




class PatternMatchingEnv(gym.Env):
    """
    Custom environment for OpenAI Gym with discrete actions and a 3x3 matrix observation space.
    """

    def __init__(self):
        super(PatternMatchingEnv, self).__init__()

        self.action_space = spaces.Discrete(3)
        low = -np.inf
        high = np.inf
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(18,), dtype=np.float32
        )

        df = (
            pl
            .read_excel(Path("data/LabelTradeSBI.NS.xlsx"))
            .drop("Datetime")
        )
        self.correct_actions = df.select("Actions").to_series().to_list()
        self.all_states = df.drop("Actions").to_numpy()
        self.action_map = {0: "BUY", 1: "HOLD", 2: "SELL"}

    def step(self, action):
        reward = 0
        truncated = False

        action_char = self.action_map[action]
        expected_action = self.correct_actions[self.current_step]

        if action_char == "BUY" or action_char == "SELL":
            reward += 50

        if action_char == "HOLD":
            reward -= 100

        if action_char == expected_action:
            self.took_correct_actions += 1
            reward += 500 if action_char in ["BUY", "SELL"] else 0.001
        else:
            # reward -= 50 if action_char == "HOLD" else 200
            reward -= 100_000
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

        done = self.current_step >= len(self.all_states) - 1
        if done:
            reward += max(0, 10000 - (self.took_incorrect_actions * 100))
            reward += self.took_correct_actions * 10
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
