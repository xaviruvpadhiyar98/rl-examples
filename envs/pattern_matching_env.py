import gymnasium as gym
import numpy as np
from gymnasium import spaces

from data.actions import correct_actions
from data.ob_space import obs


class PatternMatchingEnv(gym.Env):
    """
    Custom environment for OpenAI Gym with discrete actions and a 3x3 matrix observation space.
    """

    def __init__(self):
        super(PatternMatchingEnv, self).__init__()

        # Define a discrete action space ranging from 0 to 2 (A, B, C)
        self.action_space = spaces.Discrete(3)

        # Define an observation space, a 3x3 matrix filled with random values
        low = min(obs) - 100
        high = max(obs) + 100
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(1,), dtype=np.float32
        )

        # Define a list of length 400 with random distribution of 'A', 'B', 'C'
        self.step_sequence = correct_actions

        # self.prices = [
        #     np.random.random((3, 3)) for _ in range(len(self.step_sequence))
        # ]
        self.prices = obs
        self.action_map = {0: "BUY", 1: "HOLD", 2: "SELL"}

    def step(self, action):
        reward = 0.0
        truncated = False

        action_char = self.action_map[action]
        expected_action = self.step_sequence[self.current_step]

        if action_char != "HOLD":
            reward += 100_000

        if action_char == expected_action:
            self.took_correct_actions += 1
        else:
            reward -= 100_000
            self.took_incorrect_actions += 1
            truncated = True

        info = {
            "seed": self.seed,
            "index": self.current_step,
            "price": self.prices[self.current_step],
            "correct_actions": self.took_correct_actions,
            "incorrect_actions": self.took_incorrect_actions,
            "expected_action": self.step_sequence[self.current_step],
            "model_predicted_action": action_char,
            "reward": reward,
        }
        self.current_step += 1
        done = self.current_step >= len(self.step_sequence)
        if not done or not truncated:
            self.state = [self.prices[self.current_step]]
        return self.state, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.took_correct_actions = 0
        self.took_incorrect_actions = 0
        self.seed = seed

        self.state = self.prices[self.current_step]
        return self.state, {}

    def close(self):
        pass
