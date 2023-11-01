import gymnasium as gym
from gymnasium import spaces
import numpy as np
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
        self.observation_space = spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)

        # Define a list of length 400 with random distribution of 'A', 'B', 'C'
        self.step_sequence = correct_actions

        # self.prices = [
        #     np.random.random((3, 3)) for _ in range(len(self.step_sequence))
        # ]
        self.prices = obs
        self.action_map = {0: "BUY", 1: "HOLD", 2: "SELL"}

        # Current step in the environment
        self.current_step = 0
        self.highest_progress = 0
        self.correct_streak = 0
        self.took_correct_actions = 0
        self.took_incorrect_actions = 0
        self.prev_action = None
        self.consecutive_mistakes = 0
        self.base_penalty = 0.1  # Base penalty for an incorrect action
        self.max_penalty = 1.0  # Maximum penalty limit
        self.taken_actions = []

        # Initial state
        self.state = None

    def step(self, action):
        action_char = self.action_map[action]

        reward = 0.0
        done = False
        truncated = False

        expected_action = self.step_sequence[self.current_step]
        if action_char == expected_action:
            self.took_correct_actions += 1
            if action_char == "HOLD":
                reward += 0.01
            else:
                reward += 100
        else:
            reward -= 100_000
            self.took_incorrect_actions += 1
            truncated = True

        info = {
            "index": self.current_step,
            "price": self.prices[self.current_step],
            "correct_action": self.took_correct_actions,
            "incorrect_actions": self.took_incorrect_actions,
            "expected_action": self.step_sequence[self.current_step],
            "model_predicted_action": action_char,
            "reward": reward
        }
        self.current_step += 1
        done = self.current_step >= len(self.step_sequence)
        if not done:
            self.state = [self.prices[self.current_step]]
        return self.state, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        # Reset the environment state
        self.current_step = 0
        self.state = self.prices[self.current_step]
        return self.state, {}

    def close(self):
        pass