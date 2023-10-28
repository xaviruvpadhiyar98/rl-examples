import gymnasium as gym
from gymnasium import spaces
import numpy as np

np.random.seed(123)

correct_actions = [
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "BUY",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "SELL",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "BUY",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "SELL",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "BUY",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "SELL",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "BUY",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "SELL",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "BUY",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "SELL",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "BUY",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "SELL",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "BUY",
    "HOLD",
    "HOLD",
    "HOLD",
    "SELL",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "BUY",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "SELL",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "BUY",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "SELL",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "BUY",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "SELL",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
    "HOLD",
]


class CustomEnv(gym.Env):
    """
    Custom environment for OpenAI Gym with discrete actions and a 3x3 matrix observation space.
    """

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define a discrete action space ranging from 0 to 2 (A, B, C)
        self.action_space = spaces.Discrete(3)

        # Define an observation space, a 3x3 matrix filled with random values
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float32
        )

        # Define a list of length 400 with random distribution of 'A', 'B', 'C'
        self.step_sequence = correct_actions

        self.all_states = [
            np.random.random((3, 3)) for _ in range(len(self.step_sequence))
        ]
        self.action_map = {0: "BUY", 1: "HOLD", 2: "SELL"}

        # Current step in the environment
        self.current_step = 0
        self.highest_progress = 0
        self.correct_streak = 0
        self.took_correct_action = 0
        self.took_incorrect_actions = 0
        self.previous_action = None
        self.consecutive_mistakes = 0
        self.base_penalty = 0.1  # Base penalty for an incorrect action
        self.max_penalty = 1.0  # Maximum penalty limit

        # Initial state
        self.state = None

    def step(self, action):
        action_char = self.action_map[action]

        reward = 0.0
        done = False
        truncation = False

        step_weight = (self.current_step + 1) / len(self.step_sequence)

        if action_char == self.step_sequence[self.current_step]:
            self.correct_streak += 1
            reward = (1.0 + self.correct_streak**2) * step_weight
            self.took_correct_action += 1
        else:
            self.took_incorrect_actions += 1
            reward = -2.0 * step_weight
            self.correct_streak = 0
            # truncation = True

        self.current_step += 1
        self.state = self.all_states[self.current_step]

        done = self.current_step >= len(self.step_sequence) - 1
        if done:
            reward += 10 * (1 - (self.took_incorrect_actions / self.current_step))

        info = {
            "correct_action": self.took_correct_action,
            "incorrect_actions": self.took_incorrect_actions,
            "correct_streak": self.correct_streak,
        }
        return self.state, reward, done, truncation, info


    def reset(self, seed=None, options=None):
        # Reset the environment state
        self.current_step = 0
        self.state = self.all_states[self.current_step]
        return self.state, {}

    def close(self):
        pass


if __name__ == "__main__":
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.env_util import make_vec_env

    # Initialize the environment
    env = CustomEnv
    eval_env = CustomEnv
    num_envs = 16
    eval_envs = 2
    model_name = "ppo"
    # Vectorize environment for PPO
    vec_env = make_vec_env(env, n_envs=num_envs)
    eval_vec_env = make_vec_env(eval_env, n_envs=eval_envs)

    model = {
        "ppo": PPO("MlpPolicy", vec_env),
        "dqn": DQN("MlpPolicy", vec_env),
        "a2c": A2C("MlpPolicy", vec_env),
    }[model_name]

    # Train the model
    model.learn(total_timesteps=1_000_000, progress_bar=True)

    # Test the model
    # print(evaluate_policy(model, eval_vec_env, deterministic=True))
    counter = 0
    obs = eval_vec_env.reset()
    while counter < eval_envs:
        action, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = eval_vec_env.step(action)
        for i in range(eval_envs):
            if dones[i]:
                print(
                    f"Model: {model_name}, "
                    f"Env: {i}, "
                    f"Correct Actions: {infos[i]['correct_action']}, "
                    f"Incorrect Actions: {infos[i]['incorrect_actions']}, "
                    f"Correct Streak: {infos[i]['correct_streak']}, "
                    f"Ending Reward: {infos[i]['episode']['r']} "
                )
                counter += 1
