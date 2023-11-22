import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces
import numpy as np
import operator as op
from gymnasium.spaces import Dict, Discrete, Box, MultiDiscrete
from pathlib import Path
import math

Box(low=np.array([0, 0, 0]), high=np.array([100, 100, 3]), dtype=np.int64)


class EvalCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        pass

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        infos = self.locals["infos"]
        sorted_infos = sorted(infos, key=lambda x: x["correct"], reverse=True)
        best_info = sorted_infos[0]
        for k, v in best_info.items():
            self.logger.record(f"train/{k}", v)

    def _on_training_end(self) -> None:
        pass


class CalcEnv(gym.Env):
    metadata = {}
    max_n = 100
    # calc_mapping = {0: np.add, 1: np.subtract, 2: np.multiply, 3: np.divide}
    calc_mapping = {
        0: op.add,
        1: op.sub,
        2: op.mul,
        3: lambda x, y: x // y if y != 0 else 0,
    }

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(
            n=(self.max_n * self.max_n) + self.max_n, start=-self.max_n
        )
        self.observation_space = Box(
            low=np.array([1, 1, 0]),
            high=np.array([self.max_n, self.max_n, 3]),
            dtype=np.int64,
        )

    def step(self, action):
        num1, num2, calc_action = self.state
        calc_action = self.calc_mapping[calc_action]
        # print(num1, num2, calc_action)
        result = int(calc_action(num1, num2))
        error = abs(result - action)
        if error == 0:
            reward = 10
            self.correct += 1
        else:
            reward = -math.exp(min(error, 10))
            self.wrong += 1

        info = {
            "correct": self.correct,
            "wrong": self.wrong,
            "num1": num1,
            "num2": num2,
            "calc_action": calc_action.__name__,
            "result": result,
            "model_predicted": action,
            "reward": reward,
            "counter": self.counter,
        }
        if self.counter == 500:
            return self.state, reward, True, False, info

        self.counter += 1
        self.state = np.array(
            [
                np.random.randint(1, 101),
                np.random.randint(1, 101),
                np.random.randint(0, 4),
            ]
        )
        return self.state, reward, False, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.counter = 0
        self.correct = 0
        self.wrong = 0
        self.state = np.array(
            [
                np.random.randint(1, 101),
                np.random.randint(1, 101),
                np.random.randint(0, 4),
            ]
        )
        return self.state, {}

    def close(self):
        ...


def test(model):
    eval_envs = 1
    eval_env = CalcEnv
    eval_vec_env = make_vec_env(eval_env, n_envs=eval_envs)

    done_counter = 0
    obs = eval_vec_env.reset()
    results = []
    while done_counter < eval_envs:
        action, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = eval_vec_env.step(action)
        for i in range(len(infos)):
            print(infos[i])
            if dones[i]:
                results.append(infos[i])
                done_counter += 1
    results = sorted(results, key=lambda x: x["episode"]["r"], reverse=True)
    best_env = results[0]
    print(best_env)


env = CalcEnv
model_name = "calc_a2c"
model_name = "calc_ppo"
num_envs = 128
vec_env = VecNormalize(make_vec_env(env, n_envs=num_envs))
eval_vec_env = VecNormalize(make_vec_env(env, n_envs=num_envs), training=False)


if Path(f"{model_name}.zip").exists():
    # model = A2C.load(model_name, vec_env, print_system_info=True, device="cpu")
    model = PPO.load(model_name, vec_env, print_system_info=True, device="auto")
else:
    # model = A2C("MlpPolicy", vec_env, verbose=2, device="cpu", ent_coef=0.01, tensorboard_log="logs")
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=2,
        device="auto",
        ent_coef=0.01,
        tensorboard_log="logs",
    )

reset_num_timesteps = not Path(f"{model_name}.zip").exists()
model.learn(
    total_timesteps=10_000_000,
    progress_bar=True,
    reset_num_timesteps=reset_num_timesteps,
    callback=EvalCallback(),
    tb_log_name=model_name,
)
mean_reward, _ = evaluate_policy(model, eval_vec_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}")
model.save(model_name)
# test(model)
