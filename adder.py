import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from gymnasium.spaces import Discrete, Box
from pathlib import Path
from random import choice, shuffle, randint
from collections import Counter
from copy import copy


def generate_n_digit_sums(n):
    dataset = []
    for num1 in range(n + 1):
        for num2 in range(n + 1):
            dataset.append([num1, num2, num1 + num2])
    return dataset


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


class EvalCallback(BaseCallback):
    def __init__(self, model_name: str, eval_vec_env):
        super().__init__()
        self.model_name = model_name
        self.eval_vec_env = eval_vec_env

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        env = AdditionEnv
        eval_envs = 1
        env_kwargs = {"max_number": 9}
        eval_vec_env = VecNormalize(
            make_vec_env(env, n_envs=eval_envs, env_kwargs=env_kwargs)
        )
        obs = eval_vec_env.reset()
        trade_model = copy(self.model)
        print()
        for _ in range(5):
            while True:
                actions, _ = trade_model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = eval_vec_env.step(actions)
                if any(dones):
                    print(infos)
                    break
        print()


    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        if not np.any(dones):
            return True
        infos = self.locals["infos"]
        sorted_infos = sorted(
            infos, key=lambda x: (x["counter"], x["correct"]), reverse=True
        )
        best_info = sorted_infos[0]
        for k, v in best_info.items():
            self.logger.record(f"train/{k}", v)

        best_info.pop("TimeLimit.truncated")
        best_info.pop("terminal_observation")
        print(best_info)
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass

    def log(self) -> None:
        infos = self.locals["infos"]
        sorted_infos = sorted(infos, key=lambda x: x["correct"], reverse=True)
        best_info = sorted_infos[0]
        for k, v in best_info.items():
            self.logger.record(f"train/{k}", v)
        print(best_info)


class AdditionEnv(gym.Env):
    metadata = {}

    def __init__(self, max_number):
        super().__init__()
        self.max_number = max_number
        self.action_space = Discrete(start=0, n=max_number * 2 + 1)
        # self.observation_space = Box(-self.max_number*2+2, self.max_number*2-2, (5,5), np.int64)
        self.observation_space = Box(0, max_number, (2,), np.int64)

    def step(self, action):
        actual_sum = self.dataset[self.counter][2]
        model_predicted_sum = action

        reward = -abs(actual_sum - action) - 20
        if actual_sum == model_predicted_sum:
            reward = actual_sum + 20
            self.correct += 1
        else:
            self.wrong += 1

        info = {
            "num1": self.dataset[self.counter][0],
            "num2": self.dataset[self.counter][1],
            "sum": actual_sum,
            "model_predicted": model_predicted_sum,
            "reward": reward,
            "correct": self.correct,
            "wrong": self.wrong,
            "counter": self.counter,
            "correct %": round((self.correct / (self.correct + self.wrong)) * 100, 4),
            "seed": self.seed,
        }

        if self.counter == len(self.dataset) - 1:
            return self.state, reward, True, False, info

        self.counter += 1
        self.state = np.array(self.dataset[self.counter][:2])
        # self.state[-1] = np.array([self.num1, self.num2, self.result, action, reward])
        # self.num1, self.num2, self.result = self.dataset[self.counter]
        # new_state = np.array([[self.num1, self.num2, -1, -1, -1]])
        # self.state = np.concatenate((self.state, new_state), axis=0)[-5:]
        return self.state, reward, False, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seed = seed
        self.counter = 0
        self.correct = 0
        self.wrong = 0
        self.reward_tracker = []
        self.dataset = generate_n_digit_sums(self.max_number)
        [shuffle(self.dataset) for _ in range(10)]
        self.state = np.array(self.dataset[self.counter][:2])
        # self.num1, self.num2, self.result = self.dataset[self.counter]
        # initial_state = np.random.randint(-1, 0, size=(5,5), dtype=np.int64)
        # new_state = np.array([[self.num1, self.num2, -1, -1, -1]])
        # self.state = np.concatenate((initial_state, new_state), axis=0)[-5:]
        return self.state, {"state": self.state}

    def close(self):
        pass


def main():
    env = AdditionEnv
    model_name = "adder_ppo"
    num_envs = 64
    eval_envs = 1
    env_kwargs = {"max_number": 9}
    vec_env = VecNormalize(make_vec_env(env, n_envs=num_envs, env_kwargs=env_kwargs))
    eval_vec_env = VecNormalize(
        make_vec_env(env, n_envs=eval_envs, env_kwargs=env_kwargs)
    )

    hp = {
        "ent_coef": 0.04,
        "n_epochs": 5,
        "n_steps": 32 * num_envs,
        "batch_size": 256,
        "learning_rate": linear_schedule(0.003),
        "verbose": 0,
        "device": "auto",
        # "gamma": 0.93,
        # "gae_lambda": 0.99,
        "tensorboard_log": "tensorboard_log",
    }

    # model = PPO(
    #     "MlpPolicy",
    #     vec_env,
    #     **hp,
    # )
    model = PPO.load(model_name, vec_env, **hp)

    model.learn(
        total_timesteps=10_000_000,
        progress_bar=True,
        reset_num_timesteps=False,
        callback=EvalCallback(model_name, eval_vec_env),
        tb_log_name=model_name,
    )

    # result = evaluate_policy(model, eval_vec_env, return_episode_rewards=True)
    # print(result)
    model.save(f"{model_name}.zip")


if __name__ == "__main__":
    main()
