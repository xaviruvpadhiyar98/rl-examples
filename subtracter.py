import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from gymnasium.spaces import Discrete, Box
from random import shuffle
from copy import copy
from pathlib import Path


def generate_n_digit_sums(n):
    dataset = []
    for num1 in range(n + 1):
        for num2 in range(n + 1):
            dataset.append([num1, num2, num1 - num2])
    return dataset


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


class EvalCallback(BaseCallback):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.model_path = Path("models")
        self.model_path.mkdir(parents=True, exist_ok=True)

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        continue_training = True
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

        if best_info["correct"] == 100:
            continue_training = self.test()
        return continue_training

    def _on_rollout_end(self) -> None:
        continue_training = self.test()
        return continue_training

    def _on_training_end(self) -> None:
        pass

    def log(self) -> None:
        infos = self.locals["infos"]
        sorted_infos = sorted(infos, key=lambda x: x["correct"], reverse=True)
        best_info = sorted_infos[0]
        for k, v in best_info.items():
            self.logger.record(f"train/{k}", v)
        print(best_info)

    def test(self):
        continue_training = True
        env = SubtractionEnv
        eval_envs = 1
        env_kwargs = {"max_number": 9}
        eval_vec_env = VecNormalize(
            make_vec_env(env, n_envs=eval_envs, env_kwargs=env_kwargs)
        )
        obs = eval_vec_env.reset()
        trade_model = copy(self.model)
        for _ in range(10):
            while True:
                actions, _ = trade_model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = eval_vec_env.step(actions)
                if any(dones):
                    correct = infos[0]["correct"]
                    wrong = infos[0]["wrong"]
                    for k, v in infos[0].items():
                        self.logger.record(f"test/{k}", v)
                    if correct > 5:
                        print(f"{'='*50}eval started{'='*50}")
                        print(infos)
                        print(f"{'='*50}eval ended{'='*50}")
                    if correct == 100 and wrong == 0:
                        self.model.save(self.model_path / f"best_{self.model_name}.zip")
                        self.model.get_vec_normalize_env().save(self.model_path / f"best_normalize_{self.model_name}.zip")
                        continue_training = False
                    break
        
        return continue_training
        


class SubtractionEnv(gym.Env):
    metadata = {}

    def __init__(self, max_number):
        super().__init__()
        self.max_number = max_number
        self.action_space = Discrete(start=0, n=max_number * 2 + 1)
        self.observation_space = Box(0, max_number, (2,), np.int64)

    def step(self, action):
        terminated = False
        done = False
        num1 = self.dataset[self.counter][0]
        num2 = self.dataset[self.counter][1]
        actual_diff = self.dataset[self.counter][2]
        model_predicted_diff = action - 9
        diff = abs(actual_diff - model_predicted_diff)

        if diff == 0:
            reward = 10
            self.correct += 1
        elif 1 <= diff <= 3:
            reward = 6 - diff
            self.wrong += 1
            terminated = True
        else:
            reward = -diff
            self.wrong += 1
            terminated = True
        
        # if actual_diff < 0 and actual_diff == model_predicted_diff:
        #     reward += 2


        info = {
            "num1": num1,
            "num2": num2,
            "sum": actual_diff,
            "model_predicted": model_predicted_diff,
            "reward": reward,
            "correct": self.correct,
            "wrong": self.wrong,
            "counter": self.counter,
            "correct %": round((self.correct / (self.correct + self.wrong)) * 100, 4),
            "seed": self.seed,
        }

        if self.counter == len(self.dataset) - 1:
            print(info)
            reward += 1000
            done = True

        if done or terminated:
            return self.state, reward, done, terminated, info

        self.counter += 1
        self.state = np.array(self.dataset[self.counter][:2])
        return self.state, reward, done, terminated, info

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
        return self.state, {"state": self.state}

    def close(self):
        pass


def main():
    model_path = Path("models")
    model_path.mkdir(parents=True, exist_ok=True)
    env = SubtractionEnv
    model_name = "subtracter_ppo"
    num_envs = 32
    env_kwargs = {"max_number": 9}
    vec_env = VecNormalize(make_vec_env(env, n_envs=num_envs, env_kwargs=env_kwargs))

    hp = {
        "ent_coef": 0.08,
        "n_epochs": 5,
        "n_steps": 32 * num_envs,
        "batch_size": 256,
        "learning_rate": linear_schedule(0.003),
        "verbose": 2,
        "device": "auto",
        # "gamma": 0.93,
        # "gae_lambda": 0.99,
        "tensorboard_log": "tensorboard_log",
    }

    model = PPO(
        "MlpPolicy",
        vec_env,
        **hp,
    )
    # model = PPO.load(model_name, vec_env, **hp)

    model.learn(
        total_timesteps=50_000_000,
        progress_bar=True,
        reset_num_timesteps=False,
        callback=EvalCallback(model_name),
        tb_log_name=model_name,
    )

    # result = evaluate_policy(model, eval_vec_env, return_episode_rewards=True)
    # print(result)
    model.save(model_path / f"{model_name}.zip")


if __name__ == "__main__":
    main()
