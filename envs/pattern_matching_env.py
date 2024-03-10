import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from gymnasium.spaces import Discrete, Box
from copy import copy
from pathlib import Path
import polars as pl


df = pl.read_excel(Path("data/LabelTradeSBI.NS.xlsx"))
columns = df.columns
DATASET = df.select(columns[1:-1]).to_numpy().astype(np.float32)
LABELS = df["Actions"].to_list()
print(DATASET)
print(LABELS)


class LabelledTrading(gym.Env):
    metadata = {}

    def __init__(self, dataset, labels):
        super().__init__()
        self.dataset = dataset
        self.labels = labels
        self.action_space = Discrete(start=0, n=3)
        self.observation_space = Box(
            -np.inf, np.inf, shape=dataset[0].shape, dtype=np.float32
        )
        self.labels_mapping = {0: "BUY", 1: "HOLD", 2: "SELL"}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seed = seed
        self.counter = 0
        self.correct = 0
        self.wrong = 0
        self.state = self.dataset[self.counter]
        return self.state, {"state": self.state}

    def step(self, action):
        terminated = False
        done = False
        reward = 0
        correct_label = self.labels[self.counter]
        predicted_label = self.labels_mapping[action]

        if correct_label == predicted_label:
            if correct_label == "HOLD":
                reward += 10
            elif correct_label == "BUY":
                reward += 20
            else:
                reward += 20
            self.correct += 1
        else:
            reward -= self.state[0]
            self.wrong += 1
            terminated = True

        self.counter += 1
        info = {
            "correct_label": correct_label,
            "predicted_label": predicted_label,
            "reward": reward,
            "correct": self.correct,
            "wrong": self.wrong,
            "counter": self.counter,
            "seed": self.seed,
        }
        if self.counter == len(self.dataset):
            # print(info)
            reward += 1000
            done = True

        if done or terminated:
            return self.state, reward, done, terminated, info

        self.state = self.dataset[self.counter]
        return self.state, reward, done, terminated, info

    def close(self):
        pass


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


class EvalCallback(BaseCallback):
    def __init__(self, model_path: Path, model_name: str, normalize_model_name):
        super().__init__()
        self.model_path = model_path
        self.model_name = f"best_{model_name}"
        self.normalize_model_name = f"best_{normalize_model_name}"

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.n_calls % 20_000 == 0:
            self.model.save(self.model_path / self.model_name)
            self.model.get_vec_normalize_env().save(
                self.model_path / self.normalize_model_name
            )

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
        env = LabelledTrading
        eval_envs = 1
        env_kwargs = {"dataset": DATASET, "labels": LABELS}
        eval_vec_env = VecNormalize(
            make_vec_env(env, n_envs=eval_envs, env_kwargs=env_kwargs)
        )
        obs = eval_vec_env.reset()
        trade_model = copy(self.model)
        for _ in range(10):

            if continue_training:
                break



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
                        self.model.get_vec_normalize_env().save(
                            self.model_path / f"best_normalize_{self.model_name}.zip"
                        )
                        continue_training = False
                    break

        return continue_training


def main():
    model_path = Path("models")
    model_path.mkdir(parents=True, exist_ok=True)
    env = LabelledTrading
    log_name = "labelled_trading_ppo"
    model_name = f"{log_name}.zip"
    normalize_model_name = f"normalize_{log_name}.zip"
    num_envs = 64
    env_kwargs = {"dataset": DATASET, "labels": LABELS}
    vec_env = make_vec_env(env, n_envs=num_envs, env_kwargs=env_kwargs)
    vec_env = VecNormalize(vec_env)

    hp = {
        "ent_coef": 0.08,
        "n_epochs": 10,
        "n_steps": 32 * num_envs,
        "batch_size": 512,
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
    model.learn(
        total_timesteps=50_000_000,
        progress_bar=True,
        reset_num_timesteps=False,
        callback=EvalCallback(model_path, model_name, normalize_model_name),
        tb_log_name=log_name,
    )

    model.save(model_path / model_name)
    model.get_vec_normalize_env().save(
        model_path / normalize_model_name
    )


if __name__ == "__main__":
    main()
