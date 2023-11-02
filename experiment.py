import json
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import torch.nn as nn

from envs.pattern_matching_env import PatternMatchingEnv
from hyperparams_opt import linear_schedule

np.random.seed(123)


class TrainingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.do_eval = False

    def test(self):
        eval_envs = len(self.locals["env"].envs)

        eval_env = PatternMatchingEnv
        eval_vec_env = make_vec_env(eval_env, n_envs=eval_envs)
        eval_model = A2C(policy="MlpPolicy", env=eval_vec_env, device="cpu")
        eval_model.set_parameters(self.model.get_parameters())

        done_counter = 0
        obs = eval_vec_env.reset()
        results = []
        while done_counter < eval_envs:
            action, _ = eval_model.predict(obs, deterministic=False)
            obs, rewards, dones, infos = eval_vec_env.step(action)
            for i in range(len(infos)):
                if dones[i]:
                    results.append(infos[i])
                    done_counter += 1

        results = sorted(results, key=lambda x: x["correct_actions"], reverse=True)
        best_env = results[0]
        self.logger.record("trade/best_env_index", best_env["index"])
        self.logger.record(
            "trade/best_env_correct_actions", best_env["correct_actions"]
        )

    def on_rollout_end(self) -> None:
        self._on_rollout_end()
        self.do_eval = True

    def _on_step(self) -> bool:
        if not self.do_eval:
            return True
        self.do_eval = False
        infos = self.locals["infos"]
        results = []
        for i in range(len(infos)):
            if "episode" in infos[i]:
                if infos[i]["correct_actions"] > 2:
                    results.append(infos[i])
                    # print("train", i, infos[i])
                    self.test()
        if not results:
            return True

        results.sort(key=lambda x: x["correct_actions"], reverse=True)
        best_env = results[0]
        self.logger.record("train/best_env_index", best_env["index"])
        self.logger.record(
            "train/best_env_correct_actions", best_env["correct_actions"]
        )
        return True


def main():
    env = PatternMatchingEnv
    num_envs = 32
    model_name = "ppo"
    timestamp = 1_000_000

    vec_env = make_vec_env(env, n_envs=num_envs)

    if Path(model_name + ".zip").exists():
        model = {
            "a2c": A2C.load(model_name, vec_env, print_system_info=True, device="cpu"),
            "ppo": PPO.load(model_name, vec_env, print_system_info=True, device="auto")
        }[model_name]
    else:
        model = {
            "ppo": PPO("MlpPolicy", vec_env, verbose=2),
            "dqn": DQN("MlpPolicy", vec_env, verbose=2),
            "a2c": A2C(
                policy="MlpPolicy",
                env=vec_env,
                device="cpu",
                normalize_advantage=True,
                gamma=0.95,
                max_grad_norm=0.6,
                gae_lambda=0.99,
                n_steps=256,
                learning_rate=0.011990568639893203,
                ent_coef=0.0001762335127850959,
                vf_coef=0.15658994629928458,
                policy_kwargs=dict(
                    net_arch=dict(pi=[64, 64], vf=[64, 64]),
                    activation_fn=nn.Tanh,
                    ortho_init=True,
                ),
                verbose=2,
            ),
        }[model_name]


    model.learn(
        total_timesteps=timestamp, progress_bar=True, callback=TrainingCallback()
    )

    # counter = 0
    # results = []
    # obs = eval_vec_env.reset()
    # while counter < eval_envs:
    #     action, _ = model.predict(obs, deterministic=False)
    #     obs, rewards, dones, infos = eval_vec_env.step(action)

    #     for i in range(len(infos)):
    #         result = infos[i].copy()
    #         result.update({"model": model_name, "env_id": i, "timestamp": timestamp})
    #         results.append(result)
    #         if dones[i]:
    #             print(infos[i])
    #             counter += 1

    # with open(f"results_{model_name}.jsonl", "w", encoding="utf-8") as f:
    #     for r in results:
    #         json.dump(r, f)
    #         f.write('\n')

    model.save(model_name)


if __name__ == "__main__":
    main()