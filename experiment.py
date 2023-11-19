import json
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
import torch.nn as nn

from envs.pattern_matching_env import PatternMatchingEnv
from hyperparams_opt import linear_schedule
from copy import deepcopy

np.random.seed(123)


class TrainingCallback(BaseCallback):
    def __init__(self, eval_model, model_name):
        super().__init__()
        self.eval_model = eval_model
        self.do_eval = False
        self.model_name = model_name

    def test(self):
        eval_envs = len(self.locals["env"].envs)

        eval_env = PatternMatchingEnv
        eval_vec_env = make_vec_env(eval_env, n_envs=eval_envs)
        self.eval_model.set_parameters(self.model.get_parameters())

        done_counter = 0
        obs = eval_vec_env.reset()
        results = []
        while done_counter < eval_envs:
            action, _ = self.eval_model.predict(obs, deterministic=False)
            obs, rewards, dones, infos = eval_vec_env.step(action)
            for i in range(len(infos)):
                if dones[i]:
                    results.append(infos[i])
                    done_counter += 1

        results = sorted(results, key=lambda x: x["correct_actions"], reverse=True)
        best_env = results[0]
        print(best_env)
        self.logger.record("trade/best_env_index", best_env["current_step"])
        self.logger.record(
            "trade/best_env_correct_actions", best_env["correct_actions"]
        )

    def on_rollout_end(self) -> None:
        infos = self.locals["infos"]
        sorted_infos = sorted(infos, key=lambda x: x['correct_actions'], reverse=True)
        best_info = sorted_infos[0]
        for k, v in best_info.items():
            self.logger.record(f"train/{k}", v)
        # self._on_rollout_end()
        # self.do_eval = False

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
        print(best_env)
        self.logger.record("train/best_env_index", best_env["current_step"])
        self.logger.record(
            "train/best_env_correct_actions", best_env["correct_actions"]
        )
        return True
    
    def _on_training_end(self) -> None:
        self.test()
        return True


def create_model(model_class, vec_env, device="cpu", ent_coef=0.1, tensorboard_log="logs"):
    return model_class(
        policy="MlpPolicy",
        env=vec_env,
        device=device,
        ent_coef=ent_coef,
        tensorboard_log=tensorboard_log,
        verbose=2,
    )

def main():
    env = PatternMatchingEnv
    num_envs = 128
    model_name = "a2c"
    timestamp = 100_000_000

    vec_env = VecNormalize(make_vec_env(env, n_envs=num_envs))

    model_classes = {
        "a2c": A2C,
        "ppo": PPO,
        "dqn": DQN
    }

    if model_name not in model_classes:
        raise ValueError(f"Unsupported model name: {model_name}")


    model_class = model_classes[model_name]
    model_file = f"{model_name}.zip"

    if Path(model_file).exists():
        model = model_class.load(model_name, vec_env, print_system_info=True, device="cpu")
    else:
        model = create_model(model_class, vec_env)

    reset_num_timesteps = not Path(f"{model_name}.zip").exists()

    # if Path(model_name + ".zip").exists():
    #     reset_num_timesteps = False
    #     if model_name == "a2c":
    #         model = A2C.load(model_name, vec_env, print_system_info=True, device="cpu")
    #     elif model_name == "ppo":
    #         model = PPO.load(model_name, vec_env, print_system_info=True, device="auto")
    #     else:
    #         raise ValueError
    # else:
    #     reset_num_timesteps = True
    #     if model_name == "a2c":
    #         model = A2C(
    #             policy="MlpPolicy",
    #             env=vec_env,
    #             device="cpu",
    #             ent_coef=0.1,
    #             tensorboard_log="logs",
    #         )
    #     elif model_name == "ppo":
    #         model = PPO(
    #             policy="MlpPolicy",
    #             env=vec_env,
    #             device="cpu",
    #             ent_coef=0.1,
    #             tensorboard_log="logs",
    #         )
    #     elif model_name == "dqn":
    #         model = DQN(
    #             policy="MlpPolicy",
    #             env=vec_env,
    #             device="cpu",
    #             ent_coef=0.1,
    #             tensorboard_log="logs",
    #         )
    #     else:
    #         raise ValueError
        
        # model = {
        #     "ppo": PPO(
        #         "MlpPolicy",
        #         vec_env,
        #         verbose=2,
        #         ent_coef=0.1,
        #         tensorboard_log="logs"
        #         # policy_kwargs=dict(
        #         #     net_arch=dict(pi=[64, 128, 256, 128, 64], vf=[64, 128, 256, 128, 64]),
        #         #     activation_fn=nn.Tanh,
        #         #     ortho_init=True,
        #         # ),
        #     ),
        #     "dqn": DQN("MlpPolicy", vec_env, verbose=2),
        #     "a2c": A2C(
        #         policy="MlpPolicy",
        #         env=vec_env,
        #         device="cpu",
        #         ent_coef=0.1,
        #         tensorboard_log="logs",

        #         # normalize_advantage=True,
        #         # normalize_advantage=False,
        #         # gamma=0.95,
        #         # gamma=0.98,
        #         # max_grad_norm=0.6,
        #         # gae_lambda=0.95,
        #         # n_steps=32,
        #         # learning_rate=0.011990568639893203,
        #         # learning_rate=1.6095819036923265e-05,
        #         # ent_coef=0.0001762335127850959,
        #         # ent_coef=1.6422053936649572e-06,
        #         # vf_coef=0.15658994629928458,
        #         # vf_coef=0.5971271120046378,
        #         # use_rms_prop=True,
        #         # policy_kwargs=dict(
        #         #     net_arch=dict(pi=[64, 128, 256, 128, 64], vf=[64, 128, 256, 128, 64]),
        #         #     activation_fn=nn.Tanh,
        #         #     ortho_init=True,
        #         # ),
        #         verbose=2,
        #     ),
        # }[model_name]
        # reset_num_timesteps = True


    eval_model = deepcopy(model)
    eval_model.policy.set_training_mode(False)
    model.learn(
        total_timesteps=timestamp, progress_bar=True, callback=TrainingCallback(eval_model, model_name),
        tb_log_name=model_name,
        reset_num_timesteps=reset_num_timesteps
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
