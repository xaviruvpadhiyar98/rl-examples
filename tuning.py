import numpy as np
from envs.pattern_matching_env import PatternMatchingEnv
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path

np.random.seed(123)
from copy import deepcopy

class TuningCallback(BaseCallback):
    def __init__(self, eval_model):
        super().__init__()
        self.eval_model = eval_model
        self.do_eval = False

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
        Path("best_env_final_reward").write_text(str(best_env['episode']['r']))
    
    def _on_step(self) -> bool:
        return super()._on_step()

    def _on_training_end(self) -> None:
        self.test()
        return True



def objective(trial):
    model_name = "a2c"
    device = "auto"
    if model_name == "a2c":
        device = "cpu"

    env = PatternMatchingEnv
    eval_env = PatternMatchingEnv

    num_envs = 32
    eval_envs = 16
    timestamp = 1_000_000

    vec_env = make_vec_env(env, n_envs=num_envs)
    eval_vec_env = make_vec_env(eval_env, n_envs=eval_envs)

    hp = HYPERPARAMS_SAMPLER[model_name](trial)
    hp.update({"env": vec_env, "device": device, "policy": "MlpPolicy"})

    model = {"ppo": PPO, "dqn": DQN, "a2c": A2C}[model_name](**hp)
    eval_model = deepcopy(model)
    eval_model.policy.set_training_mode(False)
    model.learn(total_timesteps=timestamp, progress_bar=True, callback=TuningCallback(eval_model))

    best_env_final_reward = Path("best_env_final_reward").read_text()
    return float(best_env_final_reward)

    # eval_env = PatternMatchingEnv
    # eval_vec_env = make_vec_env(eval_env, n_envs=eval_envs)
    # eval_model = {"ppo": PPO, "dqn": DQN, "a2c": A2C}[model_name](**hp)
    # eval_model.set_parameters(model.get_parameters())

    # done_counter = 0
    # obs = eval_vec_env.reset()
    # results = []
    # while done_counter < eval_envs:
    #     action, _ = eval_model.predict(obs, deterministic=False)
    #     obs, rewards, dones, infos = eval_vec_env.step(action)
    #     for i in range(len(infos)):
    #         if dones[i]:
    #             results.append(infos[i])
    #             done_counter += 1

    # results = sorted(results, key=lambda x: x["correct_actions"], reverse=True)
    # best_env = results[0]
    # return best_env["correct_actions"]

    # counter = 0
    # correct_actions = []
    # obs = eval_vec_env.reset()
    # while counter < eval_envs:
    #     action, _ = model.predict(obs, deterministic=False)
    #     obs, rewards, dones, infos = eval_vec_env.step(action)
    #     for i in range(eval_envs):
    #         result = infos[i].copy()
    #         result.update({"model": model_name, "env_id": i, "timestamp": timestamp})
    #         correct_actions.append(result["correct_action"])
    #         if dones[i]:
    #             print(infos[i])
    #             counter += 1
    # correct_actions.sort(reverse=True)
    # return correct_actions[0]


if __name__ == "__main__":
    import json

    from optuna import Trial, create_study
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler
    from stable_baselines3 import A2C, DQN, PPO
    from stable_baselines3.common.env_util import make_vec_env

    from hyperparams_opt import HYPERPARAMS_SAMPLER

    N_STARTUP_TRIALS = 200
    N_TRIALS = 100
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    study = create_study(
        sampler=sampler, direction="maximize", pruner=HyperbandPruner()
    )
    try:
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    except KeyboardInterrupt:
        pass
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
