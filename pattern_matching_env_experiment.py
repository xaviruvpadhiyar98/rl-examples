import numpy as np
from envs.pattern_matching_env import PatternMatchingEnv
import json
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from pathlib import Path
from hyperparams_opt import linear_schedule


np.random.seed(123)

def main():


    env = PatternMatchingEnv
    eval_env = PatternMatchingEnv
    num_envs = 32
    eval_envs = 1
    model_name = "ppo"
    timestamp = 2_000_000

    vec_env = make_vec_env(env, n_envs=num_envs)
    eval_vec_env = make_vec_env(eval_env, n_envs=eval_envs)

    if Path(model_name+".zip").exists():
        model = PPO.load(model_name, vec_env, print_system_info=True)
    else:
        model = {
            "ppo": PPO("MlpPolicy", vec_env, verbose=2),
            "dqn": DQN("MlpPolicy", vec_env, verbose=2),
            "a2c": A2C("MlpPolicy", vec_env, verbose=2, device="cpu"),
        }[model_name]

    model.learn(total_timesteps=timestamp, progress_bar=True)

    counter = 0
    results = []
    obs = eval_vec_env.reset()
    while counter < eval_envs:
        action, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = eval_vec_env.step(action)

        for i in range(eval_envs):
            result = infos[i].copy()
            result.update({"model": model_name, "env_id": i, "timestamp": timestamp})
            results.append(result)
            if dones[i]:
                print(infos[i])
                counter += 1

    with open(f"results_{model_name}.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f)
            f.write('\n')

    model.save(model_name)


if __name__ == "__main__":
    main()

