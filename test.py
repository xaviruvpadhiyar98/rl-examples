import json
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, DQN, PPO

from stable_baselines3.common.env_util import make_vec_env
from envs.pattern_matching_env import PatternMatchingEnv
from hyperparams_opt import linear_schedule

from stable_baselines3.common.vec_env import VecNormalize


np.random.seed(123)





def main():
    env = PatternMatchingEnv
    eval_envs = 1
    model_name = "a2c"

    eval_vec_env = VecNormalize(make_vec_env(env, n_envs=eval_envs), training=False)

    model = {
        "a2c": A2C.load(model_name, eval_vec_env, print_system_info=True, device="cpu"),
        "ppo": PPO.load(model_name, eval_vec_env, print_system_info=True, device="auto"),
    }[model_name]



    counter = 0
    results = []
    obs = eval_vec_env.reset()
    while counter < eval_envs:
        action, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = eval_vec_env.step(action)
        for i in range(len(infos)):
            result = infos[i].copy()
            print(result)
            results.append(result)
            if dones[i]:
                # print(infos[i])
                counter += 1

    # with open(f"results_{model_name}.jsonl", "w", encoding="utf-8") as f:
    #     for r in results:
    #         json.dump(r, f)
    #         f.write('\n')



if __name__ == "__main__":
    main()
