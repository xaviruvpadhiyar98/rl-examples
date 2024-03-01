from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from adder import AdditionEnv
from pathlib import Path

def main():
    model_path = Path("models")
    env = AdditionEnv
    best_model_name = "best_adder_ppo"
    best_normalize_model_name = "best_normalize_adder_ppo.zip"

    # model_name = "adder_ppo"
    eval_envs = 1
    env_kwargs = {"max_number": 9}
    eval_vec_env = make_vec_env(env, n_envs=eval_envs, env_kwargs=env_kwargs)
    eval_vec_env = VecNormalize.load(model_path / best_normalize_model_name, eval_vec_env)

    model = PPO.load(model_path / best_model_name)
    obs = eval_vec_env.reset()
    while True:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_vec_env.step(actions)
        print(infos)
        if any(dones):
            # print()
            # print(infos)
            # print()
            break


if __name__ == "__main__":
    main()
