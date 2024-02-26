from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from adder import AdditionEnv


def eval_callback(l, g):
    info = l["infos"][0]
    if info["sum"] != info["model_predicted"]:
        print(info["num1"], info["num2"])


def main():
    env = AdditionEnv
    model_name = "adder_ppo"
    eval_envs = 1
    env_kwargs = {"max_number": 9}
    eval_vec_env = make_vec_env(env, n_envs=eval_envs, env_kwargs=env_kwargs)

    model = PPO.load(model_name)
    obs = eval_vec_env.reset()
    for _ in range(20):
        while True:
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_vec_env.step(actions)
            if infos[0]["sum"] != infos[0]["model_predicted"]:
                print(infos[0])
            if any(dones):
                print(infos)
                break

    # result = evaluate_policy(model, eval_vec_env, return_episode_rewards=True, callback=eval_callback, n_eval_episodes=10, deterministic=True)
    # print(result)


if __name__ == "__main__":
    main()
