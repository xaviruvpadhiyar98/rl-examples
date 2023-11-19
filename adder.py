import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces
import numpy as np
from gymnasium.spaces import Dict, Discrete, Box
from pathlib import Path



class EvalCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        infos = self.locals["infos"]
        sorted_infos = sorted(infos, key=lambda x: x['correct'], reverse=True)
        best_info = sorted_infos[0]
        for k, v in best_info.items():
            self.logger.record(f"train/{k}", v)

    def _on_training_end(self) -> None:
        pass


class AdditionEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(200)
        # self.observation_space = spaces.MultiDiscrete([100, 100])
        self.observation_space = Box(0, 100, (2,), np.int64)
        

    def step(self, action):
        actual_sum = np.sum(self.state)
        error = abs(actual_sum - action)
        if error == 0:
            # print(f"{self.counter} {self.state=} {action=} {actual_sum=}")
            reward = 2
            self.correct += 1
        else:
            reward = -float(error) * 2
            self.wrong += 1

        info = {
            'correct': self.correct,
            'wrong': self.wrong,
            "state": self.state,
            "sum": actual_sum,
            "model_predicted": action,
            "reward": reward,
            'counter': self.counter
        }
        if self.counter == 500:
            return self.state, reward, True, False, info

        self.counter += 1
        self.state = np.random.randint(100, size=(2,))
        return self.state, reward, False, False, info    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.counter = 0
        self.correct = 0
        self.wrong = 0
        self.state = np.random.randint(100, size=(2,))
        return self.state, {"sum": np.sum(self.state)}

    def close(self):
        ...


def test(model):
    eval_envs = 1
    eval_env = AdditionEnv
    eval_vec_env = make_vec_env(eval_env, n_envs=eval_envs)

    done_counter = 0
    obs = eval_vec_env.reset()
    results = []
    while done_counter < eval_envs:
        action, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = eval_vec_env.step(action)
        for i in range(len(infos)):
            print(infos[i])
            if dones[i]:
                results.append(infos[i])
                done_counter += 1
    results = sorted(results, key=lambda x: x["episode"]["r"], reverse=True)
    best_env = results[0]
    print(best_env)


env = AdditionEnv
model_name = 'adder_a2c'
num_envs = 128
vec_env = VecNormalize(make_vec_env(env, n_envs=num_envs))
eval_vec_env = VecNormalize(make_vec_env(env, n_envs=num_envs))


if Path(f"{model_name}.zip").exists():
    model = A2C.load('adder_a2c', vec_env, print_system_info=True, device="cpu")
else:
    model = A2C("MlpPolicy", vec_env, verbose=2, device='cpu', ent_coef=0.01)

reset_num_timesteps = not Path(f"{model_name}.zip").exists()
model.learn(total_timesteps=10_000_000, progress_bar=True, reset_num_timesteps=reset_num_timesteps, callback=EvalCallback())
mean_reward, _ = evaluate_policy(model, eval_vec_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}")
model.save('adder_a2c')
test(model)