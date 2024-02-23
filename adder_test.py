import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from gymnasium.spaces import Discrete, Box
from pathlib import Path
from random import choice, shuffle, randint
from collections import Counter


def generate_n_digit_sums(n):
    dataset = []
    for num1 in range(n+1):
        for num2 in range(n+1):
            dataset.append([num1, num2, num1 + num2])
    return dataset



def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func




class AdditionEnv(gym.Env):
    metadata = {}

    def __init__(self, max_number):
        super().__init__()
        self.max_number = max_number
        self.action_space = Discrete(start=0, n=max_number*2+1)
        self.observation_space = Box(-self.max_number*2+2, self.max_number*2-2, (5,5), np.int64)

    def step(self, action):
            
        reward = -abs(self.result - action)
        if self.result == action:
            reward = self.result
            self.correct += 1
        else:
            self.wrong += 1
        
        info = {
            "num1": self.num1,
            "num2": self.num2,
            "sum": self.result,
            "model_predicted": action,
            "reward": reward,
            "correct": self.correct,
            "wrong": self.wrong,
            "counter": self.counter,
            "correct %": round((self.correct / (self.correct+self.wrong)) * 100, 4),
            "seed": self.seed,
        }


        if self.counter == len(self.dataset) - 1:
            return self.state, reward, True, False, info
        
        self.counter += 1
        self.state[-1] = np.array([self.num1, self.num2, self.result, action, reward])
        self.num1, self.num2, self.result = self.dataset[self.counter]
        new_state = np.array([[self.num1, self.num2, -1, -1, -1]])
        self.state = np.concatenate((self.state, new_state), axis=0)[-5:]
        return self.state, reward, False, False, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seed = seed
        self.counter = 0
        self.correct = 0
        self.wrong = 0
        self.reward_tracker = []
        self.dataset = generate_n_digit_sums(self.max_number)
        [shuffle(self.dataset) for _ in range(10)]
        self.num1, self.num2, self.result = self.dataset[self.counter]
        initial_state = np.random.randint(-1, 0, size=(5,5), dtype=np.int64)
        new_state = np.array([[self.num1, self.num2, -1, -1, -1]])
        self.state = np.concatenate((initial_state, new_state), axis=0)[-5:]
        return self.state, {"state": self.state}

    def close(self):
        pass



def eval_callback(l, g):
    info = l['infos'][0]
    if info['sum'] != info['model_predicted']:
        print(info)



def main():
    env = AdditionEnv
    model_name = "adder_ppo"
    eval_envs = 1
    env_kwargs = {"max_number": 9}
    eval_vec_env = VecNormalize(make_vec_env(env, n_envs=eval_envs, env_kwargs=env_kwargs))

    model = PPO.load(model_name)

    result = evaluate_policy(model, eval_vec_env, return_episode_rewards=True, callback=eval_callback, n_eval_episodes=10)
    print(result)




if __name__=="__main__":
    main()