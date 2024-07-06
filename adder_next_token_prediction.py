import torch
from random import shuffle
import numpy as np
import random
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
from copy import copy


seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def make_dataset():
    ds = []
    for i in range(100):
        for j in range(100):
            s = i+j
            ds.append([i//10, i%10, j//10, j%10, s//100, (s//10)%10, s%10])

    shuffle(ds)
    ds = np.array(ds).astype(np.float32)
    ds_X = ds[:, 0:6]
    ds_Y = np.copy(ds[:, 1:])

    train_size = int(len(ds_X) * 0.85)

    ds_X_train, ds_X_test = ds_X[0:train_size], ds_X[train_size:]
    ds_Y_train, ds_Y_test = ds_Y[0:train_size], ds_Y[train_size:]
    return ds_X_train, ds_Y_train, ds_X_test, ds_Y_test


class AdderTokenPredictorEnv(gym.Env):
    metadata = {}
    
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.action_space = Discrete(start=0, n=10)
        self.observation_space = Box(0, 10, shape=(len(x[0]),), dtype=np.float32)
        self.len_of_dataset = len(x)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.wrong = 0
        self.counter = 0
        self.state = self.x[self.counter]
        return self.state, {}
    
    def step(self, action):
        terminated = False
        done = False
        info = {}
        y = self.y[self.counter]
        correct_action = int(y[-1])
        if action == correct_action:
            reward = 1
        else:
            reward = -abs(correct_action-action)
            self.wrong += 1
        
        self.counter += 1
        if self.counter == self.len_of_dataset:
            done = True

        info["done"] = done
        info["terminated"] = terminated
        info["counter"] = self.counter
        info["remaining"] = self.len_of_dataset - self.counter
        info["wrong"] = self.wrong
        info["correct"] = correct_action
        info["predicted"] = action
        info["x"] = self.state 
        info["y"] = y


        if done or terminated:
            return self.state, reward, done, terminated, info

        self.state = self.x[self.counter]
        return self.state, reward, done, terminated, info

    def close(self):
        pass



def make_env(x, y):
    def thunk():
        combined = list(zip(x, y))
        shuffle(combined)
        x_shuffled, y_shuffled = zip(*combined)
        x_shuffled, y_shuffled = list(x_shuffled), list(y_shuffled)
        env = AdderTokenPredictorEnv(x_shuffled, y_shuffled)
        env = Monitor(env)
        # env = NormalizeObservation(env)
        # env = RecordEpisodeStatistics(env)
        # env = NormalizeReward(env)
        return env

    return thunk

class EvalCallback(BaseCallback):
    def __init__(self, eval_vec_env):
        super().__init__()
        self.eval_vec_env = eval_vec_env
        self.continue_training = True


    def _on_training_start(self) -> None:
        self.continue_training = self.test2()

    def _on_step(self) -> bool:
        return self.continue_training
    
    def _on_rollout_start(self) -> None:
        pass

    def _on_rollout_end(self) -> None:
        self.continue_training = self.test2()

    def _on_training_end(self) -> None:
        self.continue_training = self.test2()

    def test2(self):
        print("-" * 100)
        sync_envs_normalization(self.training_env, self.eval_vec_env)
        trade_model = copy(self.model)
        completed_envs = np.zeros(self.eval_vec_env.num_envs, dtype=bool)
        obs = self.eval_vec_env.reset()
        wrong = 0

        while True:
            actions, _ = trade_model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = self.eval_vec_env.step(actions)
            for info in infos:
                if info["correct"] != info["predicted"]:
                    print(info["x"], info["y"], info["correct"], info["predicted"])

            newly_completed = dones & ~completed_envs
            completed_envs |= dones

            for idx, is_newly_completed in enumerate(newly_completed):
                if not is_newly_completed:
                    continue

                wrong = (infos[idx]["wrong"])
                self.logger.record(f"test/wrong", wrong)
                print(f"test/wrong: {wrong}")
                if wrong == 0:
                    return False
                
            if np.all(completed_envs):
                break
    
        return True

def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func



def main():
    ds_X_train, ds_Y_train, ds_X_test, ds_Y_test = make_dataset()    
    print(ds_X_train[0], ds_Y_train[0])
    print(ds_X_test[0], ds_Y_test[0])

    model_path = Path("models")
    model_path.mkdir(parents=True, exist_ok=True)
    log_name = "adder_token_predictor"
    model_name = f"{log_name}.zip"
    normalize_model_name = f"norm_{log_name}.zip"

    vec_env = DummyVecEnv([make_env(ds_X_train, ds_Y_train)] * 64)
    eval_vec_env = DummyVecEnv([make_env(ds_X_test, ds_Y_test)])
    # vec_env = VecNormalize(vec_env)
    # eval_vec_env = VecNormalize(eval_vec_env)


    ppo = PPO(
        "MlpPolicy",
        vec_env,
        device="cuda",
        learning_rate=linear_schedule(0.0001),
        n_steps=2048,
        batch_size=64,
        tensorboard_log="logs",
    )

    callback = EvalCallback(eval_vec_env=eval_vec_env)
    ppo.learn(
        total_timesteps=10_000_000,
        progress_bar=True,
        reset_num_timesteps=False,
        callback=callback,
        tb_log_name=log_name,
    )

    ppo.save(model_path / model_name)
    # ppo.get_vec_normalize_env().save(model_path / normalize_model_name)




if __name__=="__main__":
    main()