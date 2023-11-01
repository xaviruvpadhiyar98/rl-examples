import gymnasium as gym
from gymnasium import spaces
import numpy as np
from data.actions import correct_actions
from data.ob_space import obs
np.random.seed(123)


class PatternMatchingEnv(gym.Env):
    """
    Custom environment for OpenAI Gym with discrete actions and a 3x3 matrix observation space.
    """

    def __init__(self):
        super(PatternMatchingEnv, self).__init__()

        # Define a discrete action space ranging from 0 to 2 (A, B, C)
        self.action_space = spaces.Discrete(3)

        # Define an observation space, a 3x3 matrix filled with random values
        low = min(obs) - 100
        high = max(obs) + 100
        self.observation_space = spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)

        # Define a list of length 400 with random distribution of 'A', 'B', 'C'
        self.step_sequence = correct_actions

        # self.all_states = [
        #     np.random.random((3, 3)) for _ in range(len(self.step_sequence))
        # ]
        self.all_states = obs
        self.action_map = {0: "BUY", 1: "HOLD", 2: "SELL"}

        # Current step in the environment
        self.current_step = 0
        self.highest_progress = 0
        self.correct_streak = 0
        self.took_correct_action = 0
        self.took_incorrect_actions = 0
        self.previous_action = None
        self.consecutive_mistakes = 0
        self.base_penalty = 0.1  # Base penalty for an incorrect action
        self.max_penalty = 1.0  # Maximum penalty limit

        # Initial state
        self.state = None

    def step(self, action):
        action_char = self.action_map[action]

        reward = 0.0
        done = False
        truncation = False

        step_weight = 1 + (self.current_step / (len(self.step_sequence) - 1))

        if action_char == self.step_sequence[self.current_step]:
            self.correct_streak += 1
            reward = step_weight * (2.0 + self.correct_streak)  
            self.took_correct_action += 1
        else:
            self.took_incorrect_actions += 1
            reward = -20 * step_weight
            self.correct_streak = 0

        self.current_step += 1
        self.state = self.all_states[self.current_step]

        done = self.current_step >= len(self.step_sequence) - 1
        if done:
            reward += 5 * (self.took_correct_action / max(1, self.current_step))

        info = {
            "correct_action": self.took_correct_action,
            "incorrect_actions": self.took_incorrect_actions,
            "correct_streak": self.correct_streak,
        }
        return self.state, reward, done, truncation, info


    def reset(self, seed=None, options=None):
        # Reset the environment state
        self.current_step = 0
        self.state = self.all_states[self.current_step]
        return self.state, {}

    def close(self):
        pass


def objective(trial):
    model_name = "a2c"
    device = "auto"
    if model_name == "a2c":
        device = "cpu"

    env = PatternMatchingEnv
    eval_env = PatternMatchingEnv

    num_envs = 16
    eval_envs = 16
    timestamp = 500_000

    vec_env = make_vec_env(env, n_envs=num_envs)
    eval_vec_env = make_vec_env(eval_env, n_envs=eval_envs)

    hp = HYPERPARAMS_SAMPLER[model_name](trial)
    hp.update({"env": vec_env, "device": device, "policy": "MlpPolicy"})

    model = {"ppo": PPO, "dqn": DQN, "a2c": A2C}[model_name](**hp)
    model.learn(total_timesteps=timestamp, progress_bar=True)

    counter = 0
    correct_actions = []
    obs = eval_vec_env.reset()
    while counter < eval_envs:
        action, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = eval_vec_env.step(action)
        for i in range(eval_envs):
            if dones[i]:
                result = {
                    "Model": model_name,
                    "Env": i,
                    "Timestamp": timestamp,
                    "Correct Actions": infos[i]["correct_action"],
                    "Incorrect Actions": infos[i]["incorrect_actions"],
                    "Correct Streak": infos[i]["correct_streak"],
                    "Ending Reward": infos[i]["episode"]["r"],
                }
                correct_actions.append(infos[i]["correct_action"])
                print(json.dumps(result))
                counter += 1
    correct_actions.sort(reverse=True)
    return correct_actions[0]


if __name__ == "__main__":
    import json
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from optuna import Trial, create_study
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler
    from hyperparams_opt import HYPERPARAMS_SAMPLER

    N_STARTUP_TRIALS = 100
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

    # # Initialize the environment
    # env = PatternMatchingEnv
    # eval_env = PatternMatchingEnv
    # num_envs = 16
    # eval_envs = 2
    # model_name = "dqn"
    # timestamp = 20_000_000
    # # Vectorize environment for PPO
    # vec_env = make_vec_env(env, n_envs=num_envs)
    # eval_vec_env = make_vec_env(eval_env, n_envs=eval_envs)

    # model = {
    #     "ppo": PPO("MlpPolicy", vec_env),
    #     "dqn": DQN("MlpPolicy", vec_env),
    #     "a2c": A2C("MlpPolicy", vec_env),
    # }[model_name]

    # # Train the model
    # model.learn(total_timesteps=timestamp, progress_bar=True)

    # # Test the model
    # # print(evaluate_policy(model, eval_vec_env, deterministic=True))
    # counter = 0
    # obs = eval_vec_env.reset()
    # while counter < eval_envs:
    #     action, _ = model.predict(obs, deterministic=False)
    #     obs, rewards, dones, infos = eval_vec_env.step(action)
    #     for i in range(eval_envs):
    #         if dones[i]:
    #             result = {
    #                 "Model": model_name,
    #                 "Env": i,
    #                 "Timestamp": timestamp,
    #                 "Correct Actions": infos[i]['correct_action'],
    #                 "Incorrect Actions": infos[i]['incorrect_actions'],
    #                 "Correct Streak": infos[i]['correct_streak'],
    #                 "Ending Reward": infos[i]['episode']['r'],
    #             }
    #             print(json.dumps(result))
    #             counter += 1
