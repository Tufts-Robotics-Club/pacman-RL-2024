from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import numpy as np
from stable_baselines3.common.logger import Image
from .env import PacbotEnv

check_env(PacbotEnv())


def make_env():
    env = PacbotEnv()
    env = TimeLimit(env, max_episode_steps=1000)
    env = Monitor(
        env,
        info_keywords=(
            "is_success",
            "score",
        ),
    )

    return env


num_envs = 8
env = SubprocVecEnv([make_env for _ in range(num_envs)])
env = VecFrameStack(env, n_stack=4)

LOG_DIR = "./logs/"
CHECKPOINT_DIR = "./checkpoints/"


class RecordScoreCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        scores = [info["score"] for info in self.locals["infos"]]

        self.logger.record("eval/score", np.mean(scores))

        return True


checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=CHECKPOINT_DIR)
eval_callback = EvalCallback(
    env,
    best_model_save_path=CHECKPOINT_DIR,
    log_path=CHECKPOINT_DIR,
    eval_freq=1000,
    deterministic=True,
    render=False,
)
score_callback = RecordScoreCallback()


def linear_schedule(initial_value: float):
    def func(progress_remaining):
        return progress_remaining * initial_value

    return func


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=linear_schedule(3e-3),
)

model.learn(
    total_timesteps=1e9,
    callback=[checkpoint_callback, eval_callback, score_callback],
)

model.save("model")
