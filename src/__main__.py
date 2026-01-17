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
import torch
from .env import PacbotEnv


def make_env():
    def _init():
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
    return _init


LOG_DIR = "./logs/"
CHECKPOINT_DIR = "./checkpoints/"


class RecordScoreCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        scores = [info["score"] for info in self.locals["infos"]]
        self.logger.record("eval/score", np.mean(scores))
        return True


def linear_schedule(initial_value: float):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func


def main():
    # Check CUDA availability
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")

    # Validate environment
    check_env(PacbotEnv())

    # More parallel envs for better GPU utilization
    num_envs = 64
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    # Create evaluation environment
    eval_env = SubprocVecEnv([make_env() for _ in range(8)])

    checkpoint_callback = CheckpointCallback(
        save_freq=100000 // num_envs,  # Adjusted for num_envs
        save_path=CHECKPOINT_DIR,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=CHECKPOINT_DIR,
        log_path=CHECKPOINT_DIR,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    score_callback = RecordScoreCallback()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=linear_schedule(3e-4),
        ent_coef=0.05,
        device=device,
        # Larger batches for better GPU utilization
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
    )

    print(f"Training on device: {model.device}")

    model.learn(
        total_timesteps=1e8,
        callback=[checkpoint_callback, eval_callback, score_callback],
    )

    model.save("model")


# Required for Windows multiprocessing with SubprocVecEnv
if __name__ == "__main__":
    main()
