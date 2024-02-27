from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from .env import PacbotEnv

print(PacbotEnv)


def make_env():
    def _init():
        return PacbotEnv()

    return _init


num_envs = 12
env = SubprocVecEnv([make_env() for _ in range(num_envs)])
env = VecFrameStack(env, n_stack=4)

LOG_DIR = "./logs/"
CHECKPOINT_DIR = "./checkpoints/"

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=CHECKPOINT_DIR)
eval_callback = EvalCallback(
    env,
    best_model_save_path=CHECKPOINT_DIR,
    log_path=CHECKPOINT_DIR,
    eval_freq=500,
    deterministic=True,
    render=False,
)


def linear_schedule(initial_value: float):
    def func(progress_remaining):
        return progress_remaining * initial_value

    return func


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=linear_schedule(3e-4),
    n_steps=1024,
    batch_size=256,
    gamma=0.99,
    clip_range=0.09,
    ent_coef=0.05,
    vf_coef=0.5,
)

model.learn(
    total_timesteps=100000000,
    callback=[checkpoint_callback, eval_callback],
)

model.save("model")
