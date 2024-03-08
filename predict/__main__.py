from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from src.env import PacbotEnv
from stable_baselines3.common.env_checker import check_env
from time import sleep

env = PacbotEnv()

check_env(env)

env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)


model = PPO.load("./checkpoints/best_model")


done = True
reward = None

while True:
    if done:
        state = env.reset()
    action = model.predict(state)
    state, reward, done, info = env.step(action)
    print(info[0]["grid"])
    print("reward", reward)
    sleep(0.1)
