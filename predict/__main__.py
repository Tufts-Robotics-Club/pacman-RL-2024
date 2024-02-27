from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from src.env import PacbotEnv
from time import sleep

env = PacbotEnv()

env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)


model = PPO.load("./checkpoints/best_model")


done = True
reward = None

while True:
    if done:
        state = env.reset()
        # state, reward, done, _, info = env.step(env.action_space.sample())
    action = model.predict(state)
    state, reward, done, info = env.step(action)
    print(info[0]["grid"])
    print("reward", reward)
    sleep(0.1)
