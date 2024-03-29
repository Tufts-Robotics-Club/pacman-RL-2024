from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from src.env import PacbotEnv
from stable_baselines3.common.env_checker import check_env
import cv2
import numpy as np

env = PacbotEnv()

check_env(env)

env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)


model = PPO.load("./checkpoints/best_model")


ACTIONS = ["RIGHT", "LEFT", "UP", "DOWN"]

done = True
reward = None

while True:
    if done:
        state = env.reset()
    action = model.predict(state)
    state, reward, done, info = env.step(action)
    grid = np.array(info[0]["grid"], dtype=np.uint8)
    grid = np.transpose(grid, (1, 0, 2))
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(
        grid, (1600, 1600), interpolation=cv2.INTER_NEAREST_EXACT
    )
    cv2.putText(
        resized_image,
        f"Reward: {reward[0]}     Action: {ACTIONS[action[0][0]]}",
        (10, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("grid", resized_image)
    print("reward", reward[0])
    print(info[0]["reward_components"])
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break
