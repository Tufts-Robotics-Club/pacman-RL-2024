from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from src.env import PacbotEnv
from stable_baselines3.common.env_checker import check_env
import cv2
import numpy as np

# from src.pacbot.variables import right, left, up, down

pac_env = PacbotEnv()

check_env(pac_env)

env = DummyVecEnv([lambda: pac_env])


model = PPO.load("./checkpoints/best_model")


ACTIONS = ["RIGHT", "LEFT", "UP", "DOWN"]

done = True
reward = None
total_reward = 0

# in_action = right

while True:
    if done:
        state = env.reset()
        total_reward = 0

    # key = cv2.waitKey(2000)
    # if key == ord("s"):
    #     in_action = down
    # elif key == ord("w"):
    #     in_action = up
    # elif key == ord("a"):
    #     in_action = left
    # elif key == ord("d"):
    #     in_action = right

    # action = np.array([[in_action]])

    action = model.predict(state)
    state, reward, done, info = env.step(action)
    grid = np.array(info[0]["grid"], dtype=np.uint8)
    grid = np.flip(grid, axis=1)
    grid[27][0] = (255, 0, 0)
    grid[27][30] = (0, 255, 0)
    grid[0][30] = (0, 0, 255)
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
    total_reward += reward[0]
    # print("reward", reward[0])
    # print("total reward", total_reward)
    # print(info[0]["reward_components"])
    if cv2.waitKey(50) & 0xFF == ord("q"):
        break
