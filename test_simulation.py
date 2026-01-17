from src.env import PacbotEnv
import cv2
import numpy as np

env = PacbotEnv()
obs, info = env.reset()

while True:
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)

    # Visualize
    grid = np.array(info["grid"], dtype=np.uint8)
    grid = np.flip(grid, axis=1)
    grid = np.transpose(grid, (1, 0, 2))
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(grid, (800, 800), interpolation=cv2.INTER_NEAREST_EXACT)
    cv2.imshow("Pacman", resized)

    if done:
        obs, info = env.reset()

    if cv2.waitKey(50) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()