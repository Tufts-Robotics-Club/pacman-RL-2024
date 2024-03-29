import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .pacbot import grid, GameState
from .pacbot.variables import *


class PacbotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    _game_state: GameState

    def __init__(self, game_state=GameState()):
        super(PacbotEnv, self).__init__()
        self._game_state = game_state
        self.observation_space = spaces.Box(
            1, 18, shape=(len(grid) * len(grid[0]),), dtype=int
        )

        self.action_space = spaces.Discrete(4)

    def _closest_pellet_predicate(self, x, y):
        return self._game_state.grid[x][y] == o

    def _closest_frightened_ghost_predicate(self, x, y):
        return (
            (
                self._game_state.red.is_frightened()
                and ((x, y) == self._game_state.red.pos["current"])
            )
            or (
                self._game_state.pink.is_frightened()
                and ((x, y) == self._game_state.pink.pos["current"])
            )
            or (
                self._game_state.orange.is_frightened()
                and ((x, y) == self._game_state.orange.pos["current"])
            )
            or (
                self._game_state.blue.is_frightened()
                and ((x, y) == self._game_state.blue.pos["current"])
            )
        )

    def _closest_angry_ghost_predicate(self, x, y):
        return (
            (
                not self._game_state.red.is_frightened()
                and ((x, y) == self._game_state.red.pos["current"])
            )
            or (
                not self._game_state.pink.is_frightened()
                and ((x, y) == self._game_state.pink.pos["current"])
            )
            or (
                not self._game_state.orange.is_frightened()
                and ((x, y) == self._game_state.orange.pos["current"])
            )
            or (
                not self._game_state.blue.is_frightened()
                and ((x, y) == self._game_state.blue.pos["current"])
            )
        )

    def _find_closest(self, position, predicate):
        if not self._game_state.pacbot.is_valid_position(position):
            return None

        queue = [position]
        visited = np.array([0] * len(grid) * len(grid[0]))

        def linear_index(x, y):
            return y * len(grid) + x

        while queue:
            x, y = queue.pop(0)
            if predicate(x, y):
                return visited[linear_index(x, y)]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_x, new_y = x + dx, y + dy
                if (
                    self._game_state.pacbot.is_valid_position((new_x, new_y))
                    and visited[linear_index(new_x, new_y)] == 0
                ):
                    queue.append((new_x, new_y))
                    visited[linear_index(new_x, new_y)] = (
                        visited[linear_index(x, y)] + 1
                    )
        return None

    def _get_observation(self):
        return self._game_state.get_populated_grid().flatten()

    def _get_info(self):
        return {
            "episode": {
                "r": self._episode_rewards,
                "l": self._episode_length,
            },
            "score": self._game_state.score,
            "is_success": self._game_state._is_game_over(),
            "grid": self.rgb_array(),
        }

    def _get_reward(self):
        closest_pellet_distance = self._find_closest(
            self._game_state.pacbot.pos, self._closest_pellet_predicate
        )

        closest_angry_ghost_distance = self._find_closest(
            self._game_state.pacbot.pos, self._closest_angry_ghost_predicate
        )

        closest_frightened_ghost_distance = self._find_closest(
            self._game_state.pacbot.pos, self._closest_frightened_ghost_predicate
        )

        reward_components = {
            "exist": -5,
            "win": 50 * self._game_state._is_game_over(),
            "lost_life": -250 * self._game_state.lost_life,
            "ate_ghost": 40 * self._game_state.ate_ghost,
            "ate_pellet": 25 * self._game_state.ate_pellet,
            "ate_power_pellet": 10 * self._game_state.ate_power_pellet,
            "ate_cherry": 200 * self._game_state.ate_cherry,
            "stuck": -5 * self._game_state.pacbot.stuck,
            "reversed": -5 * self._game_state.pacbot.reversed,
            "dead": -500 * self._game_state.dead,
            "closest_pellet_distance": (
                -min(closest_pellet_distance, 10)
                if not closest_pellet_distance in [None, 0]
                else 0
            ),
            "closest_angry_ghost_distance": (
                min(closest_angry_ghost_distance, 10)
                if not closest_angry_ghost_distance in [None, 0]
                else 0
            ),
            "closest_frightened_ghost_distance": (
                -min(closest_frightened_ghost_distance, 10)
                if not closest_frightened_ghost_distance in [None, 0]
                else 0
            ),
        }

        reward = float(sum(reward_components.values()))

        return reward, reward_components

    def step(self, action):
        self._game_state.pacbot.update_from_direction(action)
        self._game_state.next_step()

        observation = self._get_observation()
        reward, reward_components = self._get_reward()

        self._episode_rewards += reward
        self._episode_length += 1
        done = self._game_state.done
        info = self._get_info()
        info["reward_components"] = reward_components

        return observation, reward, done, False, info

    def reset(self, seed=None, return_info=True, options=None):
        super().reset(seed=seed)
        self._game_state.restart()
        self._game_state.unpause()
        self._last_score = self._game_state.score
        self._last_lives = self._game_state.lives
        self._episode_rewards = 0
        self._episode_length = 0
        observation = self._get_observation()
        info = self._get_info()
        info["reward_components"] = {}
        return (observation, info) if return_info else observation

    def episode_rewards(self):
        return (None, self._episode_rewards)

    def rgb_array(self):
        return self._game_state.rgb_array()

    def render(self, mode="human"):
        if mode == "human":
            print(self._game_state)
        elif mode == "rgb_array":
            from matplotlib import pyplot as plt

            image = self.rgb_array()
            plt.imshow(image)
        else:
            raise NotImplementedError()


from gymnasium.envs.registration import register

register(
    id="Pacbot-v0",
    entry_point="src.env:PacbotEnv",
    max_episode_steps=None,
)
