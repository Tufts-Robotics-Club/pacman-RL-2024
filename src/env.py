import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .pacbot import grid, GameState
from .pacbot.variables import *

MAX_DISTANCE = 64


def normalize(x):
    return 0 if x < 0 else (1 if x > 1 else x)


class PacbotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    _game_state: GameState

    def __init__(self, game_state=GameState()):
        super(PacbotEnv, self).__init__()
        self._game_state = game_state
        self.observation_space = spaces.Box(0, 1, shape=(18,), dtype=np.float64)
        self.step_count = 1e6

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

    def _closest_intersection_predicate(self, x, y):
        return (
            (self._game_state.grid[x - 1][y] != I)
            + (self._game_state.grid[x + 1][y] != I)
            + (self._game_state.grid[x][y - 1] != I)
            + (self._game_state.grid[x][y + 1] != I)
        ) > 2

    def _find_closest(self, position, predicate, default=MAX_DISTANCE):
        if not self._game_state.pacbot.is_valid_position(position):
            return default

        queue = [position]
        visited = np.array([-1] * len(grid) * len(grid[0]))

        def linear_index(x, y):
            return y * len(grid) + x

        visited[linear_index(position[0], position[1])] = 0

        while queue:
            x, y = queue.pop(0)
            if predicate(x, y):
                return visited[linear_index(x, y)]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_x, new_y = x + dx, y + dy
                if (
                    self._game_state.pacbot.is_valid_position((new_x, new_y))
                    and visited[linear_index(new_x, new_y)] == -1
                ):
                    queue.append((new_x, new_y))
                    visited[linear_index(new_x, new_y)] = (
                        visited[linear_index(x, y)] + 1
                    )
        return default

    def _get_observation(self):
        level_progress = 1 - (self._game_state.pellets / self._game_state.total_pellets)

        power_pellet_duration = self._game_state.frightened_counter / frightened_length

        pos = self._game_state.pacbot.pos
        pos_left = (pos[0] - 1, pos[1])
        pos_right = (pos[0] + 1, pos[1])
        pos_up = (pos[0], pos[1] + 1)
        pos_down = (pos[0], pos[1] - 1)

        closest_pellet_left_distance = (
            self._find_closest(pos_left, self._closest_pellet_predicate) / MAX_DISTANCE
        )
        closest_pellet_right_distance = (
            self._find_closest(pos_right, self._closest_pellet_predicate) / MAX_DISTANCE
        )
        closest_pellet_up_distance = (
            self._find_closest(pos_up, self._closest_pellet_predicate) / MAX_DISTANCE
        )
        closest_pellet_down_distance = (
            self._find_closest(pos_down, self._closest_pellet_predicate) / MAX_DISTANCE
        )

        closest_angry_ghost_left_distance = (
            self._find_closest(pos_left, self._closest_angry_ghost_predicate, default=0)
            / MAX_DISTANCE
        )
        closest_angry_ghost_right_distance = (
            self._find_closest(
                pos_right, self._closest_angry_ghost_predicate, default=0
            )
            / MAX_DISTANCE
        )
        closest_angry_ghost_up_distance = (
            self._find_closest(pos_up, self._closest_angry_ghost_predicate, default=0)
            / MAX_DISTANCE
        )
        closest_angry_ghost_down_distance = (
            self._find_closest(pos_down, self._closest_angry_ghost_predicate, default=0)
            / MAX_DISTANCE
        )

        closest_frightened_ghost_left_distance = (
            self._find_closest(pos_left, self._closest_frightened_ghost_predicate)
            / MAX_DISTANCE
        )
        closest_frightened_ghost_right_distance = (
            self._find_closest(pos_right, self._closest_frightened_ghost_predicate)
            / MAX_DISTANCE
        )
        closest_frightened_ghost_up_distance = (
            self._find_closest(pos_up, self._closest_frightened_ghost_predicate)
            / MAX_DISTANCE
        )
        closest_frightened_ghost_down_distance = (
            self._find_closest(pos_down, self._closest_frightened_ghost_predicate)
            / MAX_DISTANCE
        )

        closest_intersection_left_distance = (
            self._find_closest(pos_left, self._closest_intersection_predicate)
            / MAX_DISTANCE
        )
        closest_intersection_right_distance = (
            self._find_closest(pos_right, self._closest_intersection_predicate)
            / MAX_DISTANCE
        )
        closest_intersection_up_distance = (
            self._find_closest(pos_up, self._closest_intersection_predicate)
            / MAX_DISTANCE
        )
        closest_intersection_down_distance = (
            self._find_closest(pos_down, self._closest_intersection_predicate)
            / MAX_DISTANCE
        )

        is_direction_left = 1 if self._game_state.pacbot.direction == left else 0
        is_direction_right = 1 if self._game_state.pacbot.direction == right else 0
        is_direction_up = 1 if self._game_state.pacbot.direction == up else 0
        is_direction_down = 1 if self._game_state.pacbot.direction == down else 0

        return np.array(
            list(
                map(
                    normalize,
                    [
                        level_progress,
                        power_pellet_duration,
                        closest_pellet_left_distance,
                        closest_pellet_right_distance,
                        closest_pellet_up_distance,
                        closest_pellet_down_distance,
                        closest_angry_ghost_left_distance
                        - closest_intersection_left_distance,
                        closest_angry_ghost_right_distance
                        - closest_intersection_right_distance,
                        closest_angry_ghost_up_distance
                        - closest_intersection_up_distance,
                        closest_angry_ghost_down_distance
                        - closest_intersection_down_distance,
                        closest_frightened_ghost_left_distance,
                        closest_frightened_ghost_right_distance,
                        closest_frightened_ghost_up_distance,
                        closest_frightened_ghost_down_distance,
                        is_direction_left,
                        is_direction_right,
                        is_direction_up,
                        is_direction_down,
                    ],
                )
            )
        )

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

        closest_angry_ghost_distance = (
            self._find_closest(
                self._game_state.pacbot.pos, self._closest_angry_ghost_predicate
            )
            if self._game_state.ghosts_enabled
            else 0
        )

        closest_frightened_ghost_distance = (
            self._find_closest(
                self._game_state.pacbot.pos, self._closest_frightened_ghost_predicate
            )
            if self._game_state.ghosts_enabled
            else 0
        )

        reward_components = {
            # "exist": -0.5,
            "win": 50 * self._game_state._is_game_over(),
            "lost_life": -50 * self._game_state.lost_life,
            "ate_ghost": 25 * self._game_state.ate_ghost,
            "ate_pellet": 15 * self._game_state.ate_pellet,
            "ate_power_pellet": 10 * self._game_state.ate_power_pellet,
            "ate_cherry": 50 * self._game_state.ate_cherry,
            "exploration": 1 * self._game_state.pacbot.new_pos,
            # "changed": -5 * self._game_state.pacbot.changed,
            # "reversed": -2 * self._game_state.pacbot.reversed,
            "dead": -25 * self._game_state.dead,
            "inaction": -0.1 * (self._game_state.pacbot.stuck > 5),
            # "closest_pellet_distance": (
            #     -min(closest_pellet_distance, 5)
            #     if not closest_pellet_distance in [None, 0]
            #     else 0
            # )
            # / 10,
            # "closest_angry_ghost_distance": (
            #     min(closest_angry_ghost_distance, 5)
            #     if not closest_angry_ghost_distance in [None, 0]
            #     else 0
            # )
            # / 10,
            # "closest_frightened_ghost_distance": (
            #     -min(closest_frightened_ghost_distance, 5)
            #     if not closest_frightened_ghost_distance in [None, 0]
            #     else 0
            # )
            # / 10,
        }

        reward = sum(reward_components.values())

        return reward, reward_components

    def step(self, action):
        self.step_count += 1

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
        self._game_state.lives = 3
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
