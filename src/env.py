import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .pacbot import grid, GameState
from .pacbot.variables import *

MAX_DISTANCE = 64


def normalize(x):
    return 0 if x < 0 else (1 if x > 1 else x)


def get_corner(x, y):
    if x < len(grid) / 2:
        return 0 if y < len(grid[0]) / 2 else 3
    return 1 if y < len(grid[0]) / 2 else 2


def opposite_corner(corner):
    return (corner + 2) % 4


def corner_position(corner):
    if corner == 0:
        return (1, 1)
    if corner == 1:
        return (len(grid) - 2, 1)
    if corner == 2:
        return (len(grid) - 2, len(grid[0]) - 2)
    return (1, len(grid[0]) - 2)


def linear_index(x, y):
    return y * len(grid) + x


def delinear_index(i):
    return (i % len(grid), i // len(grid))


class PacbotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    _game_state: GameState

    def __init__(self, game_state=GameState()):
        super(PacbotEnv, self).__init__()
        self._game_state = game_state
        self.observation_space = spaces.Box(0, 1, shape=(22,), dtype=np.float64)
        self.action_space = spaces.Discrete(4)
        self.step_count = 1e6
        self.temp_goal = None
        self.temp_goal_steps = 0

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

    def _is_predicate(self, x, y):
        return lambda _x, _y: (_x, _y) == (x, y)

    def _closest_intersection_predicate(self, x, y):
        return (
            (self._game_state.grid[x - 1][y] != I)
            + (self._game_state.grid[x + 1][y] != I)
            + (self._game_state.grid[x][y - 1] != I)
            + (self._game_state.grid[x][y + 1] != I)
        ) > 2

    def _find_closest(self, position, predicate, origin=None, default=MAX_DISTANCE):
        if not self._game_state.pacbot.is_valid_position(position):
            return default

        queue = [position]
        visited = np.array([-1] * len(grid) * len(grid[0]))

        visited[linear_index(position[0], position[1])] = 0
        if origin is not None:
            visited[linear_index(origin[0], origin[1])] = 0

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
                    visited[linear_index(new_x, new_y)] = (
                        visited[linear_index(x, y)] + 1
                    )
                    if visited[linear_index(new_x, new_y)] < MAX_DISTANCE:
                        queue.append((new_x, new_y))
        return default

    def _ghosts_flood_fill(self):
        visited = np.array([-1] * len(grid) * len(grid[0]))

        queue = [
            self._game_state.red.pos["current"],
            self._game_state.pink.pos["current"],
            self._game_state.orange.pos["current"],
            self._game_state.blue.pos["current"],
        ]

        for ghost in queue:
            visited[linear_index(*ghost)] = 0

        while queue:
            x, y = queue.pop(0)
            steps = visited[linear_index(x, y)]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_x, new_y = x + dx, y + dy
                if (
                    self._game_state.pacbot.is_valid_position((new_x, new_y))
                    and visited[linear_index(new_x, new_y)] == -1
                ):
                    visited[linear_index(new_x, new_y)] = steps + 1
                    queue.append((new_x, new_y))

        return visited

    def _safe_tiles(self, position, origin=None):
        if not self._game_state.pacbot.is_valid_position(position):
            return 0

        # check if any ghosts are at the position
        if (
            (position == self._game_state.red.pos["current"])
            or (position == self._game_state.pink.pos["current"])
            or (position == self._game_state.orange.pos["current"])
            or (position == self._game_state.blue.pos["current"])
        ):
            return 0

        # bfs while flood filling ghosts
        ghost_flood_fill = self._ghosts_flood_fill()

        queue = [position]
        visited = np.array([-1] * len(grid) * len(grid[0]))
        visited[linear_index(position[0], position[1])] = 0
        if origin is not None:
            visited[linear_index(origin[0], origin[1])] = 0

        safe_tiles = 0

        while queue:
            x, y = queue.pop(0)
            steps = visited[linear_index(x, y)]
            safe_tiles += 1

            if steps > MAX_DISTANCE:
                continue

            # if pacman is closer than the ghosts at that time, it's not yet entrapped
            if steps < ghost_flood_fill[linear_index(x, y)]:
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    new_x, new_y = x + dx, y + dy
                    if (
                        self._game_state.pacbot.is_valid_position((new_x, new_y))
                        and visited[linear_index(new_x, new_y)] == -1
                    ):
                        visited[linear_index(new_x, new_y)] = steps + 1
                        queue.append((new_x, new_y))

        return safe_tiles

    def _get_observation(self):
        level_progress = 1 - (self._game_state.pellets / self._game_state.total_pellets)

        power_pellet_duration = self._game_state.frightened_counter / frightened_length

        pos = self._game_state.pacbot.pos
        pos_left = (pos[0] - 1, pos[1])
        pos_right = (pos[0] + 1, pos[1])
        pos_up = (pos[0], pos[1] + 1)
        pos_down = (pos[0], pos[1] - 1)

        closest_pellet_left_distance = normalize(
            self._find_closest(pos_left, self._closest_pellet_predicate, origin=pos)
            / MAX_DISTANCE
        )
        closest_pellet_right_distance = normalize(
            self._find_closest(pos_right, self._closest_pellet_predicate, origin=pos)
            / MAX_DISTANCE
        )
        closest_pellet_up_distance = normalize(
            self._find_closest(pos_up, self._closest_pellet_predicate, origin=pos)
            / MAX_DISTANCE
        )
        closest_pellet_down_distance = normalize(
            self._find_closest(pos_down, self._closest_pellet_predicate, origin=pos)
            / MAX_DISTANCE
        )

        self.temp_goal_steps = max(self.temp_goal_steps - 1, 0)

        if self.temp_goal_steps == 0:
            self.temp_goal = None

        if (
            closest_pellet_left_distance == 1
            and closest_pellet_right_distance == 1
            and closest_pellet_up_distance == 1
            and closest_pellet_down_distance == 1
        ):
            # no pellets found, go to the opposite corner
            if self.temp_goal is None:
                corner = get_corner(*self._game_state.pacbot.pos)
                opp_corner = opposite_corner(corner)
                to = corner_position(opp_corner)
                self.temp_goal = to
                self.temp_goal_steps = 16
            else:
                to = self.temp_goal

            closest_pellet_left_distance = self._find_closest(
                pos_left, self._is_predicate(*to), origin=pos, default=255
            )
            closest_pellet_right_distance = self._find_closest(
                pos_right, self._is_predicate(*to), origin=pos, default=255
            )
            closest_pellet_up_distance = self._find_closest(
                pos_up, self._is_predicate(*to), origin=pos, default=255
            )
            closest_pellet_down_distance = self._find_closest(
                pos_down, self._is_predicate(*to), origin=pos, default=255
            )

        closest_angry_ghost_left_distance = (
            self._find_closest(
                pos_left, self._closest_angry_ghost_predicate, origin=pos, default=0
            )
            / MAX_DISTANCE
        )
        closest_angry_ghost_right_distance = (
            self._find_closest(
                pos_right, self._closest_angry_ghost_predicate, origin=pos, default=0
            )
            / MAX_DISTANCE
        )
        closest_angry_ghost_up_distance = (
            self._find_closest(
                pos_up, self._closest_angry_ghost_predicate, origin=pos, default=0
            )
            / MAX_DISTANCE
        )
        closest_angry_ghost_down_distance = (
            self._find_closest(
                pos_down, self._closest_angry_ghost_predicate, origin=pos, default=0
            )
            / MAX_DISTANCE
        )

        closest_frightened_ghost_left_distance = (
            self._find_closest(
                pos_left, self._closest_frightened_ghost_predicate, origin=pos
            )
            / MAX_DISTANCE
        )
        closest_frightened_ghost_right_distance = (
            self._find_closest(
                pos_right, self._closest_frightened_ghost_predicate, origin=pos
            )
            / MAX_DISTANCE
        )
        closest_frightened_ghost_up_distance = (
            self._find_closest(
                pos_up, self._closest_frightened_ghost_predicate, origin=pos
            )
            / MAX_DISTANCE
        )
        closest_frightened_ghost_down_distance = (
            self._find_closest(
                pos_down, self._closest_frightened_ghost_predicate, origin=pos
            )
            / MAX_DISTANCE
        )

        closest_intersection_left_distance = (
            self._find_closest(
                pos_left, self._closest_intersection_predicate, origin=pos
            )
            / MAX_DISTANCE
        )
        closest_intersection_right_distance = (
            self._find_closest(
                pos_right, self._closest_intersection_predicate, origin=pos
            )
            / MAX_DISTANCE
        )
        closest_intersection_up_distance = (
            self._find_closest(pos_up, self._closest_intersection_predicate, origin=pos)
            / MAX_DISTANCE
        )
        closest_intersection_down_distance = (
            self._find_closest(
                pos_down, self._closest_intersection_predicate, origin=pos
            )
            / MAX_DISTANCE
        )

        safe_tiles_left = self._safe_tiles(pos_left, origin=pos)
        safe_tiles_right = self._safe_tiles(pos_right, origin=pos)
        safe_tiles_up = self._safe_tiles(pos_up, origin=pos)
        safe_tiles_down = self._safe_tiles(pos_down, origin=pos)

        min_safe_tiles = min(
            safe_tiles_left, safe_tiles_right, safe_tiles_up, safe_tiles_down
        )

        entrapment_left = (safe_tiles_left - min_safe_tiles) / MAX_DISTANCE
        entrapment_right = (safe_tiles_right - min_safe_tiles) / MAX_DISTANCE
        entrapment_up = (safe_tiles_up - min_safe_tiles) / MAX_DISTANCE
        entrapment_down = (safe_tiles_down - min_safe_tiles) / MAX_DISTANCE

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
                        entrapment_left,
                        entrapment_right,
                        entrapment_up,
                        entrapment_down,
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
        reward_components = {
            "exist": 1,
            "win": 50 * self._game_state._is_game_over(),
            "lost_life": -100 * self._game_state.lost_life,
            "ate_ghost": 20 * self._game_state.ate_ghost,
            "ate_pellet": 12 * self._game_state.ate_pellet,
            "ate_power_pellet": 10 * self._game_state.ate_power_pellet,
            "ate_cherry": 50 * self._game_state.ate_cherry,
            # "exploration": 1 * self._game_state.pacbot.new_pos,
            "changed": -0.5 * self._game_state.pacbot.changed,
            # "reversed": -2 * self._game_state.pacbot.reversed,
            "dead": -150 * self._game_state.dead,
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
