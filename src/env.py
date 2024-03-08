import gymnasium as gym
from gymnasium import spaces

from .pacbot import grid, GameState
from .pacbot.variables import *


class PacbotEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    _game_state: GameState

    def __init__(self, game_state=GameState()):
        super(PacbotEnv, self).__init__()
        self._game_state = game_state
        self.observation_space = spaces.Box(
            1, 18, shape=(len(grid) * len(grid[0]),), dtype=int
        )

        self.action_space = spaces.Discrete(4)

        self._last_score = 0
        self._last_lives = self._game_state.lives
        self._episode_rewards = []

    def _closest_angry_ghost(self, position):
        if not self._game_state.pacbot.is_valid_position(position):
            return None

        queue = [position]
        visited = set()
        while queue:
            x, y = queue.pop(0)
            if (
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
            ):
                return (x, y)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_x, new_y = x + dx, y + dy
                if (
                    self._game_state.pacbot.is_valid_position((new_x, new_y))
                    and (new_x, new_y) not in visited
                ):
                    queue.append((new_x, new_y))
                    visited.add((new_x, new_y))
        return None

    def _closest_angry_ghost_distance(self, position):
        ghost_position = self._closest_angry_ghost(position)

        if ghost_position is None:
            return None

        return abs(ghost_position[0] - position[0]) + abs(
            ghost_position[1] - position[1]
        )

    def _closest_frightened_ghost(self, position):
        if not self._game_state.pacbot.is_valid_position(position):
            return None

        queue = [position]
        visited = set()
        while queue:
            x, y = queue.pop(0)
            if (
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
            ):
                return (x, y)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_x, new_y = x + dx, y + dy
                if (
                    self._game_state.pacbot.is_valid_position((new_x, new_y))
                    and (new_x, new_y) not in visited
                ):
                    queue.append((new_x, new_y))
                    visited.add((new_x, new_y))
        return None

    def _closest_frightened_ghost_distance(self, position):
        ghost_position = self._closest_frightened_ghost(position)

        if ghost_position is None:
            return None

        return abs(ghost_position[0] - position[0]) + abs(
            ghost_position[1] - position[1]
        )

    def _closest_pellet(self, position):
        if not self._game_state.pacbot.is_valid_position(position):
            return None

        queue = [position]
        visited = set()
        while queue:
            x, y = queue.pop(0)
            if self._game_state.grid[x][y] == o:
                return (x, y)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_x, new_y = x + dx, y + dy
                if (
                    self._game_state.pacbot.is_valid_position((new_x, new_y))
                    and (new_x, new_y) not in visited
                ):
                    queue.append((new_x, new_y))
                    visited.add((new_x, new_y))
        return None

    def _closest_pellet_distance(self, position):
        pellet_position = self._closest_pellet(position)

        if pellet_position is None:
            return None

        return abs(pellet_position[0] - position[0]) + abs(
            pellet_position[1] - position[1]
        )

    def _get_observation(self):
        return self._game_state.get_populated_grid().flatten()

    def step(self, action):
        self._game_state.pacbot.update_from_direction(action)
        self._game_state.next_step()

        observation = self._get_observation()

        reward = -5

        if self._game_state._is_game_over():
            reward += 50
        if self._game_state.lost_life:
            reward -= 250
        if self._game_state.ate_ghost:
            reward += 40
        if self._game_state.ate_pellet:
            reward += 25
        if self._game_state.ate_power_pellet:
            reward += 10
        if self._game_state.ate_cherry:
            reward += 200
        if self._game_state.pacbot.stuck:
            reward -= 20
        if self._game_state.pacbot.reversed:
            reward -= 5
        if self._game_state.dead:
            reward -= 500

        closest_pellet_distance = self._closest_pellet_distance(
            self._game_state.pacbot.pos
        )

        if closest_pellet_distance is not None:
            reward -= min(closest_pellet_distance, 10)

        closest_angry_ghost_distance = self._closest_angry_ghost_distance(
            self._game_state.pacbot.pos
        )

        if closest_angry_ghost_distance is not None:
            reward += min(closest_angry_ghost_distance, 10)

        closest_frightened_ghost_distance = self._closest_frightened_ghost_distance(
            self._game_state.pacbot.pos
        )

        if closest_frightened_ghost_distance is not None:
            reward -= min(closest_frightened_ghost_distance, 10)

        self._episode_rewards.append(reward)
        done = self._game_state.done
        info = {"score": self._game_state.score, "grid": str(self._game_state)}

        return observation, reward, done, False, info

    def reset(self, seed=None, return_info=True, options=None):
        super().reset(seed=seed)
        self._game_state.restart()
        self._game_state.unpause()
        self._last_score = self._game_state.score
        self._last_lives = self._game_state.lives
        self._episode_rewards = []
        observation = self._get_observation()
        info = {
            "score": self._game_state.score,
        }
        return (observation, info) if return_info else observation

    def episode_rewards(self):
        return (None, self._episode_rewards)

    def render(self, mode="human"):
        print(self._game_state)


from gymnasium.envs.registration import register

register(
    id="Pacbot-v0",
    entry_point="src.env:PacbotEnv",
    max_episode_steps=None,
)
