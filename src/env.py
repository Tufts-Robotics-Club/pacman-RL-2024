import gymnasium as gym
from gymnasium import spaces

from .pacbot import grid, GameState


class PacbotEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    _game_state: GameState

    def __init__(self, game_state=GameState()):
        super(PacbotEnv, self).__init__()
        self._game_state = game_state
        self.observation_space = spaces.Box(
            1, 18, shape=(len(grid), len(grid[0])), dtype=int
        )

        self.action_space = spaces.Discrete(4)

        self._last_score = 0
        self._last_lives = self._game_state.lives
        self._episode_rewards = []

    def _get_observation(self):
        return self._game_state.get_populated_grid()

    def step(self, action):
        self._game_state.pacbot.update_from_direction(action)
        self._game_state.next_step()

        observation = self._get_observation()

        reward = self._game_state.score - self._last_score
        self._last_score = self._game_state.score
        reward += (self._game_state.lives - self._last_lives) * 100
        self._last_lives = self._game_state.lives

        if self._game_state.dead:
            reward = -200
        if self._game_state._is_game_over():
            reward = 500

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
