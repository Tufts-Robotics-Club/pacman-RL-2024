import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .pacbot import GameState
from .feature_extractor import FeatureExtractor


class PacbotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    _game_state: GameState

    def __init__(self, game_state=GameState()):
        super(PacbotEnv, self).__init__()
        self._game_state = game_state
        self.feature_extractor = FeatureExtractor(self._game_state)
        self.observation_space = spaces.Box(0, 1, shape=(22,), dtype=np.float64)
        self.action_space = spaces.Discrete(4)
        self.step_count = 1e6

    def _get_observation(self):
        return self.feature_extractor.extract(self._game_state)

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
        self.feature_extractor.reset()
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
