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


class FeatureExtractor:
    """
    Extracts the 22-dimensional observation vector from a PacBot GameState.
    """
    _game_state: GameState

    def __init__(self, game_state: GameState):
        self.temp_goal = None
        self.temp_goal_steps = 0
        self._game_state = game_state

    def reset(self):
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

    def extract(self, game_state: GameState) -> np.ndarray:
        """
        Extract features from the game state into a 22 length to be passed into
        the ML algo. The vector indicies are as follows:
        -0: level_progress | ratio of pellets eaten so far (1-remaining/total)
        -1: power_pellet_duration | time remaining on power pellet (frightened_counter / frightened_length)
        -2-5: Pellet Proximity | BFS shortest-path distance to the nearest pellet if Pac-Man steps in that direction. If a region has no pellets left, it falls back to the distance to the opposite corner of the board.
        -6-9: Threat/Intersection Margin | (angry_ghost_dist - intersection_dist) in each direction. Measures whether an angry ghost is closing in before Pac-Man can escape through a junction.
        -10-13: Edible Ghost Proximity | BFS distance to the nearest frightened ghost in each direction (incentivizes hunting blue ghosts).
        -14-17: Entrapment/Escape Space | Relative number of safe tiles reachable before ghosts intercept Pac-Man. Uses a simultaneous flood-fill to detect if a direction leads into a dead-end trap.
        -18-21: Current Direction: One-hot encoded vector representing the direction Pac-Man is currently heading.
        """
        self._game_state = game_state

        #-----Level Progress-----#

        level_progress = 1 - (self._game_state.pellets / self._game_state.total_pellets)

        #-----Power Pellet Duration-----#

        power_pellet_duration = self._game_state.frightened_counter / frightened_length

        #-----Pellet Proximity-----#

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

            # when no pellets are found in any direction, set the 'closest pellet' location to the opposite corner 
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

        #-----Threat/Intersection Margin-----#

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

        #-----Edible Ghost Proximity-----#

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

        #-----Entrapment/Escape Space-----#

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

        #-----Current Direction-----#   
             
        is_direction_left = 1 if self._game_state.pacbot.direction == left else 0
        is_direction_right = 1 if self._game_state.pacbot.direction == right else 0
        is_direction_up = 1 if self._game_state.pacbot.direction == up else 0
        is_direction_down = 1 if self._game_state.pacbot.direction == down else 0

        return np.array(
            list(
                map(
                    normalize, # normalize everything to ensure it's in the range [0, 1]
                    [
                        level_progress,
                        power_pellet_duration,
                        closest_pellet_left_distance,
                        closest_pellet_right_distance,
                        closest_pellet_up_distance,
                        closest_pellet_down_distance,
                        closest_angry_ghost_left_distance - closest_intersection_left_distance,
                        closest_angry_ghost_right_distance - closest_intersection_right_distance,
                        closest_angry_ghost_up_distance - closest_intersection_up_distance,
                        closest_angry_ghost_down_distance - closest_intersection_down_distance,
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
