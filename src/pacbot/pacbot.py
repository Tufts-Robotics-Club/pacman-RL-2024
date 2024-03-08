from .variables import *
from .grid import grid


class PacBot:
    """
    Allows initializing and updating information about PacBot
    """

    def __init__(self, game_state):
        self._game_state = game_state
        self.respawn()

    def respawn(self):
        self.last_pos = None
        self.pos = pacbot_starting_pos
        self.direction = pacbot_starting_dir
        self.stuck = False
        self.reversed = False

    def update(self, position):
        if position[0] > self.pos[0]:
            self.direction = right
        elif position[0] < self.pos[0]:
            self.direction = left
        elif position[1] > self.pos[1]:
            self.direction = up
        elif position[1] < self.pos[1]:
            self.direction = down
        self.pos = position

    def is_valid_position(self, position):
        if (
            position[0] < 0
            or position[0] >= len(grid)
            or position[1] < 0
            or position[1] >= len(grid[0])
        ):
            return False
        return self._game_state.grid[position[0]][position[1]] != I

    def update_from_direction(self, direction):
        if self._game_state.update_ticks % ticks_per_update != 0:
            return

        self.direction = direction
        if direction == right:
            next_pos = (self.pos[0] + 1, self.pos[1])
        elif direction == left:
            next_pos = (self.pos[0] - 1, self.pos[1])
        elif direction == up:
            next_pos = (self.pos[0], self.pos[1] + 1)
        elif direction == down:
            next_pos = (self.pos[0], self.pos[1] - 1)
        else:
            raise ValueError("invalid direction")

        if self.is_valid_position(next_pos):
            self.reversed = next_pos == self.last_pos
            self.last_pos = self.pos
            self.pos = next_pos
            self.stuck = False
        else:
            self.stuck = True
