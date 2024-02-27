from .variables import *


class PacBot:
    """
    Allows initializing and updating information about PacBot
    """

    def __init__(self, game_state):
        self._game_state = game_state
        self.respawn()

    def respawn(self):
        self.pos = pacbot_starting_pos
        self.direction = pacbot_starting_dir

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

    def update_from_direction(self, direction):
        if self._game_state.update_ticks % ticks_per_update != 0:
            return

        direction = localize_direction(self.direction, direction)

        if direction == right and self.pos[0] == 27:
            return
        if direction == left and self.pos[0] == 0:
            return
        if direction == up and self.pos[1] == 30:
            return
        if direction == down and self.pos[1] == 0:
            return

        next_pos = self.pos

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

        if self._game_state.grid[next_pos[0]][next_pos[1]] != I:
            self.pos = next_pos


def localize_direction(current_direction, direction):
    if current_direction == right:
        if direction == right:
            return down
        elif direction == left:
            return up
        elif direction == up:
            return right
        elif direction == down:
            return left
    elif current_direction == left:
        if direction == right:
            return up
        elif direction == left:
            return down
        elif direction == up:
            return left
        elif direction == down:
            return right
    elif current_direction == up:
        if direction == right:
            return right
        elif direction == left:
            return left
        elif direction == up:
            return up
        elif direction == down:
            return down
    elif current_direction == down:
        if direction == right:
            return left
        elif direction == left:
            return right
        elif direction == up:
            return down
        elif direction == down:
            return up
