from .variables import *
from .ghostpaths import *
from .ghostAgent import *
from .pacbot import *
from .grid import grid
import copy
import time
import numpy as np


FREQUENCY = game_frequency * ticks_per_update


class GameState:
    def __init__(self):
        self.ghosts_enabled = True
        self.pacbot = PacBot(self)
        self.red = GhostAgent(
            red_init_pos[0],
            red_init_pos[1],
            red_init_npos[0],
            red_init_npos[1],
            red,
            red_init_dir,
            self,
            [],
            red_scatter_pos,
        )
        self.pink = GhostAgent(
            pink_init_pos[0],
            pink_init_pos[1],
            pink_init_npos[0],
            pink_init_npos[1],
            pink,
            pink_init_dir,
            self,
            pink_start_path,
            pink_scatter_pos,
        )
        self.orange = GhostAgent(
            orange_init_pos[0],
            orange_init_pos[1],
            orange_init_npos[0],
            orange_init_npos[1],
            orange,
            red_init_dir,
            self,
            orange_start_path,
            orange_scatter_pos,
        )
        self.blue = GhostAgent(
            blue_init_pos[0],
            blue_init_pos[1],
            blue_init_npos[0],
            blue_init_npos[1],
            blue,
            blue_init_dir,
            self,
            blue_start_path,
            blue_scatter_pos,
        )
        self.just_swapped_state = False
        self.restart()
        self.ticks_since_spawn = 0
        self.prev_cherry_pellets = 0

    # Frightens all of the ghosts and saves the old state to be restored when frightened mode ends.
    def _become_frightened(self):
        if self.state != frightened:
            self.old_state = self.state
        self.state = frightened
        self.frightened_counter = frightened_length
        self.red.become_frightened()
        self.pink.become_frightened()
        self.orange.become_frightened()
        self.blue.become_frightened()
        self.just_swapped_state = True

    # Resets the state of the game to what it was before frightened,
    # and resets the score multiplier to be equal to 1.
    def _end_frightened(self):
        self.state = self.old_state
        self.frightened_multiplier = 1

    # Decreases the remaining time each ghost should be frightened for and updates each ghost's
    # current and next move information.
    def _update_ghosts(self):
        if not self.ghosts_enabled:
            return
        self.red.update()
        self.orange.update()
        self.pink.update()
        self.blue.update()

    # Returns true if the position of Pacman is occupied by a pellet.
    def _is_eating_pellet(self):
        return self.grid[self.pacbot.pos[0]][self.pacbot.pos[1]] == o

    # Returns true if the position of Pacman is occupied by a power pellet.
    def _is_eating_power_pellet(self):
        return self.grid[self.pacbot.pos[0]][self.pacbot.pos[1]] == O

    # Sets the current position of Pacman to empty and increments the score.
    def _eat_pellet(self):
        self.grid[self.pacbot.pos[0]][self.pacbot.pos[1]] = e
        self.score += pellet_score
        self.pellets -= 1
        self.ate_pellet = True

    # Sets the current position of Pacman to empty and increments the score.
    # Also makes all ghosts frightened.
    def _eat_power_pellet(self):
        self.grid[self.pacbot.pos[0]][self.pacbot.pos[1]] = e
        self.score += power_pellet_score
        self.power_pellets -= 1
        self._become_frightened()
        self.ate_power_pellet = True

    # Returns true if the position of Pacman is occupied by a cherry.
    def _is_eating_cherry(self):
        return self.grid[self.pacbot.pos[0]][self.pacbot.pos[1]] == c

    # Sets the current position of Pacman to empty and increments the score.
    def _eat_cherry(self):
        self.grid[self.pacbot.pos[0]][self.pacbot.pos[1]] = e
        self.score += cherry_score
        self.cherry = False
        self.ate_cherry = True

    # Returns true if the cherry should be spawned; this happens
    # when only 170 pellets remain.
    def _should_spawn_cherry(self):
        if (
            self.pellets == 170 or self.pellets == 70
        ) and self.prev_cherry_pellets != self.pellets:
            # print("Cherry spawned")
            self.prev_cherry_pellets = self.pellets
            return True
        # print(self.pellets)
        return False

    def _should_remove_cherry(self):
        if self.ticks_since_spawn >= FREQUENCY * 10:
            self.ticks_since_spawn = 0
            return True
        else:
            return False

    # Places the cherry on the board.
    def _spawn_cherry(self):
        self.grid[cherry_pos[0]][cherry_pos[1]] = c
        self.cherry = True

    def _despawn_cherry(self):
        self.grid[cherry_pos[0]][cherry_pos[1]] = e
        self.cherry = False

        # cherry to disappear when pacman dies

    # Updates the score based on what Pacman has just eaten
    # (what is in Pacman's current space on the board).
    def _update_score(self):
        if self._is_eating_pellet():
            self._eat_pellet()

        if self._is_eating_power_pellet():
            self._eat_power_pellet()

        if self._is_eating_cherry():
            self._eat_cherry()

    # Updates each agent's position and behavior to reflect the beginning of a new round.

    def _respawn_agents(self):
        self.pacbot.respawn()
        self.red.respawn()
        self.pink.respawn()
        self.orange.respawn()
        self.blue.respawn()

    def _end_game(self):
        self.elapsed_time += time.time() - self.previous_start
        self.play = False
        self.done = True
        # print("Score: " + str(self.score))
        # print("Time: " + str(self.elapsed_time))

    # Resets the round if Pacman dies with lives remaining
    # and ends the game if Pacman has no lives remaining.
    def _die(self):
        if self.lives > 1:
            self._respawn_agents()
            self.start_counter = 0
            self.state_counter = 0
            self.lives -= 1
            self.old_state = chase
            self.state = scatter
            self.frightened_counter = 0
            self.frightened_multiplier = 1
            self.pause()
            self._update_score()
            self.grid[cherry_pos[0]][cherry_pos[1]] = e
            self.lost_life = True
        else:
            self._end_game()
            self.dead = True

    # Returns true if Pacman has collided with a ghost and the ghost is not frightened.
    def _should_die(self):
        if not self.ghosts_enabled:
            return False
        return (
            (
                self.red.pos["current"] == self.pacbot.pos
                and self.red.frightened_counter == 0
            )
            or (
                self.pink.pos["current"] == self.pacbot.pos
                and self.pink.frightened_counter == 0
            )
            or (
                self.orange.pos["current"] == self.pacbot.pos
                and self.orange.frightened_counter == 0
            )
            or (
                self.blue.pos["current"] == self.pacbot.pos
                and self.blue.frightened_counter == 0
            )
        )

    # Checks for ghosts that have been eaten and sends them back
    # to the respawn zone if they have been eaten.
    def _check_if_ghosts_eaten(self):
        if not self.ghosts_enabled:
            return
        self._check_if_ghost_eaten(self.red)
        self._check_if_ghost_eaten(self.pink)
        self._check_if_ghost_eaten(self.orange)
        self._check_if_ghost_eaten(self.blue)

    # If the ghost was eaten, then the ghost is sent home, the score is updated, and
    # the score multiplier for Pacman in frightened mode is increased.
    def _check_if_ghost_eaten(self, ghost):
        if not self.ghosts_enabled:
            return
        if ghost.pos["current"] == self.pacbot.pos and ghost.frightened_counter > 0:
            ghost.send_home()
            self.score += ghost_score * self.frightened_multiplier
            self.frightened_multiplier += 1
            self.ate_ghost = True

    # Returns true if there are no more pellets left on the board.
    def _are_all_pellets_eaten(self):
        return self.pellets == 0 and self.power_pellets == 0

    # Returns true if the game is over.
    def _is_game_over(self):
        return self._are_all_pellets_eaten()

    # Changes the state of the game
    def _swap_state_if_necessary(self):
        if self.state_counter in state_swap_times:
            if self.state == chase:
                self.state = scatter
            else:
                self.state = chase
            self.just_swapped_state = True
        else:
            self.just_swapped_state = False

    def pause(self):
        self.elapsed_time += time.time() - self.previous_start
        self.play = False

    def unpause(self):
        self.previous_start = time.time()
        self.play = True

    def print_ghost_pos(self):
        ghosts = [self.red, self.pink, self.blue, self.orange]
        ret = ""
        for ghost in ghosts:
            ret += str(ghost.pos["current"]) + " "
        print(ret)

    def next_step(self):
        self.lost_life = False
        self.ate_pellet = False
        self.ate_power_pellet = False
        self.ate_cherry = False
        self.ate_ghost = False
        if self._is_game_over():
            self._end_game()
        elif self._should_die():
            self._die()
        else:
            self._check_if_ghosts_eaten()
            if self.update_ticks % ticks_per_update == 0:
                self._update_ghosts()
                if self._should_die():
                    self._die()
                self._check_if_ghosts_eaten()
                if self.state == frightened:
                    if self.frightened_counter == 1:
                        self._end_frightened()
                    elif self.frightened_counter == frightened_length:
                        self.just_swapped_state = False
                    self.frightened_counter -= 1
                else:
                    self._swap_state_if_necessary()
                    self.state_counter += 1
                self.start_counter += 1
                # self.print_ghost_pos()
            self._update_score()
            if self._should_spawn_cherry():
                self._spawn_cherry()
            if self.cherry:
                self.ticks_since_spawn += 1
            if self._should_remove_cherry():
                self._despawn_cherry()
            self.update_ticks += 1

    # Sets the game back to its original state (no rounds played).
    def restart(self):
        self.grid = copy.deepcopy(grid)
        self.total_pellets = sum([col.count(o) for col in self.grid])
        self.pellets = self.total_pellets
        self.power_pellets = sum([col.count(O) for col in self.grid])
        self.cherry = False
        self.prev_cherry_pellets = 0
        self.old_state = chase
        self.state = scatter
        self.just_swapped_state = False
        self.frightened_counter = 0
        self.frightened_multiplier = 1
        self._respawn_agents()
        self.score = 0
        self.play = False
        self.start_counter = 0
        self.state_counter = 0
        self.update_ticks = 0
        self.lives = starting_lives
        self.elapsed_time = 0
        self._update_score()
        self.grid[cherry_pos[0]][cherry_pos[1]] = e
        self.ticks_since_spawn = 0
        self.done = False
        self.dead = False

    def get_populated_grid(self):
        grid = np.array(self.grid)

        grid[self.pacbot.pos] = 10
        if self.ghosts_enabled:
            grid[self.red.pos["current"]] = (
                10 + red + (4 if self.red.is_frightened() else 0)
            )
            grid[self.pink.pos["current"]] = (
                10 + pink + (4 if self.pink.is_frightened() else 0)
            )
            grid[self.orange.pos["current"]] = (
                10 + orange + (4 if self.orange.is_frightened() else 0)
            )
            grid[self.blue.pos["current"]] = (
                10 + blue + (4 if self.blue.is_frightened() else 0)
            )

        return grid

    def __str__(self):
        import sys

        np.set_printoptions(threshold=sys.maxsize, linewidth=200, edgeitems=3)
        ret = str(self.get_populated_grid())
        ret += (
            "\nPacman  Score: "
            + str(self.score)
            + "  State: "
            + str(self.state)
            + "  Lives: "
            + str(self.lives)
        )

        return ret

    def rgb_array(self):
        COLORS = [
            [],
            [255, 255, 255],  # wall
            [0, 255, 0],  # normal pellet
            [0, 0, 0],  # empty space
            [0, 255, 255],  # power pellet
            [0, 63, 27],  # ghost chamber
            [192, 127, 0],  # cherry
            [],
            [],
            [],
            [255, 255, 0],  # pacman
            [255, 0, 0],  # red ghost
            [255, 0, 255],  # pink ghost
            [255, 127, 0],  # orange ghost
            [0, 0, 255],  # blue ghost
            [255, 63, 127],  # frightened red ghost
            [255, 127, 255],  # frightened pink ghost
            [255, 127, 63],  # frightened orange ghost
            [53, 127, 255],  # frightened blue ghost
        ]

        grid = self.get_populated_grid()

        return [[COLORS[cell] for cell in row] for row in grid]
