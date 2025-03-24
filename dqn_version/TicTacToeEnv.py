"""
Tic Tac Toe Environment Module
------------------------------

This module provides an object-oriented implementation of a Tic Tac Toe game environment,
designed primarily for reinforcement learning experiments. The environment features a configurable
board size (default is 4x4) and win condition (default is 3 consecutive marks). It is built from scratch
using numpy, without relying on external game libraries.

Key Components:
    - Orientation Enum:
        Defines possible directions (LEFT, RIGHT, UP, DOWN, and four diagonals) that are used
        in the win-checking logic to evaluate consecutive marks.

    - TicTacToeEnv Class:
        Implements the game environment with the following functionalities:
            * Board Management:
                - Initializes a board as a 2D numpy array.
                - Provides a reset() method to clear the board and reset the game state.
            * Move Processing (play_step):
                - Converts a flat action value into board coordinates.
                - Checks if the selected cell is already occupied; if so, returns a negative reward and a termination flag.
                - Places the current player's mark on the board if the move is legal.
                - Evaluates the game outcome: win (returns a positive reward), draw (returns a neutral/negative reward),
                  or continuing game (returns a score based on consecutive marks).
            * Win and Draw Detection:
                - Uses helper methods like count_consecutive() to compute the number of adjacent matching marks.
                - check_is_win() traverses the board using various directional stepper functions to determine if the win condition is met.
                - check_is_draw() determines whether the board is full.
            * Utility Methods:
                - convert_action_to_coordinates(): Transforms a single integer action into row and column coordinates.
                - get_board_actual_state(): Returns the current board configuration.
                - is_inside_table(): Validates if a given coordinate is within board bounds.

This module serves as a custom, non-standard board game environment that was later adapted to provide 
a framework for training AI agents (via object-oriented re-implementation). A variant of this environment 
is also used in the humanvsai.py file, where a trained model competes against a human player.

Usage:
    To create an environment and interact with it:
        >>> env = TicTacToeEnv(size=4, win_length=3)
        >>> state = env.get_board_actual_state()
        >>> reward, done, illegal = env.play_step(action=5, current_player_mark=1)
"""


import numpy as np
from enum import Enum

class Orientation(Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    LEFT_UPPER_DIAGONAL = "left_upper_diagonal"
    RIGHT_UPPER_DIAGONAL = "right_upper_diagonal"
    LEFT_LOWER_DIAGONAL = "left_lower_diagonal"
    RIGHT_LOWER_DIAGONAL = "right_lower_diagonal"

class TicTacToeEnv():
    def __init__(self, size=4, win_length=3):
        self.size = size
        self.win_length = win_length
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_step = 0
        self.reset()

    def reset(self):
        self.board.fill(0)
        self.current_step = 0
    
    def check_is_draw(self):
        if not (self.board == 0).any():
            return True
        return False
    
    def count_consecutive(self, player, row, col):
        # the first is not rewarded
        max_streak = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            streak = 0
            r, c = row + dr, col + dc
            while self.is_inside_table(r, c) and self.board[r, c] == player:
                streak += 1
                r += dr
                c += dc
            max_streak = max(max_streak, streak)
        return max_streak
    
    def get_board_actual_state(self):
        return self.board

    def play_step(self, action, current_player_mark):
        # Convertion of action value into row and column values
        row, col = self.convert_action_to_coordinates(action)

        self.current_step += 1

        # Check weather the index is occupied or not
        if self.board[row][col] != 0:
            # need to set the termination value to True to terminate and
            # to use the pure reward(-5) in Bellman equation
            return -5.0, True, True # the move is illegal

        # Set player's mark
        self.board[row][col] = current_player_mark

        # Check for draw
        is_draw = self.check_is_draw()
        if is_draw:
            return -2.0, True, False

        # Check actual player's win
        is_win = self.check_is_win(current_player_mark)
        if is_win:
            return 10.0, True, False

        # If nothing happens we give back the value of consecutive marks
        score = self.count_consecutive(current_player_mark, row, col)
        
        return score, False, False

    def convert_action_to_coordinates(self, action):
        # action = 14
        # self.size = (e.g.:) 5 (column size)
        # row = 14 // 5 = 2
        # col = 14 - (2*5) = 4 
        row = action // self.size
        col = action - (row * self.size)
        return row, col

    def check_is_win(self, current_player_mark):
        current_player_sign = current_player_mark

        def get_direction(pre_row, pre_column, current_row, current_column):
            if pre_row == current_row and current_column > pre_column:
                return Orientation.RIGHT
            elif pre_row == current_row and current_column < pre_column:
                return Orientation.LEFT
            elif pre_column == current_column and current_row > pre_row:
                return Orientation.DOWN
            elif pre_column == current_column and current_row < pre_row:
                return Orientation.UP
            elif current_row < pre_row and current_column < pre_column:
                return Orientation.LEFT_UPPER_DIAGONAL
            elif current_row < pre_row and current_column > pre_column:
                return Orientation.RIGHT_UPPER_DIAGONAL
            elif current_row > pre_row and current_column < pre_column:
                return Orientation.LEFT_LOWER_DIAGONAL
            elif current_row > pre_row and current_column > pre_column:
                return Orientation.RIGHT_LOWER_DIAGONAL
            return None

        def is_position_match(row, column):
            current_orientation_match = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    current_row = row + i
                    current_column = column + j
                    if self.is_inside_table(current_row, current_column) and self.board[current_row, current_column] == current_player_sign:
                        current_orientation = get_direction(row, column, current_row, current_column)
                        match_count = count_matches(current_row, current_column, current_orientation)
                        current_orientation_match = max(current_orientation_match, match_count)
            return current_orientation_match >= self.win_length - 1

        def count_matches(current_row, current_column, current_orientation):
            match_functions = {
                Orientation.LEFT: left_stepper,
                Orientation.RIGHT: right_stepper,
                Orientation.UP: up_stepper,
                Orientation.DOWN: down_stepper,
                Orientation.LEFT_UPPER_DIAGONAL: left_upper_diagonal_stepper,
                Orientation.RIGHT_UPPER_DIAGONAL: right_upper_diagonal_stepper,
                Orientation.LEFT_LOWER_DIAGONAL: left_lower_diagonal_stepper,
                Orientation.RIGHT_LOWER_DIAGONAL: right_lower_diagonal_stepper,
            }
            if current_orientation in match_functions:
                return match_functions[current_orientation](current_row, current_column, 1)
            else:
                return 0

        def left_stepper(current_row, current_column, step):
            if step >= self.win_length or not self.is_inside_table(current_row, current_column - 1):
                return step
            return left_stepper(current_row, current_column - 1, step + 1) if self.board[current_row, current_column - 1] == current_player_sign else step

        def right_stepper(current_row, current_column, step):
            if step >= self.win_length or not self.is_inside_table(current_row, current_column + 1):
                return step
            return right_stepper(current_row, current_column + 1, step + 1) if self.board[current_row, current_column + 1] == current_player_sign else step

        def up_stepper(current_row, current_column, step):
            if step >= self.win_length or not self.is_inside_table(current_row - 1, current_column):
                return step
            return up_stepper(current_row - 1, current_column, step + 1) if self.board[current_row - 1, current_column] == current_player_sign else step

        def down_stepper(current_row, current_column, step):
            if step >= self.win_length or not self.is_inside_table(current_row + 1, current_column):
                return step
            return down_stepper(current_row + 1, current_column, step + 1) if self.board[current_row + 1, current_column] == current_player_sign else step

        def left_upper_diagonal_stepper(current_row, current_column, step):
            if step >= self.win_length or not self.is_inside_table(current_row - 1, current_column - 1):
                return step
            return left_upper_diagonal_stepper(current_row - 1, current_column - 1, step + 1) if self.board[current_row - 1, current_column - 1] == current_player_sign else step

        def right_upper_diagonal_stepper(current_row, current_column, step):
            if step >= self.win_length or not self.is_inside_table(current_row - 1, current_column + 1):
                return step
            return right_upper_diagonal_stepper(current_row - 1, current_column + 1, step + 1) if self.board[current_row - 1, current_column + 1] == current_player_sign else step

        def left_lower_diagonal_stepper(current_row, current_column, step):
            if step >= self.win_length or not self.is_inside_table(current_row + 1, current_column - 1):
                return step
            return left_lower_diagonal_stepper(current_row + 1, current_column - 1, step + 1) if self.board[current_row + 1, current_column - 1] == current_player_sign else step

        def right_lower_diagonal_stepper(current_row, current_column, step):
            if step >= self.win_length or not self.is_inside_table(current_row + 1, current_column + 1):
                return step
            return right_lower_diagonal_stepper(current_row + 1, current_column + 1, step + 1) if self.board[current_row + 1, current_column + 1] == current_player_sign else step

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == current_player_sign:
                    if is_position_match(i, j):
                        return True
        return False

    def is_inside_table(self, current_row, current_column):
        return 0 <= current_row < self.size and 0 <= current_column < self.size