"""
Shared Tic-Tac-Toe board logic for GUI and evaluation.

This module mirrors the board representation and win/draw logic used in
the reinforcement learning environment, so that the GUI and evaluation
code rely on the same style of board logic.
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


class TicTacToeBoard:
    def __init__(self, size: int = 4, win_length: int = 3) -> None:
        self.size = size
        self.win_length = win_length
        self.board = np.zeros((self.size, self.size), dtype=int)

    def reset(self) -> None:
        self.board.fill(0)

    def get_state(self) -> np.ndarray:
        """Return a copy of the internal board state."""
        return self.board.copy()

    def is_inside_table(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def check_is_draw(self) -> bool:
        """True if there is no empty cell left."""
        return not (self.board == 0).any()

    def convert_action_to_coordinates(self, action: int) -> tuple[int, int]:
        """Convert flat action index into (row, col)."""
        row = action // self.size
        col = action - (row * self.size)
        return row, col

    def convert_coordinates_to_action(self, row: int, col: int) -> int:
        """Convert (row, col) into flat action index."""
        return row * self.size + col

    def place_mark(self, row: int, col: int, player_mark: int) -> bool:
        """
        Place a mark (1 for X, -1 for O) on the board.
        Returns False if the cell is invalid or already occupied.
        """
        if not self.is_inside_table(row, col):
            return False
        if self.board[row, col] != 0:
            return False
        self.board[row, col] = player_mark
        return True

    def count_consecutive(self, player: int, row: int, col: int) -> int:
        """
        Count the maximum number of consecutive marks adjacent to (row, col)
        in any direction, for the given player.
        """
        max_streak = 0
        directions = [
            (-1, 0), (1, 0),   # vertical
            (0, -1), (0, 1),   # horizontal
            (-1, -1), (-1, 1), # diagonals
            (1, -1), (1, 1),
        ]
        for dr, dc in directions:
            streak = 0
            r, c = row + dr, col + dc
            while self.is_inside_table(r, c) and self.board[r, c] == player:
                streak += 1
                r += dr
                c += dc
            max_streak = max(max_streak, streak)
        return max_streak

    def check_is_win(self, current_player_mark: int) -> bool:
        """
        Check whether the given player has reached the required win length.
        Logic follows the same pattern as in tictactoeenv.py.
        """
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
                    if self.is_inside_table(current_row, current_column) and \
                            self.board[current_row, current_column] == current_player_sign:
                        current_orientation = get_direction(
                            row, column, current_row, current_column
                        )
                        match_count = count_matches(
                            current_row, current_column, current_orientation
                        )
                        current_orientation_match = max(
                            current_orientation_match, match_count
                        )
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
            return left_stepper(current_row, current_column - 1, step + 1) \
                if self.board[current_row, current_column - 1] == current_player_sign else step

        def right_stepper(current_row, current_column, step):
            if step >= self.win_length or not self.is_inside_table(current_row, current_column + 1):
                return step
            return right_stepper(current_row, current_column + 1, step + 1) \
                if self.board[current_row, current_column + 1] == current_player_sign else step

        def up_stepper(current_row, current_column, step):
            if step >= self.win_length or not self.is_inside_table(current_row - 1, current_column):
                return step
            return up_stepper(current_row - 1, current_column, step + 1) \
                if self.board[current_row - 1, current_column] == current_player_sign else step

        def down_stepper(current_row, current_column, step):
            if step >= self.win_length or not self.is_inside_table(current_row + 1, current_column):
                return step
            return down_stepper(current_row + 1, current_column, step + 1) \
                if self.board[current_row + 1, current_column] == current_player_sign else step

        def left_upper_diagonal_stepper(current_row, current_column, step):
            if step >= self.win_length or not self.is_inside_table(current_row - 1, current_column - 1):
                return step
            return left_upper_diagonal_stepper(current_row - 1, current_column - 1, step + 1) \
                if self.board[current_row - 1, current_column - 1] == current_player_sign else step

        def right_upper_diagonal_stepper(current_row, current_column, step):
            if step >= self.win_length or not self.is_inside_table(current_row - 1, current_column + 1):
                return step
            return right_upper_diagonal_stepper(current_row - 1, current_column + 1, step + 1) \
                if self.board[current_row - 1, current_column + 1] == current_player_sign else step

        def left_lower_diagonal_stepper(current_row, current_column, step):
            if step >= self.win_length or not self.is_inside_table(current_row + 1, current_column - 1):
                return step
            return left_lower_diagonal_stepper(current_row + 1, current_column - 1, step + 1) \
                if self.board[current_row + 1, current_column - 1] == current_player_sign else step

        def right_lower_diagonal_stepper(current_row, current_column, step):
            if step >= self.win_length or not self.is_inside_table(current_row + 1, current_column + 1):
                return step
            return right_lower_diagonal_stepper(current_row + 1, current_column + 1, step + 1) \
                if self.board[current_row + 1, current_column + 1] == current_player_sign else step

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == current_player_sign:
                    if is_position_match(i, j):
                        return True
        return False
