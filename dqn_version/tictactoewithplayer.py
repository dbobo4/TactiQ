"""
Tic-Tac-Toe Game with Tkinter
------------------------------

This module implements a 4x4 variant of the classic Tic-Tac-Toe game using the Tkinter GUI framework.
A win is defined by matching a specified number of characters (default is 3) in a row, column, or diagonal.
The game features:

    - A 4x4 board built dynamically with Tkinter buttons.
    - Turn management between two players ('X' and 'O') with simple state switching.
    - Win and draw detection based on custom logic that checks adjacent cells in all directions.
    - An Orientation Enum that defines possible movement directions for win checking.
    - Helper functions for board creation, input validation, button command handling, and game restart.

Development Notes:
    This script was the initial GUI prototype of the game.
    Later, it served as the **foundation for creating the object-oriented environment** used by AI agents in the reinforcement learning framework.
    A modified version of this implementation also powers the `humanvsai.py` file, 
    where a human player can compete against a trained AI model using PyTorch.

Usage:
    To start the game, run this module directly:
        python tic_tac_toe.py

This implementation is designed as an educational example, built from scratch without using external libraries
besides Tkinter, to demonstrate GUI development and basic game logic in Python.
"""

from enum import Enum
import tkinter
from tkinter import messagebox

class Orientation(Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    LEFT_UPPER_DIAGONAL = "left_upper_diagonal"
    RIGHT_UPPER_DIAGONAL = "right_upper_diagonal"
    LEFT_LOWER_DIAGONAL = "left_lower_diagonal"
    RIGHT_LOWER_DIAGONAL = "right_lower_diagonal"

number_of_matched_characters = 3

num_of_row = 4
num_of_column = 4

window_width = 130
window_height = 130

player_state = True
board = []

is_win = False
is_draw = False

def change_player():
    global player_state
    player_state = not player_state

def get_current_player_sign():
    return 'X' if player_state == True else 'O'

def create_initial_table(window):
    global board
    full_button_table_list = []
    for i in range(num_of_row):
        inner_button_table_list = []
        for j in range(num_of_column):
            # the column and rows will equally get the gridspace
            button = tkinter.Button(
                window,
                text='',
                command=lambda k=i, l=j: command_for_button(k, l),
                font=('Arial', 8, 'bold'),
                width=3,  
                height=1,  
                relief='raised',
                bd=3
            )
            # 'nsew' means the buttons will fill the empty space 
            button.grid(row=i,column=j)
            inner_button_table_list.append(button)
        full_button_table_list.append(inner_button_table_list)
            
    board = full_button_table_list

def is_valid_input(user_input):
    validation_result = False
    if user_input.isdigit():
        if 0 < int(user_input) <= num_of_row*num_of_column:
            validation_result = True
    return validation_result

def command_for_button(row, col):
    global is_win, board
    
    if is_win or board[row][col]['text'] in ['X','O']:
        return

    current_player_sign = get_current_player_sign()
    board[row][col]['text'] = current_player_sign

    check_is_win(current_player_sign)
    check_is_draw()
    
    if is_win:
        # first agrumentum is the title of the window
        messagebox.showinfo('Game Over', f'The player {current_player_sign} has won')
    elif is_draw:
        messagebox.showinfo('Draw', f'Draw! None of the players have won')
    else:
        change_player()

def check_is_draw():
    global is_draw
    is_draw = all(board[i][j]['text'] in ['X', 'O'] for i in range(num_of_row) for j in range(num_of_column))
                    
def check_is_win(current_player_sign):
    global is_win
    global board

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

    def is_position_match(row, column, current_player_sign):
        current_orientation_match = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                current_row = row + i
                current_column = column + j
                if is_inside_table(current_row, current_column) and board[current_row][current_column]['text'] == current_player_sign:
                    current_orientation = get_direction(row, column, current_row, current_column)
                    match_count = count_matches(board, current_row, current_column, current_orientation, current_player_sign)
                    current_orientation_match = max(current_orientation_match, match_count)
        return current_orientation_match >= number_of_matched_characters - 1

    def count_matches(table, current_row, current_column, current_orientation, current_player_sign):
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
            return match_functions[current_orientation](table, current_row, current_column, 1, current_player_sign)
        else:
            return 0

    def is_inside_table(current_row, current_column):
        return (0 <= current_row < num_of_row) and (0 <= current_column < num_of_column)

    def left_stepper(table, current_row, current_column, step, current_player_sign):
        if step >= number_of_matched_characters or not is_inside_table(current_row, current_column - 1):
            return step
        return left_stepper(table, current_row, current_column - 1, step + 1, current_player_sign) if table[current_row][current_column - 1]['text'] == current_player_sign else step

    def right_stepper(table, current_row, current_column, step, current_player_sign):
        if step >= number_of_matched_characters or not is_inside_table(current_row, current_column + 1):
            return step
        return right_stepper(table, current_row, current_column + 1, step + 1, current_player_sign) if table[current_row][current_column + 1]['text'] == current_player_sign else step

    def up_stepper(table, current_row, current_column, step, current_player_sign):
        if step >= number_of_matched_characters or not is_inside_table(current_row - 1, current_column):
            return step
        return up_stepper(table, current_row - 1, current_column, step + 1, current_player_sign) if table[current_row - 1][current_column]['text'] == current_player_sign else step

    def down_stepper(table, current_row, current_column, step, current_player_sign):
        if step >= number_of_matched_characters or not is_inside_table(current_row + 1, current_column):
            return step
        return down_stepper(table, current_row + 1, current_column, step + 1, current_player_sign) if table[current_row + 1][current_column]['text'] == current_player_sign else step

    def left_upper_diagonal_stepper(table, current_row, current_column, step, current_player_sign):
        if step >= number_of_matched_characters or not is_inside_table(current_row - 1, current_column - 1):
            return step
        return left_upper_diagonal_stepper(table, current_row - 1, current_column - 1, step + 1, current_player_sign) if table[current_row - 1][current_column - 1]['text'] == current_player_sign else step

    def right_upper_diagonal_stepper(table, current_row, current_column, step, current_player_sign):
        if step >= number_of_matched_characters or not is_inside_table(current_row - 1, current_column + 1):
            return step
        return right_upper_diagonal_stepper(table, current_row - 1, current_column + 1, step + 1, current_player_sign) if table[current_row - 1][current_column + 1]['text'] == current_player_sign else step

    def left_lower_diagonal_stepper(table, current_row, current_column, step, current_player_sign):
        if step >= number_of_matched_characters or not is_inside_table(current_row + 1, current_column - 1):
            return step
        return left_lower_diagonal_stepper(table, current_row + 1, current_column - 1, step + 1, current_player_sign) if table[current_row + 1][current_column - 1]['text'] == current_player_sign else step

    def right_lower_diagonal_stepper(table, current_row, current_column, step, current_player_sign):
        if step >= number_of_matched_characters or not is_inside_table(current_row + 1, current_column + 1):
            return step
        return right_lower_diagonal_stepper(table, current_row + 1, current_column + 1, step + 1, current_player_sign) if table[current_row + 1][current_column + 1]['text'] == current_player_sign else step

    for i in range(num_of_row):
        for j in range(num_of_column):
            if board[i][j]['text'] == current_player_sign:
                if is_position_match(i, j, current_player_sign):
                    is_win = True
                    return
    
def restart_game():
    global is_win, is_draw, player_state, board
    is_win = False
    is_draw = False
    player_state = True
    for row in board:
        for button in row:
            button.config(text="")

def main():
    main_window = tkinter.Tk()
    main_window.title('Tic-Tac-Toe')
    main_window.resizable(False,False)
    main_window.geometry(f'{window_width}x{window_height}')
    
    restart_button = tkinter.Button(main_window, text="Restart", command=restart_game)
    restart_button.grid(row=num_of_row, columnspan=num_of_column)

    create_initial_table(main_window)
    
    main_window.mainloop()

if __name__ == "__main__":
    main()