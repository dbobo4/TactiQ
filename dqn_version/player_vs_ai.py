"""
Tkinter GUI to play against a trained DQN agent on the 4x4 Tic-Tac-Toe board.

- At the beginning of every game you can choose to play as X or as O.
- X always starts. If you choose O, the AI starts as X.
- Board logic (win/draw detection) is shared via TicTacToeBoard.
"""

import os
import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
import torch

from tictactoeboard import TicTacToeBoard
from model import ConvolutionalNeuralNetwork


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "trained_models")
MODEL_X_PATH = os.path.join(MODELS_DIR, "trained_model_X.pth")
MODEL_O_PATH = os.path.join(MODELS_DIR, "trained_model_O.pth")

# Shared board logic (same size/win_length as training)
board_logic = TicTacToeBoard(size=4, win_length=3)

num_of_row = board_logic.size
num_of_column = board_logic.size

buttons: list[list[tk.Button]] = []

game_over = False
human_mark = 1   # +1 → X, -1 → O
ai_mark = -1
current_mark = 1  # X always starts


def load_model(model_path: str) -> ConvolutionalNeuralNetwork:
    model = ConvolutionalNeuralNetwork()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


# Load both agents' networks; we switch depending on which side the human chooses
model_x = load_model(MODEL_X_PATH)
model_o = load_model(MODEL_O_PATH)


def mark_to_char(mark: int) -> str:
    return "X" if mark == 1 else "O"


def choose_player_side(root: tk.Tk) -> None:
    global human_mark, ai_mark

    while True:
        choice = simpledialog.askstring(
            "Choose side",
            "Do you want to play as X or O?\n(X always starts)",
            parent=root,
        )
        if choice is None:
            # default: human plays X
            human_mark = 1
            ai_mark = -1
            break
        choice = choice.strip().upper()
        if choice in ("X", "O"):
            human_mark = 1 if choice == "X" else -1
            ai_mark = -human_mark
            break
        messagebox.showinfo("Invalid choice", "Please type X or O.")


def create_board_widgets(root: tk.Tk) -> None:
    global buttons
    buttons = []
    for i in range(num_of_row):
        row_buttons: list[tk.Button] = []
        for j in range(num_of_column):
            btn = tk.Button(
                root,
                text="",
                command=lambda r=i, c=j: on_button_click(r, c),
                font=("Arial", 16, "bold"),
                width=3,
                height=1,
                relief="raised",
                bd=3,
            )
            btn.grid(row=i, column=j)
            row_buttons.append(btn)
        buttons.append(row_buttons)


def reset_gui_board() -> None:
    for row in buttons:
        for btn in row:
            btn.config(text="")


def start_new_game(root: tk.Tk) -> None:
    global game_over, current_mark

    choose_player_side(root)

    board_logic.reset()
    reset_gui_board()
    game_over = False
    current_mark = 1  # X always starts

    # If AI is X, let it move immediately
    if current_mark == ai_mark:
        ai_move()


def on_button_click(row: int, col: int) -> None:
    global current_mark, game_over

    if game_over:
        return

    # Only allow human to click on their own turn
    if current_mark != human_mark:
        return

    if buttons[row][col]["text"] != "":
        return

    if not board_logic.place_mark(row, col, human_mark):
        return

    buttons[row][col]["text"] = mark_to_char(human_mark)

    if check_game_over_after_move(human_mark, is_human=True):
        return

    current_mark = -current_mark  # switch turn

    if current_mark == ai_mark:
        ai_move()


def ai_move() -> None:
    global current_mark, game_over

    if game_over:
        return

    state = board_logic.get_state()
    flat = state.flatten()
    valid = [i for i, v in enumerate(flat) if v == 0]

    if not valid:
        game_over = True
        messagebox.showinfo("Draw", "Draw! No more legal moves.")
        return

    # Choose the correct model depending on which side the AI is playing
    model = model_x if ai_mark == 1 else model_o

    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_vals = model(state_t)[0]

    # Action masking for invalid moves
    mask = torch.full_like(q_vals, float("-inf"))
    mask[valid] = q_vals[valid]
    action = int(torch.argmax(mask).item())

    row = action // board_logic.size
    col = action - row * board_logic.size

    if not board_logic.place_mark(row, col, ai_mark):
        # Fallback: pick random valid move (should be very rare)
        action = int(np.random.choice(valid))
        row = action // board_logic.size
        col = action - row * board_logic.size
        board_logic.place_mark(row, col, ai_mark)

    buttons[row][col]["text"] = mark_to_char(ai_mark)

    if check_game_over_after_move(ai_mark, is_human=False):
        return

    current_mark = -current_mark  # switch turn back to human


def check_game_over_after_move(player_mark: int, is_human: bool) -> bool:
    global game_over

    if board_logic.check_is_win(player_mark):
        game_over = True
        who = "You" if is_human else "The AI"
        side = mark_to_char(player_mark)
        messagebox.showinfo("Game Over", f"{who} ({side}) have won!")
        return True

    if board_logic.check_is_draw():
        game_over = True
        messagebox.showinfo("Draw", "Draw! None of the players have won.")
        return True

    return False


def main() -> None:
    root = tk.Tk()
    root.title("Tic-Tac-Toe: Player vs AI")
    root.resizable(False, False)

    # A kis padding csak arra kell, hogy a tábla ne tapadjon teljesen a kerethez
    root.configure(padx=5, pady=5)

    create_board_widgets(root)

    restart_button = tk.Button(
        root,
        text="Restart",
        command=lambda: start_new_game(root),
    )
    restart_button.grid(row=num_of_row, columnspan=num_of_column)

    # Méretezzük az ablakot pontosan a tartalomhoz
    root.update_idletasks()
    root.minsize(root.winfo_reqwidth(), root.winfo_reqheight())

    start_new_game(root)

    root.mainloop()


if __name__ == "__main__":
    main()
