"""Tkinter GUI to play against a trained AlphaZero-style agent on 4x4 Tic-Tac-Toe."""

import os
import random
import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
import torch

from tictactoeboard import TicTacToeBoard
from tictactoe_game import TicTacToeGame
from model import PolicyValueNet
from mcts import MCTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "trained_models")


def _latest_checkpoint_path() -> str:
    """Return path to the latest model_<iteration>.pth file."""
    candidates = [
        name
        for name in os.listdir(MODELS_DIR)
        if name.startswith("model_") and name.endswith(".pth")
    ]
    if not candidates:
        raise FileNotFoundError(f"No model_*.pth files found in {MODELS_DIR}")

    def _iteration(name: str) -> int:
        core = name[len("model_") : -len(".pth")]
        return int(core)

    latest = max(candidates, key=_iteration)
    return os.path.join(MODELS_DIR, latest)


# Game and model configuration -------------------------------------------------

env = TicTacToeGame(size=4, win_length=3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PolicyValueNet(env, num_res_blocks=4, num_hidden=64, device=device)

checkpoint = _latest_checkpoint_path()
state_dict = torch.load(checkpoint, map_location=device)
model.load_state_dict(state_dict)
model.eval()

mcts_config = {
    "C": 2.0,
    "num_searches": 200,
}
search = MCTS(env, mcts_config, model)

# GUI board state --------------------------------------------------------------

board = TicTacToeBoard(size=4, win_length=3)

board_rows = board.size
board_cols = board.size

buttons: list[list[tk.Button]] = []

game_over = False
human_player = 1   # +1 for X, -1 for O
ai_player = -1
current_player = 1  # X always starts


def _mark_to_symbol(mark: int) -> str:
    return "X" if mark == 1 else "O"


def _ask_player_side(root: tk.Tk) -> None:
    """Ask human which side they want to play."""
    global human_player, ai_player

    while True:
        choice = simpledialog.askstring(
            "Choose side",
            "Do you want to play as X or O?\n(X always starts)",
            parent=root,
        )
        if choice is None:
            human_player = 1
            ai_player = -1
            break

        choice = choice.strip().upper()
        if choice in ("X", "O"):
            human_player = 1 if choice == "X" else -1
            ai_player = -human_player
            break

        messagebox.showinfo("Invalid choice", "Please type X or O.")


def _create_grid(root: tk.Tk) -> None:
    """Create the button grid."""
    global buttons
    buttons = []
    for r in range(board_rows):
        row_buttons: list[tk.Button] = []
        for c in range(board_cols):
            btn = tk.Button(
                root,
                text="",
                command=lambda rr=r, cc=c: _on_click(rr, cc),
                font=("Arial", 16, "bold"),
                width=3,
                height=1,
                relief="raised",
                bd=3,
            )
            btn.grid(row=r, column=c)
            row_buttons.append(btn)
        buttons.append(row_buttons)


def _reset_button_labels() -> None:
    for row in buttons:
        for btn in row:
            btn.config(text="")


def _start_new_game(root: tk.Tk) -> None:
    """Reset internal state and optionally let the AI start."""
    global game_over, current_player

    _ask_player_side(root)

    board.reset()
    _reset_button_labels()
    game_over = False
    current_player = 1

    if current_player == ai_player:
        _ai_move()


def _on_click(row: int, col: int) -> None:
    """Handle human move."""
    global current_player, game_over

    if game_over:
        return
    if current_player != human_player:
        return
    if buttons[row][col]["text"] != "":
        return

    if not board.place_mark(row, col, human_player):
        return

    buttons[row][col]["text"] = _mark_to_symbol(human_player)

    if _check_after_move(human_player, is_human=True):
        return

    current_player = -current_player
    if current_player == ai_player:
        _ai_move()


def _ai_move() -> None:
    """Let the agent choose and play a move using MCTS."""
    global current_player, game_over

    if game_over:
        return

    state = board.get_state()
    flat = state.flatten()
    legal_indices = [i for i, v in enumerate(flat) if v == 0]

    if not legal_indices:
        game_over = True
        messagebox.showinfo("Draw", "Draw! No more legal moves.")
        return

    neutral_state = env.change_perspective(state, ai_player)
    action_probs = search.search(neutral_state)
    action = int(np.argmax(action_probs))

    if flat[action] != 0:
        action = int(random.choice(legal_indices))

    row = action // board.size
    col = action % board.size

    if not board.place_mark(row, col, ai_player):
        action = int(random.choice(legal_indices))
        row = action // board.size
        col = action % board.size
        board.place_mark(row, col, ai_player)

    buttons[row][col]["text"] = _mark_to_symbol(ai_player)

    if _check_after_move(ai_player, is_human=False):
        return

    current_player = -current_player


def _check_after_move(player_mark: int, is_human: bool) -> bool:
    """Check whether the move ended the game."""
    global game_over

    if board.check_is_win(player_mark):
        game_over = True
        who = "You" if is_human else "The AI"
        side = _mark_to_symbol(player_mark)
        messagebox.showinfo("Game Over", f"{who} ({side}) have won!")
        return True

    if board.check_is_draw():
        game_over = True
        messagebox.showinfo("Draw", "Draw! None of the players have won.")
        return True

    return False


def main() -> None:
    root = tk.Tk()
    root.title("Tic-Tac-Toe: AlphaZero vs Player")
    root.resizable(False, False)
    root.configure(padx=5, pady=5)

    _create_grid(root)

    restart_button = tk.Button(
        root,
        text="Restart",
        command=lambda: _start_new_game(root),
    )
    restart_button.grid(row=board_rows, columnspan=board_cols)

    root.update_idletasks()
    root.minsize(root.winfo_reqwidth(), root.winfo_reqheight())

    _start_new_game(root)

    root.mainloop()


if __name__ == "__main__":
    main()
