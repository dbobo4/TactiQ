import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tictactoe_game import TicTacToeGame
from model import PolicyValueNet
from alphazero_trainer import AlphaZeroTrainer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "runs", "alphazero_run")


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    os.makedirs(RUNS_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=RUNS_DIR)

    game = TicTacToeGame(size=4, win_length=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyValueNet(game, num_res_blocks=4, num_hidden=64, device=device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    config = {
        "C": 2.0,
        "num_searches": 60,
        "num_iterations": 3,
        "num_selfPlay_iterations": 500,
        "num_epochs": 4,
        "batch_size": 64,
        "temperature": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3,
    }

    trainer = AlphaZeroTrainer(model, optimizer, game, config, writer=writer)
    trainer.learn()

    writer.close()


if __name__ == "__main__":
    main()
