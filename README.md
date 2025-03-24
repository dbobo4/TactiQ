# TactiQ

TactiQ is a self-play reinforcement learning project that trains agents to play a 4×4 variant of Tic Tac Toe using Deep Q-Learning with convolutional neural networks (CNNs). This project was built entirely from scratch using PyTorch and numpy, without relying on external game libraries like OpenAI Gym. It serves both as an educational tool for understanding reinforcement learning and as a foundation for more advanced approaches (e.g., AlphaZero-style methods).

## Features

- **Self-Play:** Two identical agents (with a shared architecture and environment) compete against each other, enabling the system to learn through self-play.
- **Deep Q-Learning with CNNs:** Uses a CNN-based model to approximate Q-values for board states.
- **Action Masking:** Implements action masking to ensure that agents select only valid moves.
- **Custom Environment:** The Tic Tac Toe environment is custom-built, giving complete control over game rules, state representation, and reward design.
- **PyTorch Implementation:** All models and training routines are implemented in PyTorch, providing a deep dive into hands-on RL development.

## Project Structure

TactiQ/
├── alphazero_version/
│   └── coming_soon.txt           # Placeholder for future AlphaZero-style version
├── dqn_version/
│   ├── agent.py                  # RL agent logic, memory management, and action selection with masking
│   ├── humanvsai.py              # Script for playing against a trained AI model
│   ├── lastmemory.py             # Helper class for storing the last move memory
│   ├── model.py                  # CNN model and Q-value trainer (Deep Q-Learning implementation)
│   ├── TicTacToeEnv.py           # Custom Tic Tac Toe environment implementation
│   ├── tictactoewithplayer.py    # Optional player-vs-player or scripted gameplay
│   ├── train.py                  # Training loop for self-play learning
│   ├── trained_model_O.pth       # Saved PyTorch model (O agent)
│   ├── trained_model_X.pth       # Saved PyTorch model (X agent)
│   └── runs/                     # TensorBoard logs
└── README.md                     # This file

---

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/TactiQ.git
    cd TactiQ
    ```

2. **Install Dependencies**:
    ```bash
    pip install torch numpy
    ```

---

## Usage

1. **Navigate to the DQN version**:
    ```bash
    cd dqn_version
    ```

2. **Train the Agents**:
    ```bash
    python train.py
    ```
    This starts a self-play training session. If you want to monitor training progress, run:
    ```bash
    tensorboard --logdir=runs
    ```
    and open [http://localhost:6006](http://localhost:6006) in your browser.

3. **Play Against the Trained Model**:
    ```bash
    python humanvsai.py
    ```
    This allows a human player to challenge the trained DQN agent.

---

## Development

This project was developed as an educational exercise to understand reinforcement learning from the ground up. Initially implemented as a simple Tic Tac Toe game, the environment later served as a basis for training AI agents. In a subsequent version, the same environment was adapted for human versus AI gameplay (see humanvsai.py). **While the current solution is fully functional, it is not the optimal approach** for mastering the game. Therefore, the next phase involves creating an **AlphaZero-style** version (see the `alphazero_version` folder) to delve deeper into advanced reinforcement learning techniques such as Monte Carlo Tree Search (MCTS) and policy/value networks.

By evolving from a simple DQN to an AlphaZero-like architecture, the project aims to provide a comprehensive learning experience on **deep reinforcement learning**, bridging fundamental concepts and cutting-edge methods.

---

## License

This project is licensed under the **MIT License**.
