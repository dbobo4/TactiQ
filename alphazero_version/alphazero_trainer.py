import os
import random
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from mcts import MCTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "trained_models")
os.makedirs(MODELS_DIR, exist_ok=True)


class AlphaZeroTrainer:
    """AlphaZero-style self-play training loop for the 4x4 Tic-Tac-Toe game."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        game,
        config: Dict[str, Any],
        writer: Optional[Any] = None,
    ) -> None:
        self.net = model
        self.optimizer = optimizer
        self.game = game
        self.config = config

        self.search = MCTS(game, config, model)

        self.writer = writer
        self.global_step = 0

    # ------------------------------------------------------------------
    # Self-play
    # ------------------------------------------------------------------

    # Input: policy array of move probabilities, e.g. [0.1, 0.3, 0.6] over all actions.
    # If temperature > 0, rescale this distribution (sharper or flatter), else pick pure argmax.
    # Output: one action index randomly drawn according to the (possibly rescaled) probabilities.
    def _sample_action(self, policy: np.ndarray) -> int:
        """Sample an action index from the given policy distribution."""
        temperature = self.config.get("temperature", 1.0)
        if temperature == 0:
            return int(np.argmax(policy))

        scaled = policy.astype(np.float64) ** (1.0 / float(temperature))
        scaled_sum = scaled.sum()
        if scaled_sum <= 0:
            scaled = np.ones_like(policy, dtype=np.float64)
            scaled_sum = scaled.size
        scaled /= scaled_sum
        return int(np.random.choice(self.game.action_size, p=scaled))

    # Input: one full self-play episode:
    #   episode = [
    #       (state_0, policy_0, +1),
    #       (state_1, policy_1, -1),
    #       (state_2, policy_2, +1),
    #       ...
    #       (state_k, policy_k, +1),  # X wins with this move
    #   ]
    #   final_value = +1       # from the last player's point of view: win
    #   last_player = +1       # last move was made by X
    # For each step, attach the final game result from the current player's perspective (+1/0/-1).
    # Output: list of (encoded_state, policy_target, value_target), e.g.:
    #   [
    #       (encoded_state_0, policy_0, outcome_0),
    #       (encoded_state_1, policy_1, outcome_1),
    #       ...
    #       (encoded_state_k, policy_k, outcome_k),
    #   ]
    # The network trains on this: encoded_state_* as input, policy_* as policy head target
    # (cross-entropy), outcome_* as value head target (MSE on [-1, 0, 1]).
    def _encode_episode(
        self,
        episode: List[Tuple[np.ndarray, np.ndarray, int]],
        final_value: float,
        last_player: int,
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Convert a played episode into training samples."""
        encoded = []
        for state, policy, player in episode:
            if player == last_player:
                outcome = final_value
            else:
                outcome = self.game.get_opponent_value(final_value)

            encoded.append(
                (self.game.get_encoded_state(state), policy.astype(np.float32), float(outcome))
            )
        return encoded

    # Play one full self-play game:
    #   start from an empty board with current_player = +1 (X),
    #   at each step build a neutral_state (current player as +1), run MCTS to get policy,
    #   store (neutral_state, policy, current_player), sample an action and update the real state.
    # When the game ends, we know the final result (value) from the last player's perspective,
    # and _encode_episode(...) converts the collected trajectory into
    # (encoded_state, policy_target, value_target) tuples ready for training.
    def _run_self_play_episode(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Play one complete self-play game and return encoded training data."""
        trajectory: List[Tuple[np.ndarray, np.ndarray, int]] = []
        current_player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_perspective(state, current_player)
            policy = self.search.search(neutral_state)

            trajectory.append((neutral_state, policy, current_player))

            action = self._sample_action(policy)
            state = self.game.get_next_state(state, action, current_player)

            value, terminal = self.game.get_value_and_terminated(state, action)
            if terminal:
                return self._encode_episode(trajectory, value, current_player)

            current_player = self.game.get_opponent(current_player)

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    # Given a replay buffer of (encoded_state, policy_target, value_target) tuples,
    # shuffle it and iterate in mini-batches:
    #   - stack states/policies/values into tensors,
    #   - run a forward pass to get logits (policy head) and value_pred (value head),
    #   - compute policy loss (cross-entropy vs. MCTS policy) and value loss (MSE vs. final outcome),
    #   - backprop and update network parameters once per batch.
    # For the policy head, we explicitly use cross-entropy between MCTS target policy and network policy:
    #   log_probs = log_softmax(logits) gives log p_pred(j),
    #   policy_loss = -sum_j p_target(j) * log p_pred(j), averaged over the batch (one scalar loss).
    def _train_on_buffer(self, buffer: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        """Run one epoch over the replay buffer."""
        if not buffer:
            return

        random.shuffle(buffer)
        batch_size = int(self.config["batch_size"])

        for start in range(0, len(buffer), batch_size):
            end = start + batch_size
            batch = buffer[start:end]
            if not batch:
                continue

            states, policy_targets, value_targets = zip(*batch)

            states_np = np.asarray(states, dtype=np.float32)
            policy_np = np.asarray(policy_targets, dtype=np.float32)
            values_np = np.asarray(value_targets, dtype=np.float32).reshape(-1, 1)

            states_t = torch.tensor(states_np, device=self.net.device)
            policy_t = torch.tensor(policy_np, device=self.net.device)
            values_t = torch.tensor(values_np, device=self.net.device)

            logits, value_pred = self.net(states_t)

            log_probs = torch.log_softmax(logits, dim=1)
            policy_loss = -(policy_t * log_probs).sum(dim=1).mean()

            value_loss = F.mse_loss(value_pred, values_t)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.writer is not None:
                self.writer.add_scalar("train/policy_loss", policy_loss.item(), self.global_step)
                self.writer.add_scalar("train/value_loss", value_loss.item(), self.global_step)
                self.writer.add_scalar("train/total_loss", loss.item(), self.global_step)
                self.global_step += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    #   for each iteration:
    #     1) Self-play: use the current network + MCTS to generate many games,
    #        collect (encoded_state, MCTS_policy, final_outcome) into a replay buffer
    #        and track win/draw/loss statistics for Player 1.
    #     2) Training: switch the network to train mode and run several epochs over
    #        the replay buffer, updating weights using policy (cross-entropy) and
    #        value (MSE) losses.
    #     3) Checkpoint: save the current network parameters as model_<iteration>.pth.

    def learn(self) -> None:
        """Run the full AlphaZero training loop."""
        num_iterations = int(self.config["num_iterations"])
        num_self_play_games = int(self.config["num_selfPlay_iterations"])
        num_epochs = int(self.config["num_epochs"])

        for iteration in range(num_iterations):
            replay_buffer: List[Tuple[np.ndarray, np.ndarray, float]] = []

            # 1) Self-play phase
            self.net.eval()

            wins = 0
            draws = 0
            losses = 0
            total_moves = 0

            for _ in trange(num_self_play_games, desc=f"Self-play {iteration}"):
                episode_data = self._run_self_play_episode()
                replay_buffer.extend(episode_data)

                if episode_data:
                    final_outcome = episode_data[0][2]
                    total_moves += len(episode_data)

                    if final_outcome > 0:
                        wins += 1
                    elif final_outcome < 0:
                        losses += 1
                    else:
                        draws += 1

            if self.writer is not None and num_self_play_games > 0:
                win_rate = wins / num_self_play_games
                draw_rate = draws / num_self_play_games
                loss_rate = losses / num_self_play_games
                avg_game_length = total_moves / num_self_play_games

                self.writer.add_scalar("selfplay/win_rate", win_rate, iteration)
                self.writer.add_scalar("selfplay/draw_rate", draw_rate, iteration)
                self.writer.add_scalar("selfplay/loss_rate", loss_rate, iteration)
                self.writer.add_scalar("selfplay/avg_game_length", avg_game_length, iteration)

            # 2) Training phase
            self.net.train()
            for _ in trange(num_epochs, desc=f"Train {iteration}"):
                self._train_on_buffer(replay_buffer)

            # 3) Save checkpoint
            checkpoint_path = os.path.join(MODELS_DIR, f"model_{iteration}.pth")
            torch.save(self.net.state_dict(), checkpoint_path)

# to run tensorboard GO TO THE ROOT (TACTIQ/) directory and write this into cmd: tensorboard --logdir=alphazero_version/runs
# then navigate to http://localhost:6006/