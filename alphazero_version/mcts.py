import math
from typing import Optional, List

import numpy as np
import torch


class SearchNode:
    """Single node in the Monte Carlo search tree."""

    def __init__(
        self,
        game,
        config: dict,
        state: np.ndarray,
        parent: Optional["SearchNode"] = None,
        action_taken: Optional[int] = None,
        prior: float = 0.0,
    ) -> None:
        self.game = game
        self.config = config
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = float(prior)

        self.children: List["SearchNode"] = []
        self.visit_count: int = 0
        self.value_sum: float = 0.0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def select_child(self) -> "SearchNode":
        """Select a child according to PUCT."""
        best_score = -float("inf")
        best_child: Optional["SearchNode"] = None

        for child in self.children:
            score = self._puct_score(child)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child  # type: ignore[return-value]

    def _puct_score(self, child: "SearchNode") -> float:
        if child.visit_count == 0:
            q_value = 0.0
        else:
            q_value = 1.0 - ((child.value + 1.0) / 2.0)

        c = float(self.config["C"])
        exploration = c * math.sqrt(self.visit_count + 1e-8) / (1.0 + child.visit_count)
        return q_value + exploration * child.prior

    def expand(self, policy: np.ndarray) -> None:
        """Create children according to the given policy distribution."""
        for action, prob in enumerate(policy):
            if prob <= 0.0:
                continue

            next_state = self.game.get_next_state(self.state, action, player=1)
            next_state = self.game.change_perspective(next_state, player=-1)

            child = SearchNode(
                game=self.game,
                config=self.config,
                state=next_state,
                parent=self,
                action_taken=action,
                prior=float(prob),
            )
            self.children.append(child)

    def backpropagate(self, value: float) -> None:
        """Propagate value back up the tree, flipping perspective at each step."""
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            flipped = self.game.get_opponent_value(value)
            self.parent.backpropagate(flipped)


class MCTS:
    """Monte Carlo Tree Search guided by a neural network."""

    def __init__(self, game, config: dict, model: torch.nn.Module) -> None:
        self.game = game
        self.config = config
        self.model = model

    @torch.no_grad()
    def search(self, state: np.ndarray) -> np.ndarray:
        """
        Run MCTS from the given state.
        The state is given from the next player's neutral perspective.
        """
        root = SearchNode(self.game, self.config, state)
        root.visit_count = 1

        encoded = self.game.get_encoded_state(state)
        policy_logits, _ = self.model(
            torch.tensor(encoded, device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves

        # normalization
        if policy.sum() > 0:
            policy /= policy.sum()
        else:
            policy = valid_moves / valid_moves.sum()

        if "dirichlet_epsilon" in self.config and "dirichlet_alpha" in self.config:
            alpha = float(self.config["dirichlet_alpha"])
            epsilon = float(self.config["dirichlet_epsilon"])
            noise = np.random.dirichlet([alpha] * self.game.action_size)
            policy = (1.0 - epsilon) * policy + epsilon * noise

        root.expand(policy)

        num_searches = int(self.config["num_searches"])
        for _ in range(num_searches):
            node = root

            # selection
            while node.is_expanded():
                node = node.select_child()

            # evaluate leaf
            value, terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not terminal:
                encoded_leaf = self.game.get_encoded_state(node.state)
                logits, value_net = self.model(
                    torch.tensor(encoded_leaf, device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(logits, dim=1).cpu().numpy()[0]
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                if policy.sum() > 0:
                    policy /= policy.sum()
                else:
                    policy = valid_moves / valid_moves.sum()

                value = float(value_net.item())
                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size, dtype=np.float32)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count

        if action_probs.sum() > 0:
            action_probs /= action_probs.sum()
        return action_probs
