from typing import Any, Optional

class LastMemory:
    """
A helper class for storing the last memory of an agent's move.

This class is used to store the state before an action,
the action itself, the reward obtained, the resulting state, and
whether the episode has ended. This information is then used
for short-term training updates, especially to penalize the opponent
if the current move resulted in a win for the other player.

Attributes:
    state_old (Any): The state before the action was taken.
    action (Any): The action taken by the agent.
    reward (float): The reward received after the action.
    state_new (Any): The state after the action.
    done (bool): Flag indicating whether the episode has ended.
    """
    
    state_old: Any
    action: Optional[int]
    reward: Optional[float]
    state_new: Any
    done: Optional[bool]

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state_old = None
        self.action    = None
        self.reward    = None
        self.state_new = None
        self.done      = None