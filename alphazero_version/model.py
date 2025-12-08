import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block with two conv-BN-ReLU layers."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + identity
        return F.relu(x)


class PolicyValueNet(nn.Module):
    """
    Input:  (batch, 3, board_size, board_size)
    Output:
        - policy_logits: (batch, action_size)
        - value:         (batch, 1) in [-1, 1]
    """

    def __init__(
        self,
        game,
        num_res_blocks: int,
        num_hidden: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device

        self.stem = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_hidden) for _ in range(num_res_blocks)]
        )

        board_area = game.row_count * game.column_count

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_area, game.action_size),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * board_area, 1),
            nn.Tanh(),
        )

        self.to(device)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        for block in self.res_blocks:
            x = block(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value
