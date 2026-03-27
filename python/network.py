from typing import Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


from pybind import engine_bind  # pyright: ignore
from engine_bind import Game


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, X):
        residual = X
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.bn2(self.conv2(X))
        X += residual
        return F.relu(X)


class AlphaZeroNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int,
        height: int,
        width: int,
        num_residual_blocks: int,
        action_size: int,
        num_filters: int,
    ):
        super().__init__()
        self._height = height
        self._width = width

        self.conv_in = nn.Conv2d(
            input_channels, num_filters, kernel_size=3, stride=1, padding=1
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )
        self.bn_in = nn.BatchNorm2d(num_filters)

        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * height * width, action_size)

        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(height * width, num_filters)
        self.value_fc2 = nn.Linear(num_filters, 1)

    def forward(self, X):
        X = F.relu(self.bn_in(self.conv_in(X)))
        X = self.residual_blocks(X)

        policy = F.relu(self.policy_bn(self.policy_conv(X)))
        policy = policy.flatten(start_dim=1)
        policy = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(X)))
        value = value.flatten(start_dim=1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    @torch.jit.export
    def infer(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass and return policy and value.
        This method is exposed for TorchScript.
        """
        return self.forward(X)

    def save_az_network(self, path: str):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "init_args": {
                    "input_channels": self.conv_in.in_channels,
                    "height": self._height,
                    "width": self._width,
                    "num_residual_blocks": len(self.residual_blocks),
                    "action_size": self.policy_fc.out_features,
                    "num_filters": self.conv_in.out_channels,
                },
            },
            path,
        )

    def script_and_save_network(self, path: str):
        scripted_model = torch.jit.script(self)

        scripted_model.save(path)

    @staticmethod
    def load_az_network(path: str, device: torch.device) -> "AlphaZeroNetwork":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = AlphaZeroNetwork(**checkpoint["init_args"])
        model = model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
