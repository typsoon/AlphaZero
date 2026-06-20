from torch.optim import Adam
from network import AlphaZeroNetwork
from train import AlphaZeroTrainer
from typing import Type

import torch
from torch import nn

from pybind.engine_bind import Game, ReplayBuffer  # pyright: ignore


def get_network(game: Type[Game], resblock_filter_size=64, residual_block_count=10):
    state_dim = game.state_dim
    action_dim = game.action_dim

    network = AlphaZeroNetwork(
        state_dim[0],
        state_dim[1],
        state_dim[2],
        residual_block_count,
        action_dim,
        resblock_filter_size,
    )

    return network


def get_trainer(
    model: nn.Module,
    device: torch.device,
    replay_buffer: ReplayBuffer,
    minibatch_size=4096,
) -> AlphaZeroTrainer:
    optimizer = Adam(model.parameters(), weight_decay=1e-4)

    return AlphaZeroTrainer(model, replay_buffer, optimizer, device, minibatch_size)
