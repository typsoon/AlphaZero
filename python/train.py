from checkpoint_manager import CheckpointManager
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Callable, Type
from torch.utils.data import TensorDataset, DataLoader


from network import AlphaZeroNetwork

import sys
from tqdm import tqdm as base_tqdm
from tqdm.notebook import tqdm as notebook_tqdm

from pybind.self_play_bind import self_play  # pyright: ignore
import logging
from pybind.engine_bind import Game, ReplayBuffer  # pyright: ignore


tqdm = notebook_tqdm if "ipykernel" in sys.modules else base_tqdm


class AlphaZeroTrainer:
    def __init__(
        self,
        model: nn.Module,
        replay_buffer: ReplayBuffer,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device("cpu"),
        minibatch_size=4096,
    ):
        self.model = model
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.minibatch_size = minibatch_size
        self.device = device
        assert (
            torch.device(device).type == next(model.parameters()).device.type
            # and device.index == next(model.parameters()).device.index
        ), (
            f"device par: {device} should be the same as {next(model.parameters()).device}"
        )

    def train(self, batch_size=64, train_steps=1000):
        self.model.train()
        # torch.autograd.detect_anomaly()
        accum_steps = self.minibatch_size // batch_size
        assert self.minibatch_size % batch_size == 0
        # print(self.replay_buffer.size, self.minibatch_size)

        progress_bar: Any = range(train_steps)

        policy_loss: torch.Tensor = torch.tensor(0.0)
        value_loss: torch.Tensor = torch.tensor(0.0)

        if __debug__:
            progress_bar = tqdm(progress_bar)

        for step in progress_bar:
            states, target_policies, target_values = self.replay_buffer.sample(
                self.minibatch_size
            )

            minibatch = TensorDataset(
                states.to(self.device),
                target_policies.to(self.device),
                target_values.to(self.device),
            )
            dataloader = DataLoader(minibatch, batch_size=batch_size)

            # print(states.shape, target_policies.shape, target_values.shape)
            # print(states, target_policies, target_values)
            # print(target_policies, target_values)
            if states.shape[0] < self.minibatch_size:
                # Not enough data yet — skip training this step.
                logging.info(
                    f"Not enough data to train yet ({states.shape[0]} < {self.minibatch_size}). Skipping training step."
                )
                return

            debug_counter = 0
            for s_batch, pi_batch, v_batch in dataloader:
                assert s_batch.shape[0] != 0, (
                    f"s_batch shouldn't be empty, {debug_counter}"
                )

                p_logits, v_preds = self.model(s_batch)
                v_preds = v_preds.squeeze()

                logp = F.log_softmax(p_logits, dim=1)

                # Cross-entropy: -sum(pi * log_softmax(p)), matching AlphaZero paper.
                # Only average over entries where pi > 0 to avoid distorting targets.
                policy_loss = -(pi_batch * logp).sum(dim=1).mean()
                value_loss = F.mse_loss(v_preds, v_batch)

                loss = policy_loss + value_loss
                loss /= accum_steps
                loss.backward()

                debug_counter += 1

            self.optimizer.step()
            self.optimizer.zero_grad()

            if __debug__:
                if step % 100 == 0:
                    progress_bar.set_postfix(
                        {
                            "policy loss": f"{policy_loss.item():.4f}",  # pyright: ignore
                            "value loss": f"{value_loss.item():.4f}",  # pyright: ignore
                        }
                    )


def self_play_and_train_loop(
    # """
    # Main orchestrator for the AlphaZero training pipeline.
    #
    # Args:
    #     checkpoint_manager: Handles tracking, saving, and rotating network history. Must be initialized and populated before passing.
    # """
    checkpoint_manager: CheckpointManager,
    network_type: Type[AlphaZeroNetwork],
    network_device: torch.device,
    game_type: Type[Game],
    trainer_factory: Callable[
        [nn.Module, torch.device, ReplayBuffer, int], AlphaZeroTrainer
    ],
    loop_iterations: int,
    games_in_each_iteration: int,
    batch_size: int,
    training_iterations: int,
    thread_count: int,
    replay_buffer_size: int,
    minibatch_size: int,
):
    # We MUST pass network_device here so the C++ clone() inherits the same device.
    # Otherwise, it defaults to CPU, causing a CUDA/CPU mismatch during batched inference.
    game = game_type(network_device)
    replay_buffer = ReplayBuffer(replay_buffer_size)

    network = network_type.load_az_network(
        checkpoint_manager.get_latest_checkpoint_file(), network_device
    )

    for _ in range(loop_iterations):
        self_play(
            game,
            checkpoint_manager.get_latest_scripted_checkpoint_file(),
            replay_buffer,
            games_in_each_iteration,
            thread_count,
        )

        trainer = trainer_factory(
            network, network_device, replay_buffer, minibatch_size
        )
        trainer.train(batch_size, training_iterations)

        checkpoint_manager.add_checkpoint(network)
