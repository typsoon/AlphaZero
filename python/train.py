from .checkpoint_manager import CheckpointManager
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Callable, Type

from .network import AlphaZeroNetwork

import sys
from tqdm import tqdm as base_tqdm
from tqdm.notebook import tqdm as notebook_tqdm

import logging
from .pybind.engine_bind import Game, ReplayBuffer  # pyright: ignore


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

        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        assert (
            torch.device(device).type == next(model.parameters()).device.type
            # and device.index == next(model.parameters()).device.index
        ), (
            f"device par: {device} should be the same as {next(model.parameters()).device}"
        )

    def train(self, batch_size=64, train_steps=1000):
        self.model.train()
        accum_steps = self.minibatch_size // batch_size
        assert self.minibatch_size % batch_size == 0

        progress_bar: Any = range(train_steps)

        policy_loss: torch.Tensor = torch.tensor(0.0)
        value_loss: torch.Tensor = torch.tensor(0.0)

        if __debug__:
            progress_bar = tqdm(progress_bar)

        for step in progress_bar:
            states, target_policies, target_values = self.replay_buffer.sample(
                self.minibatch_size
            )

            if states.shape[0] < self.minibatch_size:
                # Not enough data yet — skip training this step.
                logging.info(
                    f"Not enough data to train yet ({states.shape[0]} < {self.minibatch_size}). Skipping training step."
                )
                return

            states = states.to(self.device, non_blocking=True)
            target_policies = target_policies.to(self.device, non_blocking=True)
            target_values = target_values.to(self.device, non_blocking=True)

            for i in range(0, self.minibatch_size, batch_size):
                s_batch = states[i : i + batch_size]
                pi_batch = target_policies[i : i + batch_size]
                v_batch = target_values[i : i + batch_size]

                with torch.autocast(
                    device_type=self.device.type, enabled=(self.device.type == "cuda")
                ):
                    p_logits, v_preds = self.model(s_batch)
                    v_preds = v_preds.squeeze(-1)

                    # Fused cross-entropy is much faster and more memory-efficient
                    # than manual log_softmax + multiply + sum
                    policy_loss = F.cross_entropy(p_logits, pi_batch)
                    value_loss = F.mse_loss(v_preds, v_batch)

                    loss = policy_loss + value_loss
                    loss /= accum_steps

                self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

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
    game_data: tuple[Type[Game], Callable],
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
    game_type, self_play_method = game_data
    game = game_type(network_device)
    replay_buffer = ReplayBuffer(replay_buffer_size)

    network = network_type.load_az_network(
        checkpoint_manager.get_latest_checkpoint_file(), network_device
    )

    for _ in range(loop_iterations):
        self_play_method(
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
