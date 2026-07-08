from .checkpoint_manager import CheckpointManager
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Callable, Optional, Type

from .network import AlphaZeroNetwork

import sys
from tqdm import tqdm as base_tqdm
from tqdm.notebook import tqdm as notebook_tqdm
from torch.utils.tensorboard import SummaryWriter

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

    def train(
        self,
        batch_size=64,
        train_steps=1000,
        writer: Optional[SummaryWriter] = None,
        global_step: int = 0,
        log_interval: int = 50,
    ) -> int:
        """Runs train_steps optimizer steps. Returns the global_step reached, so
        callers that create a new trainer each outer iteration (see
        self_play_and_train_loop) can keep TensorBoard's step axis contiguous
        across iterations instead of resetting to 0 every time.

        Sampling goes through one CachedSampler (see ReplayBuffer::get_sampler
        in replay_buffer.hpp) shared across every minibatch in this call, so a
        slot resampled in a later step can reuse its already-densified policy
        row instead of redoing the scatter_() from scratch. This used to risk
        the cache growing to cover the whole replay buffer (capacity *
        ~82KB/entry for chess - 26GB at a 320,000-entry buffer, a direct
        contributor to this project's training OOM kills) - it's now safe
        regardless of how long this sampler lives, because ReplayBuffer itself
        LRU-bounds the cache at max_cache_entries (see its constructor). The
        `with` block guarantees the cache is freed at the block boundary on
        every exit path (including the early return on insufficient data, or
        an exception), rather than whenever CPython's refcounting happens to
        collect the sampler.
        """
        self.model.train()
        accum_steps = self.minibatch_size // batch_size
        assert self.minibatch_size % batch_size == 0

        progress_bar: Any = range(train_steps)

        policy_loss: torch.Tensor = torch.tensor(0.0)
        value_loss: torch.Tensor = torch.tensor(0.0)

        if __debug__:
            progress_bar = tqdm(progress_bar)

        with self.replay_buffer.get_sampler() as sampler:
            for step in progress_bar:
                states, target_policies, target_values = sampler.sample(
                    self.minibatch_size
                )

                if states.shape[0] < self.minibatch_size:
                    # Not enough data yet — skip training this step.
                    logging.info(
                        f"Not enough data to train yet ({states.shape[0]} < {self.minibatch_size}). Skipping training step."
                    )
                    return global_step

                states = states.to(self.device, non_blocking=True)
                target_policies = target_policies.to(self.device, non_blocking=True)
                target_values = target_values.to(self.device, non_blocking=True)

                for i in range(0, self.minibatch_size, batch_size):
                    s_batch = states[i : i + batch_size]
                    pi_batch = target_policies[i : i + batch_size]
                    v_batch = target_values[i : i + batch_size]

                    with torch.autocast(
                        device_type=self.device.type,
                        enabled=(self.device.type == "cuda"),
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

                if writer is not None and (
                    step % log_interval == 0 or step == train_steps - 1
                ):
                    writer.add_scalar(
                        "loss/policy", policy_loss.item(), global_step + step
                    )
                    writer.add_scalar(
                        "loss/value", value_loss.item(), global_step + step
                    )
                    writer.add_scalar(
                        "loss/total",
                        policy_loss.item() + value_loss.item(),
                        global_step + step,
                    )

                if __debug__:
                    if step % 100 == 0:
                        progress_bar.set_postfix(
                            {
                                "policy loss": f"{policy_loss.item():.4f}",  # pyright: ignore
                                "value loss": f"{value_loss.item():.4f}",  # pyright: ignore
                            }
                        )

        return global_step + train_steps


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
    max_moves: int = 512,
    mcts_batch_size: int = 32,
    mcts_simulations: int = 800,
    log_dir: Optional[str] = None,
    log_interval: int = 50,
):
    game_type, self_play_method = game_data
    game = game_type()

    # max_cache_entries bounds the sparse-to-dense policy cache independently
    # of replay_buffer_size (see ReplayBuffer::dense_policy_cache) - one
    # minibatch's worth caps it at minibatch_size * ~82KB/entry for chess
    # (e.g. ~336MB at minibatch_size=4096), regardless of how large
    # replay_buffer_size is. A cache tied to replay_buffer_size instead grew
    # to capacity * ~82KB/entry (26GB at 320,000 entries) and directly
    # contributed to this project's training OOM kills.
    replay_buffer = ReplayBuffer(
        replay_buffer_size, game_type.action_dim, max_cache_entries=minibatch_size
    )

    network = network_type.load_az_network(
        checkpoint_manager.get_latest_checkpoint_file(), network_device
    )

    # Constructed once and reused for every outer iteration - AlphaZeroTrainer
    # owns the Adam optimizer (and GradScaler), both of which carry running
    # per-parameter state (momentum/variance estimates, loss-scale) that's only
    # useful if it survives across iterations. Recreating the trainer each
    # iteration would throw that state away and restart optimization from
    # Adam's early-warmup behavior every ~15-25 minutes, on every single
    # iteration of the run, not just across process restarts.
    trainer = trainer_factory(network, network_device, replay_buffer, minibatch_size)

    # global_step (and the writer itself) has to live out here to keep
    # TensorBoard's x-axis contiguous across iterations instead of resetting to 0
    # on every one.
    writer = SummaryWriter(log_dir) if log_dir is not None else None
    global_step = 0

    try:
        for iteration in range(loop_iterations):
            self_play_method(
                game=game,
                network_path=checkpoint_manager.get_latest_inference_model_file(),
                replay_buf=replay_buffer,
                num_games=games_in_each_iteration,
                thread_count=thread_count,
                mcts_num_simulations=mcts_simulations,
                mcts_batch_size=mcts_batch_size,
                max_moves=max_moves,
            )

            if writer is not None:
                writer.add_scalar(
                    "self_play/replay_buffer_size", replay_buffer.get_size(), iteration
                )

            global_step = trainer.train(
                batch_size,
                training_iterations,
                writer=writer,
                global_step=global_step,
                log_interval=log_interval,
            )

            checkpoint_manager.add_checkpoint(network)
    finally:
        if writer is not None:
            writer.close()
