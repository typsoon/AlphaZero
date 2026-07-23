from .checkpoint_manager import CheckpointManager
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Callable, Optional, Type

from .network import AlphaZeroNetwork

import sys
import queue
import threading
from tqdm import tqdm as base_tqdm
from tqdm.notebook import tqdm as notebook_tqdm
from torch.utils.tensorboard import SummaryWriter

import logging
from .pybind.engine_bind import Game, ReplayBuffer, StateEncoder  # pyright: ignore
from .replay_persistence import load_buffer_if_compatible, save_buffer


tqdm = notebook_tqdm if "ipykernel" in sys.modules else base_tqdm


class AlphaZeroTrainer:
    def __init__(
        self,
        model: nn.Module,
        replay_buffer: ReplayBuffer,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device("cpu"),
        minibatch_size=4096,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        self.model = torch.compile(model.to(memory_format=torch.channels_last), mode="reduce-overhead")
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.minibatch_size = minibatch_size
        self.device = device
        # Optional LR schedule, stepped once per train() call (i.e. once per
        # outer self-play/train iteration). None keeps the previous fixed-LR
        # behavior. Note: the schedule lives on the optimizer, which is created
        # fresh on every process (re)start, so a restart resets the LR to its
        # initial value and re-runs the decay - acceptable while restarts are
        # ~hours apart and the decay completes within that window.
        self.scheduler = scheduler
        # Networks advertise their value-head format ("scalar" tanh regression
        # by default; chess-v2's is "wdl", 3 win/draw/loss logits trained with
        # cross-entropy against the game outcome). See ChessAzV2Network.
        self.value_is_wdl = getattr(model, "value_head", "scalar") == "wdl"

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
            # Prefetch pipeline: a worker thread samples minibatch N+1 (the
            # CPU-side gather + policy densify into pinned host tensors, with the
            # GIL released in the C++ sample() binding) while the GPU trains on
            # minibatch N, so that per-step CPU sampling is hidden behind compute
            # rather than serialized in front of it. Queue depth 1 stages at most
            # one batch ahead. The already-pinned tensors are moved to the device
            # with a non-blocking DMA copy on the main thread, as before.
            prefetch: queue.Queue = queue.Queue(maxsize=1)
            stop_prefetch = threading.Event()

            def producer() -> None:
                try:
                    while not stop_prefetch.is_set():
                        batch = sampler.sample(self.minibatch_size)
                        # Hand off, but stay responsive to shutdown if the
                        # consumer has already stopped draining the queue.
                        while not stop_prefetch.is_set():
                            try:
                                prefetch.put(batch, timeout=0.5)
                                break
                            except queue.Full:
                                continue
                except Exception as exc:  # surface to the consumer, don't hang
                    prefetch.put(exc)

            worker = threading.Thread(target=producer, daemon=True)
            worker.start()

            try:
                for step in progress_bar:
                    item = prefetch.get()
                    if isinstance(item, Exception):
                        raise item
                    states, target_policies, target_values = item

                    if states.shape[0] < self.minibatch_size:
                        # Not enough data yet — skip training this step.
                        logging.info(
                            f"Not enough data to train yet ({states.shape[0]} < {self.minibatch_size}). Skipping training step."
                        )
                        return global_step

                    states = states.to(self.device, memory_format=torch.channels_last, non_blocking=True)
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
                            if self.value_is_wdl:
                                # WDL nets deploy forward() = (policy, scalar)
                                # for the C++ engine; training needs the raw
                                # WDL logits instead.
                                p_logits, v_preds = self.model.train_outputs(s_batch)
                            else:
                                p_logits, v_preds = self.model(s_batch)

                            # Fused cross-entropy is much faster and more
                            # memory-efficient than manual log_softmax + sum
                            policy_loss = F.cross_entropy(p_logits, pi_batch)
                            if self.value_is_wdl:
                                # v_preds are [B, 3] WDL logits; targets come
                                # as game outcomes z in {1, 0, -1}, mapping to
                                # classes 0=win, 1=draw, 2=loss via 1 - z.
                                wdl_target = (1.0 - v_batch).round().long().clamp(0, 2)
                                value_loss = F.cross_entropy(v_preds, wdl_target)
                            else:
                                value_loss = F.mse_loss(v_preds.squeeze(-1), v_batch)

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
                        writer.add_scalar(
                            "train/lr",
                            self.optimizer.param_groups[0]["lr"],
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
            finally:
                # Stop the producer and let it observe the flag: drain one slot
                # in case it is blocked on a full queue, then join.
                stop_prefetch.set()
                try:
                    prefetch.get_nowait()
                except queue.Empty:
                    pass
                worker.join(timeout=5.0)

        # Advance the LR schedule once per outer iteration (after this call's
        # train_steps optimizer steps), and surface the new LR in the log so a
        # decay experiment is visible without TensorBoard.
        if self.scheduler is not None:
            self.scheduler.step()
            logging.info(
                f"LR after scheduler step: {self.optimizer.param_groups[0]['lr']:.2e}"
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
    fast_mcts_simulations: int = 100,
    full_search_probability: float = 0.25,
    transposition_cache_entries: int = 1_000_000,
    use_gumbel_search: bool = False,
    max_num_considered_actions: int = 16,
    resignation_enabled: bool = False,
    resignation_threshold: float = -0.95,
    resignation_consecutive_moves: int = 3,
    resignation_min_ply: int = 60,
    resignation_disable_probability: float = 0.1,
    fpu_reduction: float = 0.0,
    self_play_network_path: Optional[str] = None,
    self_play_value_network_path: Optional[str] = None,
    # The input encoding self-play feeds into inference and records into
    # training trajectories. None (default) lets the C++ side pick the game's
    # default encoder (ChessEncoderV1 for chess). Must match whatever the
    # network being trained/loaded actually expects - see
    # ChessAzV2Network.stm_plane_index and get_network()'s chess_encoder_history
    # in injectors.py.
    encoder: Optional[StateEncoder] = None,
    # Only meaningful with a frozen generator (self_play_network_path set) whose
    # encoding differs from the trainee's `encoder`: the encoding fed to the
    # generator.
    self_play_encoder: Optional[StateEncoder] = None,
    self_play_value_encoder: Optional[StateEncoder] = None,
    # Cross-run replay-buffer persistence. When replay_buffer_path is set, the
    # buffer is preloaded from it at startup (if a compatible sidecar is present)
    # and re-saved every replay_buffer_save_every iterations plus once on exit,
    # so a restart of the same config resumes instantly instead of refilling from
    # empty self-play. replay_buffer_tag / replay_buffer_state_shape are the
    # compatibility signature the load guard checks (see python/replay_persistence.py).
    replay_buffer_path: Optional[str] = None,
    replay_buffer_save_every: int = 10,
    replay_buffer_tag: Optional[str] = None,
    replay_buffer_state_shape: Optional[list] = None,
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

    # Preload a persisted buffer (same-config restart, or a compatible-encoding
    # prior run) to skip the self-play refill ramp. Refused on any encoding
    # mismatch - see python/replay_persistence.py.
    if replay_buffer_path:
        load_buffer_if_compatible(
            replay_buffer,
            replay_buffer_path,
            replay_buffer_tag or "",
            game_type.action_dim,
            replay_buffer_state_shape,
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

    # Frozen-generator bootstrap: when self_play_network_path is set, EVERY
    # iteration's self-play games come from that fixed engine instead of the
    # latest checkpoint, while training/checkpointing continue normally. Used
    # to warm up a fresh (e.g. new-architecture) net by distilling from a
    # strong frozen net's games: the trainee learns the generator's
    # search-amplified policy + real outcomes instead of climbing up from
    # random self-play. NB the trainee cannot go far BEYOND the frozen
    # generator this way - once it reaches arena parity with the generator,
    # unset this to hand generation over to the trainee (normal AlphaZero).
    if self_play_network_path is not None:
        logging.info(
            f"Self-play games are generated by the FROZEN network "
            f"'{self_play_network_path}' (frozen-generator bootstrap), not by "
            f"the latest checkpoint. Unset self_play_network to return to "
            f"normal self-improvement."
        )

    try:
        for iteration in range(loop_iterations):
            self_play_method(
                game=game,
                network_path=(
                    self_play_network_path
                    if self_play_network_path is not None
                    else checkpoint_manager.get_latest_inference_model_file()
                ),
                replay_buf=replay_buffer,
                num_games=games_in_each_iteration,
                thread_count=thread_count,
                mcts_num_simulations=mcts_simulations,
                mcts_batch_size=mcts_batch_size,
                max_moves=max_moves,
                fast_mcts_num_simulations=fast_mcts_simulations,
                full_search_probability=full_search_probability,
                transposition_cache_entries=transposition_cache_entries,
                use_gumbel_search=use_gumbel_search,
                max_num_considered_actions=max_num_considered_actions,
                resignation_enabled=resignation_enabled,
                resignation_threshold=resignation_threshold,
                resignation_consecutive_moves=resignation_consecutive_moves,
                resignation_min_ply=resignation_min_ply,
                resignation_disable_probability=resignation_disable_probability,
                fpu_reduction=fpu_reduction,
                encoder=encoder,
                self_play_encoder=self_play_encoder,
                value_network_path=self_play_value_network_path or "",
                value_network_encoder=self_play_value_encoder,
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

            # Periodic buffer snapshot so a crash loses at most the games since
            # the last save (not the whole buffer). Atomic write - see
            # replay_persistence.save_buffer.
            if (
                replay_buffer_path
                and replay_buffer_save_every > 0
                and (iteration + 1) % replay_buffer_save_every == 0
            ):
                save_buffer(
                    replay_buffer,
                    replay_buffer_path,
                    replay_buffer_tag or "",
                    game_type.action_dim,
                    replay_buffer_state_shape,
                )
    finally:
        # Best-effort final snapshot on graceful exit (covers a clean stop
        # between save intervals; a hard crash still has the last periodic save).
        if replay_buffer_path:
            try:
                save_buffer(
                    replay_buffer,
                    replay_buffer_path,
                    replay_buffer_tag or "",
                    game_type.action_dim,
                    replay_buffer_state_shape,
                )
            except Exception as e:  # pragma: no cover - shutdown best-effort
                logging.warning(f"Final replay-buffer save failed: {e}")
        if writer is not None:
            writer.close()
