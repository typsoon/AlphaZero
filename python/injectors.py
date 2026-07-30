from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from .network import AlphaZeroNetwork, ChessAzV2Network
from .train import AlphaZeroTrainer
from typing import Optional, Type

import torch
from torch import nn

from .pybind.engine_bind import (  # pyright: ignore
    ChessEncoderV1,
    ChessEncoderV2History,
    Game,
    ReplayBuffer,
    StateEncoder,
)


# --- LR schedule (config-driven; introduced 2026-07-15, parameterized 2026-07-17)
# Decay the LR exponentially from initial_lr down to lr_lower_bound over
# LR_DECAY_ITERS outer iterations, then hold at lr_lower_bound. Both endpoints
# come from the training config (initial_lr / lr_lower_bound).
#
# Setting initial_lr == lr_lower_bound gives a FLAT LR, and that is usually what
# you want for the ongoing run: the auto-restart loop reloads the already-trained
# net from chess_AZNetwork_0.pt while the scheduler restarts at iteration 0, so a
# fresh initial_lr > lr_lower_bound re-decays from scratch on every restart and
# jolts a converged net into a multi-hour "recovery dip" (each OOM restart cost
# ~1-2 h of re-climbing; a compounded double restart briefly read ~400 Elo down).
# Use a real initial_lr > lr_lower_bound decay only for a fresh-weights /
# plateau-break run.
LR_DECAY_ITERS = 300  # outer iterations to reach the floor (~6-8 h at ~1-2 min/iter)
DEFAULT_INITIAL_LR = 1e-3
DEFAULT_LR_LOWER_BOUND = 1e-4


def _make_lr_multiplier(initial_lr: float, lr_lower_bound: float):
    # LambdaLR scales the optimizer's base LR (== initial_lr) by this factor.
    # factor(0) = 1 -> initial_lr; factor(>=LR_DECAY_ITERS) = ratio -> lr_lower_bound.
    # ratio < 1 decays; ratio == 1 (endpoints equal) holds the LR flat.
    ratio = lr_lower_bound / initial_lr

    def _lr_multiplier(iteration: int) -> float:
        frac = min(iteration, LR_DECAY_ITERS) / LR_DECAY_ITERS
        return ratio**frac

    return _lr_multiplier


def get_chess_encoder(chess_encoder_history: Optional[int] = None) -> StateEncoder:
    """The StateEncoder matching get_network()'s chess_encoder_history choice,
    for callers that construct self-play's `encoder=` argument (see
    __main__.py). None -> ChessEncoderV1 (the engine's own default - passing
    this to self_play()'s encoder= is equivalent to omitting it, but this
    keeps network sizing and self-play's encoder selection driven by the same
    single config value instead of two independently-set ones)."""
    if chess_encoder_history is None:
        return ChessEncoderV1()
    return ChessEncoderV2History(chess_encoder_history)


def get_network(
    game: Type[Game],
    resblock_filter_size=64,
    residual_block_count=10,
    network_arch: str = "legacy",
    # Chess-v2 only: selects ChessEncoderV2History(history=chess_encoder_history)
    # instead of the default ChessEncoderV1, sizing the network's input_channels
    # and stm_plane_index (see ChessAzV2Network) to match. Must be 1, 4, or 8
    # (ChessEncoderV2History's own supported values) or None for the default
    # single-frame v1 encoding. The matching encoder OBJECT for self-play is
    # built separately by get_chess_encoder() with the same argument - see
    # __main__.py, which passes both consistently.
    chess_encoder_history: Optional[int] = None,
):
    state_dim = game.state_dim
    action_dim = game.action_dim

    if network_arch == "chess_v2":
        # Fixed-size chess-only architecture (12x128 SE blocks, spatial policy,
        # WDL value head - see ChessAzV2Network); the legacy size knobs above
        # do not apply to it.
        if chess_encoder_history is not None:
            encoder = ChessEncoderV2History(chess_encoder_history)
            input_channels, _, _ = encoder.state_shape()
            stm_plane_index = chess_encoder_history * 14
        else:
            input_channels, stm_plane_index = state_dim[0], 12
        return ChessAzV2Network(
            input_channels=input_channels,
            height=state_dim[1],
            width=state_dim[2],
            action_size=action_dim,
            stm_plane_index=stm_plane_index,
        )
    if network_arch != "legacy":
        raise ValueError(f"unknown network_arch: {network_arch!r}")

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
    initial_lr: float = DEFAULT_INITIAL_LR,
    lr_lower_bound: float = DEFAULT_LR_LOWER_BOUND,
    value_loss_weight: float = 1.0,
) -> AlphaZeroTrainer:
    optimizer = Adam(
        model.parameters(),
        lr=initial_lr,
        weight_decay=1e-4,
        fused=(device.type == "cuda"),
    )
    scheduler = LambdaLR(optimizer, _make_lr_multiplier(initial_lr, lr_lower_bound))

    return AlphaZeroTrainer(
        model,
        replay_buffer,
        optimizer,
        device,
        minibatch_size,
        scheduler=scheduler,
        value_loss_weight=value_loss_weight,
    )
