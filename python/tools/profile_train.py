import argparse
import cProfile
import logging
import pstats
from contextlib import nullcontext

import torch
from torch.profiler import ProfilerActivity, profile

from python.injectors import get_network, get_trainer
from python.pybind.engine_bind import Chess, Connect4, ReplayBuffer, Transition  # pyright: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

GAMES = {"connect4": Connect4, "chess": Chess}


def get_args():
    parser = argparse.ArgumentParser(
        description="Profile AlphaZeroTrainer.train() in isolation, using a "
        "replay buffer seeded with synthetic transitions instead of running "
        "self-play (see profile_self_play_* doit tasks for that)."
    )
    parser.add_argument("--game", type=str, default="connect4", choices=list(GAMES))
    parser.add_argument(
        "--network-arch",
        type=str,
        default="legacy",
        choices=["legacy", "chess_v2"],
        help="Architecture of the profiled network - match the run you care "
        "about (the live history-encoder bootstrap is 'chess_v2'). Default "
        "'legacy' keeps the original behavior.",
    )
    parser.add_argument(
        "--chess-encoder-history",
        type=int,
        default=None,
        choices=[1, 4, 8],
        help="chess_v2 only: profile the ChessEncoderV2History(N) input "
        "(14*N+7 planes) instead of the 19-plane default. Sizes both the "
        "network and the synthetic states so the profiled forward/backward "
        "matches the real training workload.",
    )
    parser.add_argument(
        "--replay-size",
        type=int,
        default=8192,
        help="Number of synthetic transitions to seed the replay buffer with",
    )
    parser.add_argument(
        "--training-iterations",
        type=int,
        default=200,
        help="Number of optimizer steps to profile",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument(
        "--kineto-out",
        type=str,
        default="",
        help="If set, wraps the run in PyTorch Kineto and writes a chrome trace "
        "to this path. Left empty, Kineto profiling is skipped.",
    )
    parser.add_argument(
        "--cprofile-out",
        type=str,
        default="",
        help="If set, wraps the run in cProfile (Python-level call profiling) and "
        "writes pstats output to this path. Left empty, cProfile profiling is "
        "skipped.",
    )
    return parser.parse_args()


def make_synthetic_transitions(game_type, count, max_policy_entries=32, state_dim=None):
    """Builds transitions with random states/policies/rewards - real MCTS/self-play
    values aren't needed since only AlphaZeroTrainer.train()'s compute (sampling,
    forward/backward, optimizer step) is being profiled here. state_dim overrides
    game_type.state_dim so a non-default input encoding (e.g. the 63-plane
    ChessEncoderV2History) is profiled at its real channel count."""
    if state_dim is None:
        state_dim = game_type.state_dim
    action_dim = game_type.action_dim
    num_policy_entries = min(max_policy_entries, action_dim)

    transitions = []
    for _ in range(count):
        # ReplayBuffer::sample squeezes off dim 0 (see replay_buffer.cpp), since
        # real states come from get_board_state() with a leading batch dim of 1.
        state = torch.rand(1, *state_dim)
        policy_indices = torch.randperm(action_dim)[:num_policy_entries].to(torch.int64)
        policy_values = torch.softmax(torch.rand(num_policy_entries), dim=0)
        reward = float(torch.empty(1).uniform_(-1, 1).item())
        transitions.append(Transition(state, policy_indices, policy_values, reward))
    return transitions


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_type = GAMES[args.game]

    network = get_network(
        game_type,
        network_arch=args.network_arch,
        chess_encoder_history=args.chess_encoder_history,
    ).to(device)
    # Match the synthetic states to what the network actually consumes: the
    # encoder's channel count, not game_type.state_dim (which is the fixed
    # 19-plane default and would mismatch a history-encoder net).
    if args.chess_encoder_history is not None:
        from python.pybind.engine_bind import ChessEncoderV2History  # pyright: ignore

        state_dim = tuple(
            ChessEncoderV2History(args.chess_encoder_history).state_shape()
        )
    else:
        state_dim = game_type.state_dim
    replay_buffer = ReplayBuffer(
        args.replay_size, game_type.action_dim, max_cache_entries=args.minibatch_size
    )
    replay_buffer.add(
        make_synthetic_transitions(game_type, args.replay_size, state_dim=state_dim)
    )

    trainer = get_trainer(network, device, replay_buffer, args.minibatch_size)

    use_kineto = bool(args.kineto_out)
    profiler_ctx = (
        profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        if use_kineto
        else nullcontext()
    )

    use_cprofile = bool(args.cprofile_out)
    cprofiler = cProfile.Profile() if use_cprofile else nullcontext()

    logger.info("Running 3 warmup steps to let torch.compile and CUDA graphs finish compiling...")
    trainer.train(batch_size=args.batch_size, train_steps=3)

    logger.info(
        f"Profiling {args.training_iterations} training step(s) "
        f"(batch_size={args.batch_size}, minibatch_size={args.minibatch_size})"
    )

    with profiler_ctx as prof, cprofiler as cprof:
        trainer.train(batch_size=args.batch_size, train_steps=args.training_iterations)

    if use_kineto:
        prof.export_chrome_trace(args.kineto_out)
        logger.info(f"Kineto profile saved to {args.kineto_out}")

    if use_cprofile:
        pstats.Stats(cprof).dump_stats(args.cprofile_out)
        logger.info(f"cProfile stats saved to {args.cprofile_out}")

    logger.info("Profiling completed.")


if __name__ == "__main__":
    main()
