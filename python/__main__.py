import torch
from .pybind.engine_bind import Connect4, Chess  # pyright: ignore
from .pybind.self_play_bind import self_play_connect4, self_play  # pyright: ignore
from .checkpoint_manager import CheckpointManager
from .injectors import (
    get_network,
    get_trainer,
)
from .network import AlphaZeroNetwork
from .train import self_play_and_train_loop
import argparse
import os
import shutil
import logging
from datetime import datetime
from pathlib import Path
from python.utils import PROJ_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# thread_count = 4
# games_in_each_iteration = 500
# Sized to retain ~4 iterations of games_in_each_iteration=1000 chess games
# (run_training.sh's setting), at an estimated ~80 plies/game - we don't have an
# exact transitions-per-iteration figure (get_size() saturates at capacity, so
# it can't tell us how many transitions self-play actually produced past that
# point). With the sparse ReplayBuffer (~5KB/transition instead of ~85KB dense),
# this is only ~1.5GB, well inside the ~6-10GB of headroom measured during
# self-play's peak memory usage - if the real average game length turns out
# longer than 80 plies, there's room to raise this further.
replay_buffer_size = 1000 * 80 * 4


def prune_old_runs(runs_root: Path, max_runs: int) -> None:
    """Deletes the oldest auto-named run directories under runs_root, keeping at
    most max_runs. Run directory names are timestamps (%Y%m%d-%H%M%S), so
    lexicographic sort is also chronological order. Unlike checkpoints (bounded
    by CheckpointManager), nothing else caps how many of these accumulate across
    repeated invocations - left unchecked they grow forever."""
    if max_runs <= 0 or not runs_root.is_dir():
        return
    run_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir())
    for old_dir in run_dirs[:-max_runs] if len(run_dirs) > max_runs else []:
        shutil.rmtree(old_dir, ignore_errors=True)


def get_args():
    parser = argparse.ArgumentParser(description="My arg parser")

    parser.add_argument(
        "--initial-network",
        type=str,
        default=None,
        help="Path to an existing network to initialize from if no checkpoints exist",
    )
    parser.add_argument(
        "--checkpoint_stem",
        type=str,
        default="AZNetwork",
        help="Checkpoint file prefix",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="connect4",
        choices=["connect4", "chess"],
        help="Which game to train",
    )

    parser.add_argument(
        "--games-in-each-iteration",
        type=int,
        default=400,
        help="Number of games in each iteration",
    )
    parser.add_argument(
        "--training-iterations",
        type=int,
        default=2000,
        help="Number of training iterations",
    )

    parser.add_argument(
        "--loop-iterations", type=int, default=100, help="Number of loop iterations"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=4096, help="Training minibatch size"
    )

    parser.add_argument(
        "--thread-count", type=int, default=os.cpu_count(), help="Thread count"
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to store historical checkpoints",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=5,
        help="Maximum number of historical checkpoints to keep",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=512,
        help="Maximum number of moves before a game is truncated",
    )
    parser.add_argument(
        "--mcts-batch-size",
        type=int,
        default=32,
        help="MCTS batch size for neural network inference",
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=800,
        help="Number of MCTS simulations per move",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="TensorBoard log directory. Defaults to runs/<game>/<timestamp>. "
        "Pass an empty string to disable TensorBoard logging.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=5,
        help="Maximum number of auto-named TensorBoard runs to keep per game "
        "(oldest are deleted). Only applies when --log-dir is left at its "
        "default; ignored if you pass an explicit --log-dir. 0 disables pruning.",
    )

    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    args = get_args()

    manager = CheckpointManager(
        f"{args.game}_{args.checkpoint_stem}",
        PROJ_ROOT / args.checkpoint_dir / args.game,
        args.max_checkpoints,
    )

    if args.initial_network:
        if os.path.isfile(args.initial_network):
            logging.info(f"Initializing network from '{args.initial_network}'.")
            network = AlphaZeroNetwork.load_az_network(args.initial_network, device)
        else:
            logging.warning(
                f"Initial network file '{args.initial_network}' not found. Initializing a fresh AlphaZero network."
            )
        if args.game == "connect4":
            network = get_network(Connect4)
        else:
            network = get_network(Chess)
    else:
        logging.info(
            "No initial network provided. Initializing a fresh AlphaZero network."
        )
        if args.game == "connect4":
            network = get_network(Connect4)
        else:
            network = get_network(Chess)

    manager.add_checkpoint(network)

    logging.info(
        f"Starting AlphaZero training loop: {args.loop_iterations} outer iterations, {args.games_in_each_iteration} games per iter."
    )

    if args.game == "connect4":
        game_data = (Connect4, self_play_connect4)
    else:
        game_data = (Chess, self_play)

    if args.log_dir == "":
        log_dir = None
    elif args.log_dir is not None:
        log_dir = args.log_dir
    else:
        runs_root = PROJ_ROOT / "runs" / args.game
        prune_old_runs(runs_root, args.max_runs)
        log_dir = str(runs_root / datetime.now().strftime("%Y%m%d-%H%M%S"))

    if log_dir is not None:
        logging.info(
            f"Logging to TensorBoard at '{log_dir}'. Run `tensorboard --logdir {log_dir}` to monitor training."
        )

    self_play_and_train_loop(
        checkpoint_manager=manager,
        network_type=AlphaZeroNetwork,
        network_device=device,
        game_data=game_data,
        trainer_factory=get_trainer,
        loop_iterations=args.loop_iterations,
        games_in_each_iteration=args.games_in_each_iteration,
        replay_buffer_size=replay_buffer_size,
        training_iterations=args.training_iterations,
        minibatch_size=args.minibatch_size,
        batch_size=args.batch_size,
        thread_count=args.thread_count,
        max_moves=args.max_moves,
        mcts_batch_size=args.mcts_batch_size,
        mcts_simulations=args.mcts_simulations,
        log_dir=log_dir,
    )
