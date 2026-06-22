import torch
from .pybind.engine_bind import Connect4  # pyright: ignore
from .checkpoint_manager import CheckpointManager
from .injectors import (
    get_network,
    get_trainer,
)
from .network import AlphaZeroNetwork
from .train import self_play_and_train_loop
import argparse
import os
import logging
from python.utils import PROJ_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# thread_count = 4
# games_in_each_iteration = 500
replay_buffer_size = 1500 * 35


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

    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    args = get_args()

    manager = CheckpointManager(
        args.checkpoint_stem,
        PROJ_ROOT / args.checkpoint_dir,
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
            network = get_network(Connect4)
    else:
        logging.info(
            "No initial network provided. Initializing a fresh AlphaZero network."
        )
        network = get_network(Connect4)

    manager.add_checkpoint(network)

    logging.info(
        f"Starting AlphaZero training loop: {args.loop_iterations} outer iterations, {args.games_in_each_iteration} games per iter."
    )

    self_play_and_train_loop(
        checkpoint_manager=manager,
        network_type=AlphaZeroNetwork,
        network_device=device,
        game_type=Connect4,
        trainer_factory=get_trainer,
        loop_iterations=args.loop_iterations,
        games_in_each_iteration=args.games_in_each_iteration,
        replay_buffer_size=replay_buffer_size,
        training_iterations=args.training_iterations,
        minibatch_size=args.minibatch_size,
        batch_size=args.batch_size,
        thread_count=args.thread_count,
    )
