import torch
from injectors import (
    get_network,
    get_trainer,
)
from network import AlphaZeroNetwork
from train import self_play_and_train_loop
import argparse
import os

from pybind import engine_bind
from engine_bind import Connect4  # pyright: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# thread_count = 4
# games_in_each_iteration = 500
training_iterations = 20
minibatch_size = 4096
replay_buffer_size = 1500 * 35


def get_args():
    parser = argparse.ArgumentParser(description="My arg parser")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="AZNetwork.pt",
        help="Path to network file, or AZNetwork for default",
    )

    parser.add_argument(
        "--games-in-each-iteration",
        type=int,
        default=500,
        help="Number of games in each iteration",
    )
    parser.add_argument(
        "--training-iterations",
        type=int,
        default=20,
        help="Number of training iterations",
    )

    parser.add_argument(
        "--loop-iterations", type=int, default=1, help="Number of loop iterations"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Training batch size"
    )

    parser.add_argument("--thread-count", type=int, default=os.cpu_count(), help="Thread count")
    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    args = get_args()

    if not os.path.isfile(args.checkpoint):
        network = get_network(Connect4)
        network.save_az_network(args.checkpoint)

    self_play_and_train_loop(
        AlphaZeroNetwork,
        args.checkpoint,
        network_device=device,
        game_type=Connect4,
        trainer_factory=get_trainer,
        loop_iterations=args.loop_iterations,
        games_in_each_iteration=args.games_in_each_iteration,
        replay_buffer_size=replay_buffer_size,
        training_iterations=training_iterations,
        minibatch_size=minibatch_size,
        batch_size=args.batch_size,
        thread_count=args.thread_count,
    )
