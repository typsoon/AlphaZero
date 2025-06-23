import argparse

from agent import AlphaZeroAgent, UserAgent
from play_game_utils import play_connect4


def parse_args():
    parser = argparse.ArgumentParser(description="Play Connect-4 against AlphaZero AI.")
    parser.add_argument(
        "--socket",
        type=str,
        default="/tmp/alphazero.sock",
        help="Path to inference server Unix socket",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    alpha_zero_agent = AlphaZeroAgent(socket_path=parsed_args.socket)

    az_agent = AlphaZeroAgent(args.network_path, torch.device(args.device))
    user_agent = UserAgent()

    if args.user_first:
        play_connect4(user_agent, az_agent)
    else:
        play_connect4(az_agent, user_agent)
