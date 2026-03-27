from agent import AlphaZeroAgent, UserAgent
import argparse

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

    user_agent = UserAgent()

    play_connect4(alpha_zero_agent, user_agent)
