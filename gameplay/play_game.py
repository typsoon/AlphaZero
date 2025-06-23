import argparse
import torch

from agent import AlphaZeroAgent, UserAgent
from play_game_utils import play_connect4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-path", type=str, required=True, help="Path to network")
    parser.add_argument("--device", type=str, default="cuda", help="Device string, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--user-first", action="store_true", help="Make user play first")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    az_agent = AlphaZeroAgent(args.network_path, torch.device(args.device))
    user_agent = UserAgent()

    if args.user_first:
        play_connect4(user_agent, az_agent)
    else:
        play_connect4(az_agent, user_agent)
