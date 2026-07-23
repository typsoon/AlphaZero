"""Function-preserving network deepening (Net2DeeperNet).

Takes an existing AlphaZero checkpoint and produces a DEEPER one (more residual
blocks) that computes the IDENTICAL function at save time: the appended blocks
have their second BatchNorm's gamma/beta zero-initialized, so each new block
reduces to relu(X + 0) = X (the residual tower's activations are already
post-relu, hence non-negative). Training then grows the new blocks' gamma off
zero and the extra capacity comes online gradually - no strength reset, unlike
starting a fresh bigger net.

Width (filter count) canNOT be grown this way; changing filters requires a
fresh network.

Usage:
    python -m python.tools.deepen_network \
        --input checkpoints/chess/chess_AZNetwork_0.pt \
        --output /tmp/deepened.pt --extra-blocks 10
"""

import argparse
import logging

import torch
from torch import nn

from python.network import AlphaZeroNetwork

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def deepen(net: AlphaZeroNetwork, extra_blocks: int) -> AlphaZeroNetwork:
    old_blocks = len(net.residual_blocks)
    deeper = AlphaZeroNetwork(
        input_channels=net.conv_in.in_channels,
        height=net._height,
        width=net._width,
        num_residual_blocks=old_blocks + extra_blocks,
        action_size=net.policy_fc.out_features,
        num_filters=net.conv_in.out_channels,
    )
    # Copies every module the two nets share (stem, blocks 0..old-1, both
    # heads); the appended blocks are absent from the source state_dict and
    # keep their fresh init, which we then neutralize below.
    missing, unexpected = deeper.load_state_dict(net.state_dict(), strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert all(k.startswith("residual_blocks.") for k in missing), (
        f"missing keys outside the appended blocks: {missing}"
    )

    for block in deeper.residual_blocks[old_blocks:]:
        nn.init.zeros_(block.bn2.weight)
        nn.init.zeros_(block.bn2.bias)
    return deeper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Deepened checkpoint (.pt)")
    parser.add_argument(
        "--extra-blocks", type=int, default=10, help="Residual blocks to append"
    )
    args = parser.parse_args()

    net = AlphaZeroNetwork.load_az_network(args.input, torch.device("cpu")).eval()
    deeper = deepen(net, args.extra_blocks).eval()

    # Function-preservation check: identical outputs on random input.
    with torch.no_grad():
        x = torch.rand(8, net.conv_in.in_channels, net._height, net._width)
        p0, v0 = net(x)
        p1, v1 = deeper(x)
    dp = (p0 - p1).abs().max().item()
    dv = (v0 - v1).abs().max().item()
    logging.info(
        f"deepened {len(net.residual_blocks)} -> {len(deeper.residual_blocks)} "
        f"blocks; function-preservation max|dpolicy|={dp:.2e} max|dvalue|={dv:.2e}"
    )
    assert dp < 1e-5 and dv < 1e-5, "deepened net does not preserve the function!"

    deeper.save_az_network(args.output)
    logging.info(f"saved deepened checkpoint to {args.output}")


if __name__ == "__main__":
    main()
