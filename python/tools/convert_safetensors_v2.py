"""Converts an engine-zoo chess-v2 safetensors checkpoint (12x128 SE-ResNet,
spatial policy, WDL value head) into this repo's ChessAzV2Network format.

The two formats share almost identical parameter names (both were built from
the same architecture - see [[chess-v2-network-port]]), except for two things
the safetensors file never contains, since they aren't learned weights:
  - _map_white/_map_black: the canonical-flip action-index buffers, computed
    by ChessAzV2Network's own constructor from the fixed action space.
  - Each BatchNorm's num_batches_tracked counter.
Both come from a freshly constructed network instead of the checkpoint.

Usage:
    python -m python.tools.convert_safetensors_v2 \
        ~/generation-000100.safetensors \
        mateusz_champions/best_v2/best_v2 \
        --chess-encoder-history 4

Writes <output_stem>.pt, <output_stem>.pt_scripted, and (if CUDA is
available) <output_stem>.pt_trt.
"""

import argparse
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file

from python.network import ChessAzV2Network

logger = logging.getLogger(__name__)


def convert(
    safetensors_path: Path, chess_encoder_history: int, device: torch.device
) -> ChessAzV2Network:
    source_sd = load_file(str(safetensors_path))

    stem_weight = source_sd.get("stem_conv.weight")
    if stem_weight is None:
        raise ValueError(
            f"'{safetensors_path}' has no 'stem_conv.weight' - doesn't look like "
            "an engine-zoo chess-v2 checkpoint"
        )
    input_channels = stem_weight.shape[1]
    expected_channels = 14 * chess_encoder_history + 7
    if input_channels != expected_channels:
        raise ValueError(
            f"stem_conv.weight has {input_channels} input channels, but "
            f"--chess-encoder-history {chess_encoder_history} implies "
            f"{expected_channels} (=14*history+7) - pass the history depth "
            "this checkpoint was actually trained with."
        )

    net = ChessAzV2Network(
        input_channels=input_channels,
        stm_plane_index=14 * chess_encoder_history,
    ).to(device)

    target_sd = net.state_dict()
    computed_only = {"_map_white", "_map_black"}
    num_batches_tracked = {k for k in target_sd if k.endswith(".num_batches_tracked")}
    expected_from_source = set(target_sd) - computed_only - num_batches_tracked

    missing = expected_from_source - set(source_sd)
    unexpected = set(source_sd) - expected_from_source
    if missing:
        raise ValueError(f"safetensors file is missing expected keys: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"safetensors file has unexpected keys: {sorted(unexpected)}")

    for key in expected_from_source:
        target_sd[key] = source_sd[key].to(dtype=target_sd[key].dtype, device=device)

    net.load_state_dict(target_sd, strict=True)
    net.eval()
    return net


def main():
    parser = argparse.ArgumentParser(
        description="Convert an engine-zoo chess-v2 safetensors checkpoint into "
        "this repo's ChessAzV2Network format."
    )
    parser.add_argument("safetensors_path", type=Path)
    parser.add_argument(
        "output_stem",
        type=Path,
        help="Output path stem, e.g. mateusz_champions/best_v2/best_v2 - writes "
        "<stem>.pt, <stem>.pt_scripted, and (if CUDA is available) <stem>.pt_trt",
    )
    parser.add_argument(
        "--chess-encoder-history",
        type=int,
        default=4,
        choices=[1, 4, 8],
        help="History depth this checkpoint was trained with - determines the "
        "input channel count (14*history+7) and the side-to-move plane index "
        "(14*history). Default 4, matching the current live run.",
    )
    parser.add_argument(
        "--skip-tensorrt",
        action="store_true",
        help="Skip compiling a TensorRT engine even if CUDA is available.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Converting %s (device=%s)", args.safetensors_path, device)
    net = convert(args.safetensors_path, args.chess_encoder_history, device)

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)

    pt_path = args.output_stem.with_suffix(".pt")
    net.save_az_network(pt_path)
    logger.info("Saved %s", pt_path)

    scripted_path = args.output_stem.with_suffix(".pt_scripted")
    net.script_and_save_network(scripted_path)
    logger.info("Saved %s", scripted_path)

    if device.type == "cuda" and not args.skip_tensorrt:
        trt_path = args.output_stem.with_suffix(".pt_trt")
        try:
            net.tensorrt_and_save_network(trt_path, backend="onnx")
            logger.info("Saved %s", trt_path)
        except Exception as e:
            logger.warning(
                "TensorRT compile failed (checkpoint is still usable via "
                ".pt_scripted): %s",
                e,
            )


if __name__ == "__main__":
    main()
