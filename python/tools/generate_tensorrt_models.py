import argparse
import logging
from pathlib import Path
import torch
from python.network import AlphaZeroNetwork
from python.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate TensorRT models for all existing checkpoints"
    )
    parser.add_argument(
        "--max_first_dim",
        type=int,
        default=None,
        help="Max first dim of input (max TensorRT batch size). Defaults to "
        "AlphaZeroNetwork.tensorrt_and_save_network's own default if not set.",
    )
    parser.add_argument(
        "--opt_first_dim",
        type=int,
        default=None,
        help="Optimization-point first dim of input (the batch size TensorRT "
        "actually tunes kernel selection for) - should match the real median "
        "DynamicBatcher batch size on the deployment machine, not an arbitrary "
        "value. Defaults to AlphaZeroNetwork.tensorrt_and_save_network's own "
        "default if not set.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Root directory containing per-game checkpoint subdirectories. "
        "Defaults to 'checkpoints'.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["torch_tensorrt", "onnx"],
        help="Compilation backend to pass to tensorrt_and_save_network. "
        "'torch_tensorrt' (default) produces a TorchScript-wrapped .pt_trt "
        "loaded via libtorch; 'onnx' produces a raw TensorRT engine (faster "
        "to compile, needs no torch_tensorrt, but must be loaded by a "
        "TensorRT runtime, not torch.jit.load). Defaults to "
        "AlphaZeroNetwork.tensorrt_and_save_network's own default if not set.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Only scripted models will be generated.")

    checkpoints_dir = Path(args.checkpoint_dir)
    if not checkpoints_dir.exists():
        logger.info("No checkpoints directory found: %s", checkpoints_dir)
        return

    # Iterate over all game directories in checkpoints
    for game_dir in checkpoints_dir.iterdir():
        if not game_dir.is_dir():
            continue
        logger.info("Processing game: %s", game_dir.name)

        some_files_were_processed = False

        for file_path in game_dir.iterdir():
            if file_path.is_dir():
                continue

            some_files_were_processed = True
            logger.info("  Loading checkpoint: %s", file_path.name)

            try:
                # Load the raw checkpoint
                network = AlphaZeroNetwork.load_az_network(file_path, device)

                # Extract the index from the filename to save it correctly
                # E.g., chess_AZNetwork_0.pt -> 0
                trt_path = (
                    game_dir
                    / CheckpointManager.tensorrt_dir_name
                    / f"{file_path.stem}{CheckpointManager.tensorrt_checkpoint_suffix}"
                )

                if torch.cuda.is_available():
                    logger.info(
                        "    Compiling and saving TensorRT model: %s",
                        Path(trt_path).name,
                    )
                    try:
                        kwargs = {}
                        if args.max_first_dim is not None:
                            kwargs["max_first_dim_of_input"] = args.max_first_dim
                        if args.opt_first_dim is not None:
                            kwargs["opt_first_dim_of_input"] = args.opt_first_dim
                        if args.backend is not None:
                            kwargs["backend"] = args.backend
                        network.tensorrt_and_save_network(trt_path, **kwargs)
                    except Exception as e:
                        logger.error(
                            "    Failed to compile TRT engine: %s",
                            e,
                        )
            except Exception as e:
                logger.error("  Failed to process %s: %s", file_path.name, e)

        if not some_files_were_processed:
            logger.info("No raw .pt checkpoints found for %s", game_dir.name)
            continue


if __name__ == "__main__":
    main()
