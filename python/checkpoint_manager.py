import shutil
from pathlib import Path
from .network import AlphaZeroNetwork
import torch


class CheckpointManager:
    checkpoint_suffix = ".pt"
    scripted_checkpoint_suffix = ".pt_scripted"
    tensorrt_checkpoint_suffix = ".pt_trt"
    tensorrt_dir_name = "tensorrt"
    scripted_dir_name = "scripted"

    def __init__(self, network_name_stem, checkpoint_dir: Path, max_checkpoints):
        self.checkpoint_count = 0
        self.max_checkpoints = max_checkpoints
        assert self.max_checkpoints > 0, "Max checkpoints must be greater than 0"

        self.checkpoint_dir = checkpoint_dir
        self.scripted_dir = checkpoint_dir / self.scripted_dir_name
        self.tensorrt_dir = checkpoint_dir / self.tensorrt_dir_name

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.scripted_dir.mkdir(parents=True, exist_ok=True)
        self.tensorrt_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_name_fmstr = str(
            self.checkpoint_dir / f"{network_name_stem}_%d{self.checkpoint_suffix}"
        )
        self.scripted_name_fmstr = str(
            self.scripted_dir
            / f"{network_name_stem}_%d{self.scripted_checkpoint_suffix}"
        )
        self.tensorrt_name_fmstr = str(
            self.tensorrt_dir
            / f"{network_name_stem}_%d{self.tensorrt_checkpoint_suffix}"
        )

        # Detect existing checkpoints to properly resume
        for i in range(self.max_checkpoints):
            if Path(self.checkpoint_name_fmstr % i).exists():
                self.checkpoint_count += 1
            else:
                break

    def add_checkpoint(self, network: AlphaZeroNetwork):
        if self.checkpoint_count < self.max_checkpoints:
            self.checkpoint_count += 1

        for i in range(self.checkpoint_count - 2, -1, -1):
            if Path(self.checkpoint_name_fmstr % i).exists():
                shutil.move(
                    self.checkpoint_name_fmstr % i, self.checkpoint_name_fmstr % (i + 1)
                )
            if Path(self.scripted_name_fmstr % i).exists():
                shutil.move(
                    self.scripted_name_fmstr % i, self.scripted_name_fmstr % (i + 1)
                )
            if Path(self.tensorrt_name_fmstr % i).exists():
                shutil.move(
                    self.tensorrt_name_fmstr % i, self.tensorrt_name_fmstr % (i + 1)
                )

        network.save_az_network(self.checkpoint_name_fmstr % 0)
        network.script_and_save_network(self.scripted_name_fmstr % 0)

        if torch.cuda.is_available():
            try:
                network.tensorrt_and_save_network(self.tensorrt_name_fmstr % 0)
            except Exception as e:
                print(f"Failed to compile TensorRT engine: {e}")

    def get_latest_checkpoint_file(self):
        if self.checkpoint_count <= 0:
            raise Exception("No checkpoints have been saved")

        answer = Path(self.checkpoint_name_fmstr % 0)
        assert answer.exists()
        return str(answer)

    def get_latest_inference_model_file(self):
        if self.checkpoint_count <= 0:
            raise Exception("No checkpoints have been saved")

        trt_path = Path(self.tensorrt_name_fmstr % 0)
        if trt_path.exists():
            return str(trt_path)

        scripted_path = Path(self.scripted_name_fmstr % 0)
        assert scripted_path.exists(), "Neither TRT nor Scripted checkpoint exists"
        return str(scripted_path)
