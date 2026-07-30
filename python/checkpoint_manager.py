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

    def __init__(
        self, network_name_stem, checkpoint_dir: Path, max_checkpoints, tensorrt_every=1
    ):
        self.checkpoint_count = 0
        self.max_checkpoints = max_checkpoints
        assert self.max_checkpoints > 0, "Max checkpoints must be greater than 0"
        # Compile a TensorRT engine only every `tensorrt_every` checkpoints (0 =
        # never). The .pt (weights) and .pt_scripted (TorchScript) engines are
        # always written - they are cheap and the scripted one is a usable
        # inference fallback. TRT compilation, by contrast, costs tens of seconds
        # per checkpoint and is pure waste during a frozen-generator bootstrap,
        # where self-play runs off the FROZEN generator's engine and nothing
        # consumes the trainee's TRT until it takes over generation (phase B).
        # Set 0 for such a bootstrap (arena/eval can use the scripted engine),
        # 1 for normal self-improvement where self-play needs the freshest TRT.
        self.tensorrt_every = tensorrt_every
        self._trt_counter = 0

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
        # Stable path (not per-checkpoint) so the onnx backend's timing cache
        # (see network.py's _onnx_to_trt_engine) warms up and stays warm
        # across every iteration of this run - the network architecture is
        # constant between iterations (only weights change), so the same
        # kernel-selection decisions apply every time.
        self.tensorrt_timing_cache_path = str(self.tensorrt_dir / "timing_cache.bin")

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

        self._trt_counter += 1
        compile_trt = self.tensorrt_every > 0 and (
            self._trt_counter % self.tensorrt_every == 0
        )
        if torch.cuda.is_available() and compile_trt:
            try:
                import torch_tensorrt  # noqa: F401

                backend = "torch_tensorrt"
            except ImportError:
                # No cp314 wheel for torch_tensorrt. The onnx backend
                # (network.py's backend="onnx") produces a raw TensorRT
                # engine rather than a TorchScript module -
                # get_latest_inference_model_file() below hands anything at
                # the .pt_trt path straight to self-play's loader, and (as of
                # engine/inference/basic_infer.cpp's TensorRTInferenceBackend
                # - see [[native-tensorrt-engine-loading]]) that loader now
                # natively understands this format too when the binary was
                # built with TensorRT SDK headers. Falls back to the plain
                # scripted network automatically if it wasn't.
                backend = "onnx"
            try:
                # Free PyTorch cached memory so TensorRT has enough VRAM to compile
                torch.cuda.empty_cache()
                network.tensorrt_and_save_network(
                    self.tensorrt_name_fmstr % 0,
                    backend=backend,
                    timing_cache_path=self.tensorrt_timing_cache_path,
                )
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
