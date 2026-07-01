import unittest
import torch
import shutil
from pathlib import Path
from python.network import AlphaZeroNetwork
from python.checkpoint_manager import CheckpointManager


class TestTensorRTCompilation(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_checkpoints")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @unittest.skipIf(
        not torch.cuda.is_available(), "CUDA is required for TensorRT compilation"
    )
    def test_tensorrt_compilation(self):
        # Create a tiny network
        net = AlphaZeroNetwork(
            input_channels=3,
            height=6,
            width=7,
            num_residual_blocks=1,
            action_size=7,
            num_filters=16,
        )

        manager = CheckpointManager("AZNetwork", self.test_dir, max_checkpoints=1)

        # Add checkpoint, which should trigger script and TRT compilation
        manager.add_checkpoint(net)

        # Verify the structure
        self.assertTrue((self.test_dir / "scripted").exists())
        self.assertTrue((self.test_dir / "tensorrt").exists())

        # Verify the files exist
        scripted_file = Path(manager.get_latest_inference_model_file())
        self.assertTrue(scripted_file.exists())
        self.assertEqual(scripted_file.suffix, ".pt_trt")

        # Verify we can load the tensorrt model in PyTorch
        import torch_tensorrt  # noqa: F401

        trt_model = torch.jit.load(scripted_file)

        # Run a dummy forward pass
        dummy_input = torch.randn(2, 3, 6, 7).cuda()
        policy, value = trt_model(dummy_input)

        self.assertEqual(policy.shape, (2, 7))
        self.assertEqual(value.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
