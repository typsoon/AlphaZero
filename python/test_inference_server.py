import subprocess
import pytest
import os


def test_inference_server_missing_network():
    """
    Test that the inference server gracefully handles a missing network file
    and exits with a clean error code rather than crashing.
    """
    binary_path = "build/inference_server/inference_server"
    if not os.path.exists(binary_path):
        pytest.skip("inference_server binary not found. Skipping test.")

    # Run the binary with a non-existent network
    result = subprocess.run(
        [
            binary_path,
            "--network-path",
            "this_file_does_not_exist.pt",
            "--socket",
            "/tmp/dummy.sock",
        ],
        capture_output=True,
        text=True,
    )

    # It should fail gracefully, meaning a non-zero exit code but NOT a segfault (which would be < 0 usually)
    assert result.returncode == 1, "Expected exit code 1 for graceful failure"
    assert (
        "File this_file_does_not_exist.pt doesn't exist" in result.stderr
        or "File this_file_does_not_exist.pt doesn't exist" in result.stdout
    )
    assert (
        "Network file not found" in result.stderr
        or "Network file not found" in result.stdout
    )
