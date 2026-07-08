import subprocess
from pathlib import Path

PROJ_ROOT = Path(__file__).parents[2]
BUILD_DIR = PROJ_ROOT / "build"

SUPPORTED_GAMES = (("connect4", "Connect4"), ("chess", "Chess"))

NETWORK_PARAM = {
    "name": "network_path",
    "long": "network_path",
    "type": str,
    "default": "",
    "help": "Path to the scripted network file. Defaults to checkpoints/<game>/scripted/AZNetwork_0.pt_scripted",
}


def resolve_network_path(network_path, game="connect4"):
    from python.checkpoint_manager import CheckpointManager

    if not network_path:
        game_dir = PROJ_ROOT / "checkpoints" / game
        file_name = f"{game}_AZNetwork_0.pt_trt"

        trt_path = game_dir / CheckpointManager.tensorrt_dir_name / file_name
        if trt_path.exists():
            return str(trt_path)

        scripted_path = game_dir / CheckpointManager.scripted_dir_name / file_name
        if scripted_path.exists():
            return str(scripted_path)

        return str(
            PROJ_ROOT
            / "checkpoints"
            / game
            / "scripted"
            / f"{game}_AZNetwork_0.pt_scripted"
        )
    return network_path


def run_protected(cmd):
    """Run a shell command and wait for it to finish even if Ctrl+C is pressed, allowing children to exit cleanly."""
    p = subprocess.Popen(cmd, shell=True)
    try:
        p.wait()
    except KeyboardInterrupt:
        # Give the child process (like perf) a chance to handle SIGINT and flush data
        p.wait()
    return p.returncode == 0
