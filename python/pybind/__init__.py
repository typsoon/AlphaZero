import sys
from pathlib import Path
import torch

proj_root = Path(__file__).parents[2]
build_dir = proj_root / "build"

sys.path.append(str(build_dir / "training/"))
sys.path.append(str(build_dir / "engine"))

import engine_bind  # pyright: ignore
import self_play_bind  # pyright: ignore
