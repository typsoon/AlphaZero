import sys
from pathlib import Path

proj_root = Path(__file__).parents[2]
build_dir = proj_root / "build"

sys.path.append(str(build_dir / "training/"))
sys.path.append(str(build_dir / "engine"))

import engine_bind  # pyright: ignore # noqa: E402, F401
import self_play_bind  # pyright: ignore # noqa: E402, F401
