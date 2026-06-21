# torch has to be imported before importing pybind libtorch modules otherwise a segfault will be thrown
import torch  # noqa F401
import sys
from pathlib import Path

proj_root = Path(__file__).parents[2]
build_dir = proj_root / "build"

sys.path.append(str(build_dir / "training/"))
sys.path.append(str(build_dir / "engine"))

import engine_bind  # pyright: ignore # noqa: E402, F401
import self_play_bind  # pyright: ignore # noqa: E402, F401

sys.modules[__name__ + ".engine_bind"] = engine_bind
sys.modules[__name__ + ".self_play_bind"] = self_play_bind
sys.modules["pybind.engine_bind"] = engine_bind
sys.modules["pybind.self_play_bind"] = self_play_bind
