# torch has to be imported before importing pybind libtorch modules otherwise a segfault will be thrown
from python.utils import BUILD_DIR
import torch  # noqa F401
import sys

sys.path.append(str(BUILD_DIR / "training/"))
sys.path.append(str(BUILD_DIR / "engine"))

import engine_bind  # pyright: ignore # noqa: E402, F401
import self_play_bind  # pyright: ignore # noqa: E402, F401

sys.modules[__name__ + ".engine_bind"] = engine_bind
sys.modules[__name__ + ".self_play_bind"] = self_play_bind
sys.modules["pybind.engine_bind"] = engine_bind
sys.modules["pybind.self_play_bind"] = self_play_bind
