import json
from pathlib import Path
from python.utils import PROJ_ROOT

base = PROJ_ROOT / "performance_evaluation" / "games" / "connect4"


def write_json(path, data):
    target = base / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(data, f, indent=2)


write_json(
    Path("prevent_enemy_winning_in_1") / "horizontal.json",
    {
        "board": [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [2, 2, 2, 0, 0, 0, 0],
        ],
        "expected_moves": [3],
    },
)

write_json(
    Path("prevent_enemy_winning_in_1") / "horizontal_flipped.json",
    {
        "board": [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 2, 2, 2],
        ],
        "expected_moves": [3],
    },
)

write_json(
    Path("forced_win_in_2") / "diagonal_setup.json",
    {
        "board": [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [2, 0, 1, 1, 0, 0, 2],
        ],
        "expected_moves": [4],
    },
)

write_json(
    Path("forced_win_in_2") / "diagonal_setup_flipped.json",
    {
        "board": [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 1, 1, 0, 2],
        ],
        "expected_moves": [2],
    },
)

write_json(
    Path("prevent_enemy_winning_in_2") / "prevent_diagonal_setup.json",
    {
        "board": [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 2, 2, 0, 0, 1],
        ],
        "expected_moves": [1, 4, 5],
    },
)

write_json(
    Path("prevent_enemy_winning_in_2") / "prevent_diagonal_setup_flipped.json",
    {
        "board": [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 2, 2, 0, 1],
        ],
        "expected_moves": [1, 2, 5],
    },
)
