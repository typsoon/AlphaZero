import json
import os

base = "../performance_evaluation/games/connect4"


def write_json(path, data):
    with open(os.path.join(base, path), "w") as f:
        json.dump(data, f, indent=2)


write_json(
    "prevent_enemy_winning_in_1/horizontal.json",
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
    "prevent_enemy_winning_in_1/horizontal_flipped.json",
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
    "forced_win_in_2/diagonal_setup.json",
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
    "forced_win_in_2/diagonal_setup_flipped.json",
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
    "prevent_enemy_winning_in_2/prevent_diagonal_setup.json",
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
    "prevent_enemy_winning_in_2/prevent_diagonal_setup_flipped.json",
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
