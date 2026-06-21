import json
import os


def save_puzzle(category, name, board, expected_moves):
    # original
    path = os.path.join(category, f"{name}.json")
    with open(path, "w") as f:
        json.dump({"board": board, "expected_moves": expected_moves}, f, indent=2)

    # horizontally flipped
    flipped_board = [row[::-1] for row in board]
    flipped_expected = [6 - m for m in expected_moves]

    path_flipped = os.path.join(category, f"{name}_flipped.json")
    with open(path_flipped, "w") as f:
        json.dump(
            {"board": flipped_board, "expected_moves": flipped_expected}, f, indent=2
        )


def empty_board():
    return [[0] * 7 for _ in range(6)]


# WIN IN 1
# Player 1's turn. Has 3 in a row horizontally.
b = empty_board()
b[5][0] = 1
b[5][1] = 1
b[5][2] = 1
b[4][0] = 2
b[4][1] = 2
b[4][2] = 2
save_puzzle("win_in_1", "horizontal", b, [3])

# Player 1's turn. Has 3 in a row vertically.
b = empty_board()
b[5][0] = 1
b[4][0] = 1
b[3][0] = 1
b[5][1] = 2
b[4][1] = 2
b[3][1] = 2
save_puzzle("win_in_1", "vertical", b, [0])


# PREVENT ENEMY WINNING IN 1
# Player 1's turn. Player 2 has 3 in a row horizontally.
b = empty_board()
b[5][0] = 2
b[5][1] = 2
b[5][2] = 2
b[4][0] = 1
b[4][1] = 1
b[4][3] = 1
save_puzzle("prevent_enemy_winning_in_1", "horizontal", b, [3])

# Player 1's turn. Player 2 has 3 in a row vertically.
b = empty_board()
b[5][6] = 2
b[4][6] = 2
b[3][6] = 2
b[5][5] = 1
b[4][5] = 1
b[3][5] = 1
save_puzzle("prevent_enemy_winning_in_1", "vertical", b, [6])


# FORCED WIN IN 2
# Player 1's turn. Needs to create an unblockable double-threat.
b = empty_board()
b[5][1] = 1
b[5][2] = 1
b[4][1] = 2
b[4][2] = 2
save_puzzle(
    "forced_win_in_2", "double_threat_horiz", b, [3]
)  # playing at 3 gives 1,2,3, forcing 0 or 4 on next turn

b = empty_board()
# Diagonal double threat setup
b[5][1] = 1
b[5][2] = 2
b[5][3] = 2
b[5][4] = 1
b[4][2] = 1
b[4][3] = 2
b[4][4] = 2
b[3][3] = 1
b[3][4] = 1
save_puzzle("forced_win_in_2", "diagonal_setup", b, [4])


# PREVENT ENEMY WINNING IN 2
# Player 1's turn. Player 2 is about to create a double threat.
b = empty_board()
b[5][1] = 2
b[5][2] = 2
b[4][1] = 1
b[4][2] = 1
save_puzzle("prevent_enemy_winning_in_2", "prevent_double_threat_horiz", b, [3])

b = empty_board()
b[5][2] = 2
b[5][3] = 2
b[5][4] = 2
b[5][5] = 1
b[4][3] = 2
b[4][4] = 1
b[3][4] = 2
save_puzzle("prevent_enemy_winning_in_2", "prevent_diagonal_setup", b, [1])

print("Generated all puzzles successfully!")
