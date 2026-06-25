import json
import sys
from python.utils import PROJ_ROOT


def get_legal_moves(board):
    return [c for c in range(7) if board[0][c] == 0]


def make_move(board, col, player):
    new_board = [row[:] for row in board]
    for r in range(5, -1, -1):
        if new_board[r][col] == 0:
            new_board[r][col] = player
            return new_board
    return new_board


def check_win(board, player):
    # horizontal
    for r in range(6):
        for c in range(4):
            if (
                board[r][c] == player
                and board[r][c + 1] == player
                and board[r][c + 2] == player
                and board[r][c + 3] == player
            ):
                return True
    # vertical
    for r in range(3):
        for c in range(7):
            if (
                board[r][c] == player
                and board[r + 1][c] == player
                and board[r + 2][c] == player
                and board[r + 3][c] == player
            ):
                return True
    # diagonal \
    for r in range(3):
        for c in range(4):
            if (
                board[r][c] == player
                and board[r + 1][c + 1] == player
                and board[r + 2][c + 2] == player
                and board[r + 3][c + 3] == player
            ):
                return True
    # diagonal /
    for r in range(3):
        for c in range(3, 7):
            if (
                board[r][c] == player
                and board[r + 1][c - 1] == player
                and board[r + 2][c - 2] == player
                and board[r + 3][c - 3] == player
            ):
                return True
    return False


def get_win_in_1(board, player):
    if check_win(board, 3 - player):
        return []
    wins = []
    for m in get_legal_moves(board):
        b1 = make_move(board, m, player)
        if check_win(b1, player):
            wins.append(m)
    return wins


def get_forced_win_in_2(board, player):
    if check_win(board, 3 - player):
        return []
    if len(get_win_in_1(board, player)) > 0:
        return []

    forcing_moves = []
    for m1 in get_legal_moves(board):
        b1 = make_move(board, m1, player)
        if check_win(b1, player):
            continue

        opp = 3 - player
        opp_replies = get_legal_moves(b1)
        if len(opp_replies) == 0:
            continue

        is_forced_win = True
        for m2 in opp_replies:
            b2 = make_move(b1, m2, opp)
            if check_win(b2, opp):
                is_forced_win = False
                break
            if len(get_win_in_1(b2, player)) == 0:
                is_forced_win = False
                break

        if is_forced_win:
            forcing_moves.append(m1)

    return forcing_moves


def get_prevent_enemy_winning_in_1(board, player):
    opp = 3 - player
    if len(get_win_in_1(board, opp)) == 0:
        return None
    if len(get_win_in_1(board, player)) > 0:
        return None

    preventing_moves = []
    for m in get_legal_moves(board):
        b1 = make_move(board, m, player)
        if len(get_win_in_1(b1, opp)) == 0:
            preventing_moves.append(m)
    return preventing_moves


def get_prevent_enemy_winning_in_2(board, player):
    opp = 3 - player
    if len(get_forced_win_in_2(board, opp)) == 0:
        return None
    if len(get_win_in_1(board, player)) > 0:
        return None

    preventing_moves = []
    for m in get_legal_moves(board):
        b1 = make_move(board, m, player)
        if len(get_win_in_1(b1, opp)) == 0 and len(get_forced_win_in_2(b1, opp)) == 0:
            preventing_moves.append(m)
    return preventing_moves


def is_valid_board(board):
    p1 = sum(row.count(1) for row in board)
    p2 = sum(row.count(2) for row in board)
    if p1 != p2 and p1 != p2 + 1:
        return False, f"Piece count mismatch: p1={p1}, p2={p2}"

    for r in range(5):
        for c in range(7):
            if board[r][c] != 0 and board[r + 1][c] == 0:
                return False, f"Floating piece at ({r}, {c})"

    return True, ""


def current_player(board):
    p1 = sum(row.count(1) for row in board)
    p2 = sum(row.count(2) for row in board)
    return 1 if p1 == p2 else 2


def process_puzzle(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    board = data["board"]
    expected_moves = set(data.get("expected_moves", []))

    valid, msg = is_valid_board(board)
    if not valid:
        return False, msg, None

    category = file_path.parent.name
    player = current_player(board)

    match category:
        case "win_in_1":
            actual = get_win_in_1(board, player)
            if set(actual) != expected_moves:
                return (
                    False,
                    f"win_in_1 expected {expected_moves}, got {set(actual)}",
                    actual,
                )
        case "forced_win_in_2":
            actual = get_forced_win_in_2(board, player)
            if set(actual) != expected_moves:
                return (
                    False,
                    f"forced_win_in_2 expected {expected_moves}, got {set(actual)}",
                    actual,
                )
        case "prevent_enemy_winning_in_1":
            actual = get_prevent_enemy_winning_in_1(board, player)
            if actual is None:
                return False, "Enemy is not threatening a win in 1", None
            if set(actual) != expected_moves:
                return (
                    False,
                    f"prevent_enemy_winning_in_1 expected {expected_moves}, got {set(actual)}",
                    actual,
                )
        case "prevent_enemy_winning_in_2":
            actual = get_prevent_enemy_winning_in_2(board, player)
            if actual is None:
                return False, "Enemy is not threatening a forced win in 2", None
            if set(actual) != expected_moves:
                return (
                    False,
                    f"prevent_enemy_winning_in_2 expected {expected_moves}, got {set(actual)}",
                    actual,
                )
        case "positional_puzzles":
            return True, "OK", None
        case _:
            return False, f"Unknown category {category}", None

    return True, "OK", None


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-fix", action="store_true")
    args = parser.parse_args()

    files = list(
        (PROJ_ROOT / "performance_evaluation" / "games" / "connect4").glob("*/*.json")
    )
    all_ok = True
    for file_path in files:
        ok, msg, actual = process_puzzle(file_path)
        if not ok:
            print(f"FAIL: {file_path}")
            print(f"  Reason: {msg}")

            if args.auto_fix and actual is not None:
                print(f"  Auto-fixing {file_path} to expected_moves={actual}")
                with open(file_path, "r") as f:
                    data = json.load(f)
                data["expected_moves"] = sorted(list(actual))
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2)
                    f.write("\n")
            else:
                all_ok = False
        else:
            # print(f"PASS: {file_path}")
            pass

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
