"""Validate the correctness of hand-authored Chess puzzles in games/chess/*/*.json.

Uses python-chess (pure-Python chess rules engine) as an independent authority to check
that each puzzle's `expected_moves` are legal and actually achieve the tactical goal
implied by their category (checkmate for mate_in_1, a genuinely forced mate for
mate_in_2, an unrecapturable capture worth real material for win_material, a real
"the opponent otherwise mates in 1" threat for avoid_mate_in_1, and a "deliver mate now
or every other move lets the opponent mate you in 1" race for mate_in_1_or_lose).

Board/action encoding (see performance_evaluation/games/chess puzzle files and
engine/game/chess.cpp for the source of truth):
  - board[row][col]: row 0 = rank 8 (black back rank) .. row 7 = rank 1 (white back rank);
    col 0 = file a .. col 7 = file h.
  - action = (from_square*64 + to_square)*5 + promotion, from/to = row*8+col,
    promotion: 0=none, 1=Q, 2=R, 3=N, 4=B. Matches Chess::encode_action/decode_action.
  - `en_passant` is a dead field for these puzzles (the engine's set_custom_state always
    resets move_count/en_passant_move to 0, so an en-passant capture can never be legal
    for a state loaded this way) - puzzles here always use en_passant=-1.
  - `castling` = [k_mc, r1_mc, r2_mc, K_mc, R1_mc, R2_mc] (lower=black, upper=white;
    k=king, r1=a-rook, r2=h-rook); puzzles here always disable castling (all 1s).
"""

import glob
import json
import sys
from pathlib import Path

import chess

from python.utils import PROJ_ROOT

PIECE_MAP = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6,
}
INV_PIECE_MAP = {v: k for k, v in PIECE_MAP.items()}
INV_PROMO = {0: "", 1: "q", 2: "r", 3: "n", 4: "b"}
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def rc_to_square_name(row, col):
    return f"{chr(ord('a') + col)}{8 - row}"


def action_to_uci(action):
    promo = action % 5
    rest = action // 5
    to_sq = rest % 64
    frm = rest // 64
    r1, c1 = divmod(frm, 8)
    r2, c2 = divmod(to_sq, 8)
    return rc_to_square_name(r1, c1) + rc_to_square_name(r2, c2) + INV_PROMO[promo]


def fen_placement_from_board(board):
    rows = []
    for row in board:
        s, empty = "", 0
        for v in row:
            if v == 0:
                empty += 1
            else:
                if empty:
                    s += str(empty)
                    empty = 0
                s += INV_PIECE_MAP[v]
        if empty:
            s += str(empty)
        rows.append(s)
    return "/".join(rows)


def board_from_puzzle(data):
    placement = fen_placement_from_board(data["board"])
    side = "w" if data["player"] == 0 else "b"
    fen = f"{placement} {side} - - 0 1"
    return chess.Board(fen)


def mate_in_1_moves(board):
    res = []
    for mv in board.legal_moves:
        board.push(mv)
        if board.is_checkmate():
            res.append(mv)
        board.pop()
    return res


def forced_mate(board, plies, attacker):
    if plies == 0:
        return False
    if board.turn == attacker:
        for mv in board.legal_moves:
            board.push(mv)
            ok = board.is_checkmate() or forced_mate(board, plies - 1, attacker)
            board.pop()
            if ok:
                return True
        return False
    legal = list(board.legal_moves)
    if not legal:
        return False
    for mv in legal:
        board.push(mv)
        if not forced_mate(board, plies - 1, attacker):
            board.pop()
            return False
        board.pop()
    return True


def check_mate_in_1(board, expected_ucis):
    for uci in expected_ucis:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            return False, f"{uci} is not legal"
        b = board.copy()
        b.push(mv)
        if not b.is_checkmate():
            return False, f"{uci} does not deliver checkmate"
    return True, "OK"


def check_win_material(board, expected_ucis):
    for uci in expected_ucis:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            return False, f"{uci} is not legal"
        captured = board.piece_at(mv.to_square)
        if captured is None:
            return False, f"{uci} is not a capture"
        if PIECE_VALUES[captured.piece_type] < 3:
            return False, f"{uci} only captures a pawn"
        b = board.copy()
        b.push(mv)
        if b.attackers(b.turn, mv.to_square):
            return False, f"{uci} is recapturable"
    return True, "OK"


def check_mate_in_2(board, expected_ucis):
    if len(expected_ucis) != 1:
        return False, "mate_in_2 puzzles must have exactly one expected first move"
    uci = expected_ucis[0]
    mv = chess.Move.from_uci(uci)
    if mv not in board.legal_moves:
        return False, f"{uci} is not legal"
    attacker = board.turn
    if forced_mate(board, 1, attacker):
        return False, f"position is already mate_in_1, not mate_in_2"
    b = board.copy()
    b.push(mv)
    if b.is_checkmate():
        return False, f"{uci} is already checkmate, this is mate_in_1 not mate_in_2"
    if not forced_mate(b, 2, attacker):
        return False, f"{uci} does not force mate within 2 plies for the opponent"
    return True, "OK"


def check_avoid_mate_in_1(board, expected_ucis):
    for uci in expected_ucis:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            return False, f"{uci} is not legal"
        b = board.copy()
        b.push(mv)
        if b.is_checkmate():
            return False, f"{uci} walks into immediate checkmate"
        if mate_in_1_moves(b):
            return False, f"{uci} still allows a mate-in-1 reply"
    found_losing_move = False
    for mv in board.legal_moves:
        if mv.uci() in expected_ucis:
            continue
        b = board.copy()
        b.push(mv)
        if not b.is_checkmate() and mate_in_1_moves(b):
            found_losing_move = True
            break
    if not found_losing_move:
        return False, "no legal move actually walks into a mate-in-1 (no real threat)"
    return True, "OK"


def check_mate_in_1_or_lose(board, expected_ucis):
    legal = list(board.legal_moves)
    mate_ucis = set(uci for uci in expected_ucis)
    for uci in expected_ucis:
        mv = chess.Move.from_uci(uci)
        if mv not in legal:
            return False, f"{uci} is not legal"
        b = board.copy()
        b.push(mv)
        if not b.is_checkmate():
            return False, f"{uci} does not deliver checkmate"
    for mv in legal:
        if mv.uci() in mate_ucis:
            continue
        b = board.copy()
        b.push(mv)
        if b.is_checkmate() or b.is_stalemate():
            return False, f"{mv.uci()} ends the game instead of allowing a reply"
        if not mate_in_1_moves(b):
            return False, f"{mv.uci()} does not lose to an opponent mate-in-1 reply"
    return True, "OK"


CATEGORY_CHECKS = {
    "mate_in_1": check_mate_in_1,
    "win_material": check_win_material,
    "mate_in_2": check_mate_in_2,
    "avoid_mate_in_1": check_avoid_mate_in_1,
    "mate_in_1_or_lose": check_mate_in_1_or_lose,
}


def process_puzzle(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    category = file_path.parent.name
    checker = CATEGORY_CHECKS.get(category)
    if checker is None:
        return False, f"Unknown category {category}"

    try:
        board = board_from_puzzle(data)
    except Exception as e:  # noqa: BLE001
        return False, f"Failed to build board: {e}"

    if board.is_game_over():
        return False, "Position is already game over"

    expected_ucis = [action_to_uci(a) for a in data.get("expected_moves", [])]
    if not expected_ucis:
        return False, "No expected_moves"

    ok, msg = checker(board, expected_ucis)
    return ok, msg


def main():
    files = [
        Path(p)
        for p in glob.glob(str(PROJ_ROOT / "performance_evaluation" / "games" / "chess" / "*" / "*.json"))
    ]

    all_ok = True
    for file_path in sorted(files):
        ok, msg = process_puzzle(file_path)
        if not ok:
            print(f"FAIL: {file_path}")
            print(f"  Reason: {msg}")
            all_ok = False

    if not all_ok:
        sys.exit(1)
    else:
        print(f"All {len(files)} chess puzzles are correct.")


if __name__ == "__main__":
    main()
