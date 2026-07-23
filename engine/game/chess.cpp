// Original source: https://github.com/geochri/AlphaZero_Chess/blob/master/src/chess_board.py
#include "chess.hpp"
#include "bitboard.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

namespace {

// One random 64-bit key per (square, piece type), plus side-to-move, castling-rights
// and en-passant-file keys - the standard Zobrist hashing scheme used to identify
// chess positions for threefold-repetition purposes. Keys are generated once, with a
// fixed seed so hashes (and therefore repetition counts) are reproducible across runs
// - nothing here needs to be cryptographically random, just well-distributed and
// stable for the lifetime of the process.
struct ZobristKeys {
    uint64_t piece[8][8][12]; // NOLINT(*-avoid-c-arrays)
    uint64_t side_to_move;
    // 0=white kingside, 1=white queenside, 2=black kingside, 3=black queenside
    uint64_t castling[4];        // NOLINT(*-avoid-c-arrays)
    uint64_t en_passant_file[8]; // NOLINT(*-avoid-c-arrays)

    ZobristKeys() // NOLINT(cppcoreguidelines-pro-type-member-init)
    {
        // Fixed seed is intentional (see comment above) - not meant to be
        // unpredictable.
        // NOLINTNEXTLINE(*-msc32-c,*-msc51-cpp,bugprone-random-generator-seed)
        std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
        for (auto &plane : piece)
            for (auto &row : plane)
                for (auto &key : row)
                    key = rng();
        side_to_move = rng();
        for (auto &key : castling)
            key = rng();
        for (auto &key : en_passant_file)
            key = rng();
    }
};

const ZobristKeys &zobrist_keys() {
    static const ZobristKeys keys;
    return keys;
}

// Maps piece values (-6..-1, 1..6; 0/EMPTY is never looked up) onto a dense 0..11
// index: white pieces 0..5, black pieces 6..11.
int zobrist_piece_index(int8_t piece) {
    return piece > 0 ? (piece - 1) : (6 + (-piece - 1));
}

} // namespace

Chess::Chess() {
    Chess::reset();
}

uint64_t Chess::compute_position_hash() const {
    const auto &z = zobrist_keys();
    uint64_t h = 0;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            int8_t p = current_board[r][c]; // NOLINT(*-avoid-c-arrays)
            if (p != EMPTY)
                h ^= z.piece[r][c][zobrist_piece_index(p)];
        }
    }
    if (player == 1)
        h ^= z.side_to_move;
    if (K_move_count == 0 && R2_move_count == 0)
        h ^= z.castling[0]; // White kingside
    if (K_move_count == 0 && R1_move_count == 0)
        h ^= z.castling[1]; // White queenside
    if (k_move_count == 0 && r2_move_count == 0)
        h ^= z.castling[2]; // Black kingside
    if (k_move_count == 0 && r1_move_count == 0)
        h ^= z.castling[3]; // Black queenside
    // Mirrors the exact condition move_rules_P/move_rules_p use to decide whether an
    // en-passant capture is legal right now (only true for the single ply right after
    // the double pawn push) - a position isn't "the same" for repetition purposes if
    // one occurrence has a capturable en-passant pawn and the other doesn't.
    if (en_passant != -1 && std::abs(en_passant_move - move_count) == 1)
        h ^= z.en_passant_file[en_passant];
    return h;
}

bool Chess::is_fifty_move_draw() const {
    return halfmove_clock >= 100;
}

bool Chess::is_threefold_repetition() const {
    return repetition_count >= 3;
}

static constexpr Chess::board_t INITIAL_BOARD = {
    {{B_ROOK, B_KNIGHT, B_BISHOP, B_QUEEN, B_KING, B_BISHOP, B_KNIGHT, B_ROOK},
     {B_PAWN, B_PAWN, B_PAWN, B_PAWN, B_PAWN, B_PAWN, B_PAWN, B_PAWN},
     {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
     {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
     {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
     {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
     {W_PAWN, W_PAWN, W_PAWN, W_PAWN, W_PAWN, W_PAWN, W_PAWN, W_PAWN},
     {W_ROOK, W_KNIGHT, W_BISHOP, W_QUEEN, W_KING, W_BISHOP, W_KNIGHT, W_ROOK}}};

void Chess::reset() {
    current_board = INITIAL_BOARD;

    move_count = 0;
    en_passant = -1;
    en_passant_move = 0;
    r1_move_count = 0;
    r2_move_count = 0;
    k_move_count = 0;
    R1_move_count = 0;
    R2_move_count = 0;
    K_move_count = 0;
    player = 0;

    halfmove_clock = 0;
    position_history.clear();
    position_history.push_back(compute_position_hash());
    repetition_count = 1;
    history_count = 0;
}

bool Chess::is_white(int8_t p) {
    return p > 0;
}
bool Chess::is_black(int8_t p) {
    return p < 0;
}

int Chess::encode_action(const ChessAction<> &a) {
    int from = (a.r1 * 8) + a.c1;
    int to = (a.r2 * 8) + a.c2;
    return (from * 64 + to) * 5 + a.promotion;
}

ChessAction<> Chess::decode_action(int a) {
    ChessAction<> act{};
    act.promotion = a % 5;
    a /= 5;
    auto to = a % 64;
    a /= 64;
    auto from = a;
    act.r1 = from / 8;
    act.c1 = from % 8;
    act.r2 = to / 8;
    act.c2 = to % 8;
    return act;
}

int Chess::getActionSize() const {
    return 64 * 64 * 5;
}

std::vector<int> Chess::get_legal_actions() const {
    auto acts = actions();
    std::vector<int> res;
    res.reserve(acts.size());
    for (const auto &a : acts) {
        res.push_back(encode_action(a));
    }
    return res;
}

void Chess::step(int action) {
    // TEMP DIAGNOSTIC (env-gated, zero-cost when unset): verify every stepped
    // action is actually legal in this position, to catch the source of the
    // illegal moves behind the move_rules_P/p "pawn already on promotion rank"
    // guard storms and the glibc heap-corruption aborts seen in training. Dumps
    // the full position + the offending action, then aborts so a debugger/core
    // shows the caller.
    static const bool verify_legality = std::getenv("ALPHAZERO_VERIFY_STEP_LEGALITY") != nullptr;
    if (verify_legality) {
        auto legal = get_legal_actions();
        if (std::find(legal.begin(), legal.end(), action) == legal.end()) {
            ChessAction<> bad = decode_action(action);
            std::string board_dump;
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 8; ++c)
                    board_dump += std::to_string((int)current_board[r][c]) + " ";
                board_dump += "| ";
            }
            std::string legal_dump;
            for (int la : legal)
                legal_dump += std::to_string(la) + " ";
            spdlog::critical("ILLEGAL STEP: action={} (r1={} c1={} r2={} c2={} promo={}) player={} "
                             "move_count={} en_passant={} en_passant_move={} halfmove_clock={} "
                             "board=[{}] legal_actions=[{}]",
                             action, (int)bad.r1, (int)bad.c1, (int)bad.r2, (int)bad.c2,
                             (int)bad.promotion, (int)player, move_count, (int)en_passant,
                             en_passant_move, halfmove_clock, board_dump, legal_dump);
            std::abort();
        }
    }
    ChessAction<> a = decode_action(action);
    int promo = (player == 0) ? W_QUEEN : B_QUEEN;
    switch (a.promotion) {
    case 1:
        promo = (player == 0) ? W_QUEEN : B_QUEEN;
        break;
    case 2:
        promo = (player == 0) ? W_ROOK : B_ROOK;
        break;
    case 3:
        promo = (player == 0) ? W_KNIGHT : B_KNIGHT;
        break;
    case 4:
        promo = (player == 0) ? W_BISHOP : B_BISHOP;
        break;
    default:
        break;
    }
    move_piece(a.r1, a.c1, a.r2, a.c2, promo);
}

void Chess::set_custom_state(const board_t &board, int8_t active_player, int8_t en_passant_col,
                             int16_t k_mc, int16_t r1_mc, int16_t r2_mc, int16_t K_mc,
                             int16_t R1_mc, int16_t R2_mc) {
    current_board = board;
    player = active_player;
    en_passant = en_passant_col;
    en_passant_move = 0; // Reset or leave as 0 unless needed
    move_count = 0;      // Same, start from 0 for the puzzle state
    k_move_count = k_mc;
    r1_move_count = r1_mc;
    r2_move_count = r2_mc;
    K_move_count = K_mc;
    R1_move_count = R1_mc;
    R2_move_count = R2_mc;

    // Puzzle/custom positions start with no prior game history to draw on - matches
    // move_count's existing "start fresh" treatment just above.
    halfmove_clock = 0;
    position_history.clear();
    position_history.push_back(compute_position_hash());
    repetition_count = 1;
    history_count = 0;
}

bool Chess::is_terminal() const {
    // Cheap checks first: is_fifty_move_draw() is O(1) and is_threefold_repetition()
    // is now O(1) too (repetition_count is maintained incrementally in move_piece()),
    // both far cheaper than actions(true)'s full legal-move search, which simulates
    // check-avoidance for every candidate move it generates.
    if (is_fifty_move_draw() || is_threefold_repetition())
        return true;
    return actions(true).empty();
}

int Chess::get_current_player() const {
    return player;
}

float Chess::reward() const {
    if (!is_terminal())
        return 0.0f;
    // No-legal-moves takes precedence over the automatic draw rules in determining
    // the *outcome*: a checkmated (or stalemated) position is decided by that fact
    // alone, even if the fifty-move clock or a repetition count also happens to have
    // been reached on this exact move - the game already ended via checkmate before
    // either automatic-draw rule would even matter.
    if (actions(true).empty())
        return check_status() ? -1.0f : 0.0f; // checkmate vs stalemate
    return 0.0f;                              // fifty-move-rule or threefold-repetition draw
}

std::shared_ptr<const GameState> Chess::get_canonical_state() const {
    if (weak_from_this().expired()) {
        // If the game object is on the stack (not managed by shared_ptr),
        // return an aliasing shared_ptr with a no-op deleter to prevent crashes
        // while avoiding heap allocation/copying.
        return {this, [](const GameState *) {}};
    }
    return shared_from_this();
}

std::shared_ptr<Game> Chess::clone() const {
    return std::make_shared<Chess>(*this);
}

void Chess::render() const {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            auto p = current_board[i][j];
            char c = ' ';
            switch (p) {
            case W_PAWN:
                c = 'P';
                break;
            case W_KNIGHT:
                c = 'N';
                break;
            case W_BISHOP:
                c = 'B';
                break;
            case W_ROOK:
                c = 'R';
                break;
            case W_QUEEN:
                c = 'Q';
                break;
            case W_KING:
                c = 'K';
                break;
            case B_PAWN:
                c = 'p';
                break;
            case B_KNIGHT:
                c = 'n';
                break;
            case B_BISHOP:
                c = 'b';
                break;
            case B_ROOK:
                c = 'r';
                break;
            case B_QUEEN:
                c = 'q';
                break;
            case B_KING:
                c = 'k';
                break;
            default:
                break;
            }
            std::cout << c << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

Chess::board_t Chess::get_board_state() const {
    return current_board;
}

// The neural-network input encoding used to live here (write_canonical_state /
// get_state_shape); it has moved to ChessEncoderV1 (engine/game/chess_encoder.cpp),
// which reads this class's position via friendship. See engine/game/state_encoder.hpp.

void Chess::move_rules_P(int8_t i, int8_t j, PosList &moves) const {
    const auto &board_state = current_board;
    if (i == 6) {
        if (board_state[i - 1][j] == EMPTY) {
            moves.emplace_back(i - 1, j);
            if (board_state[i - 2][j] == EMPTY)
                moves.emplace_back(i - 2, j);
        }
    } else if (i == 3 && en_passant != -1) {
        if (j - 1 == en_passant && std::abs(en_passant_move - move_count) == 1)
            moves.emplace_back(i - 1, j - 1);
        else if (j + 1 == en_passant && std::abs(en_passant_move - move_count) == 1)
            moves.emplace_back(i - 1, j + 1);
    }
    if ((i == 1 || i == 2 || i == 3 || i == 4 || i == 5) && board_state[i - 1][j] == EMPTY)
        moves.emplace_back(i - 1, j);
    // Guard board_state[i - 1][...] below against i == 0: a white pawn should
    // never actually be sitting on row 0 (it must have promoted into a
    // different piece the move it got there - see move_piece()'s promotion
    // handling), but if that invariant is ever violated for any reason, this
    // would otherwise silently read board_state[-1] - i.e. current_board[-1],
    // which wraps to an enormous out-of-bounds std::array index (UB, not
    // caught by anything at this optimization level). Logged once since this
    // is only ever expected to fire on a genuine bug upstream, not in normal
    // operation - see the identical guard's comment in move_rules_p() below
    // for the (already-confirmed-reachable) black-side mirror of this.
    if (i == 0) {
        spdlog::error("move_rules_P called with a white pawn already on row 0 (should have "
                      "promoted) - skipping its diagonal-capture generation to avoid an "
                      "out-of-bounds board_state[-1] read");
        return;
    }
    if (j == 0 && is_black(board_state[i - 1][j + 1])) {
        moves.emplace_back(i - 1, j + 1);
    } else if (j == 7 && is_black(board_state[i - 1][j - 1])) {
        moves.emplace_back(i - 1, j - 1);
    } else if (j >= 1 && j <= 6) {
        if (is_black(board_state[i - 1][j + 1]))
            moves.emplace_back(i - 1, j + 1);
        if (is_black(board_state[i - 1][j - 1]))
            moves.emplace_back(i - 1, j - 1);
    }
}

void Chess::move_rules_p(int8_t i, int8_t j, PosList &moves) const {
    const auto &board_state = current_board;
    if (i == 1) {
        if (board_state[i + 1][j] == EMPTY) {
            moves.emplace_back(i + 1, j);
            if (board_state[i + 2][j] == EMPTY)
                moves.emplace_back(i + 2, j);
        }
    } else if (i == 4 && en_passant != -1) {
        if (j - 1 == en_passant && std::abs(en_passant_move - move_count) == 1)
            moves.emplace_back(i + 1, j - 1);
        else if (j + 1 == en_passant && std::abs(en_passant_move - move_count) == 1)
            moves.emplace_back(i + 1, j + 1);
    }
    if ((i == 2 || i == 3 || i == 4 || i == 5 || i == 6) && board_state[i + 1][j] == EMPTY)
        moves.emplace_back(i + 1, j);
    // Guard board_state[i + 1][...] below against i == 7: unlike the forward-push
    // check above (explicitly restricted to i in [2,6]), these diagonal-capture
    // reads had no bound on i at all - if a black pawn is ever found on row 7
    // (it should always have promoted into a different piece the move it got
    // there; see move_piece()'s promotion handling), board_state[i + 1] =
    // board_state[8] silently reads 8 bytes of whatever memory follows
    // current_board (board_t is this class's last member - see its declaration)
    // instead of being caught by any bounds check, since std::array::operator[]
    // performs none. That stray read can pass is_white()/is_black() by chance,
    // fabricating a move to a nonexistent row-8 square - encode_action() then
    // has no way to detect that square is invalid (its formula assumes r1/r2/c1/
    // c2 are already in [0,7]), producing an action index that decodes back to a
    // *different*, coincidentally in-range-looking (from, to) pair entirely -
    // this was confirmed as the root cause of a rare "BAD ACTION INDEX"
    // CUDA gather() crash during real training (see execute_tensor_batch() in
    // engine/inference/basic_infer.cpp), traced here via a temporary
    // RAW BAD CHESSACTION diagnostic that caught exactly board_state[8]'s
    // fingerprint (r2=8) at its source.
    if (i == 7) {
        spdlog::error("move_rules_p called with a black pawn already on row 7 (should have "
                      "promoted) - skipping its diagonal-capture generation to avoid an "
                      "out-of-bounds board_state[8] read");
        return;
    }
    if (j == 0 && is_white(board_state[i + 1][j + 1])) {
        moves.emplace_back(i + 1, j + 1);
    } else if (j == 7 && is_white(board_state[i + 1][j - 1])) {
        moves.emplace_back(i + 1, j - 1);
    } else if (j >= 1 && j <= 6) {
        if (is_white(board_state[i + 1][j + 1]))
            moves.emplace_back(i + 1, j + 1);
        if (is_white(board_state[i + 1][j - 1]))
            moves.emplace_back(i + 1, j - 1);
    }
}

void Chess::move_rules_r(int8_t i, int8_t j, PosList &moves) const {
    const auto &board_state = current_board;
    int8_t a = 0;
    int8_t b = 0;

    a = i;
    while (a != 0) {
        if (board_state[a - 1][j] != EMPTY) {
            if (is_white(board_state[a - 1][j]))
                moves.emplace_back(a - 1, j);
            break;
        }
        moves.emplace_back(a - 1, j);
        a -= 1;
    }
    a = i;
    while (a != 7) {
        if (board_state[a + 1][j] != EMPTY) {
            if (is_white(board_state[a + 1][j]))
                moves.emplace_back(a + 1, j);
            break;
        }
        moves.emplace_back(a + 1, j);
        a += 1;
    }
    a = j;
    while (a != 7) {
        if (board_state[i][a + 1] != EMPTY) {
            if (is_white(board_state[i][a + 1]))
                moves.emplace_back(i, a + 1);
            break;
        }
        moves.emplace_back(i, a + 1);
        a += 1;
    }
    a = j;
    while (a != 0) {
        if (board_state[i][a - 1] != EMPTY) {
            if (is_white(board_state[i][a - 1]))
                moves.emplace_back(i, a - 1);
            break;
        }
        moves.emplace_back(i, a - 1);
        a -= 1;
    }
}

void Chess::move_rules_R(int8_t i, int8_t j, PosList &moves) const {
    const auto &board_state = current_board;
    int8_t a = 0;
    int8_t b = 0;

    a = i;
    while (a != 0) {
        if (board_state[a - 1][j] != EMPTY) {
            if (is_black(board_state[a - 1][j]))
                moves.emplace_back(a - 1, j);
            break;
        }
        moves.emplace_back(a - 1, j);
        a -= 1;
    }
    a = i;
    while (a != 7) {
        if (board_state[a + 1][j] != EMPTY) {
            if (is_black(board_state[a + 1][j]))
                moves.emplace_back(a + 1, j);
            break;
        }
        moves.emplace_back(a + 1, j);
        a += 1;
    }
    a = j;
    while (a != 7) {
        if (board_state[i][a + 1] != EMPTY) {
            if (is_black(board_state[i][a + 1]))
                moves.emplace_back(i, a + 1);
            break;
        }
        moves.emplace_back(i, a + 1);
        a += 1;
    }
    a = j;
    while (a != 0) {
        if (board_state[i][a - 1] != EMPTY) {
            if (is_black(board_state[i][a - 1]))
                moves.emplace_back(i, a - 1);
            break;
        }
        moves.emplace_back(i, a - 1);
        a -= 1;
    }
}

void Chess::move_rules_b(int8_t i, int8_t j, PosList &moves) const {
    const auto &board_state = current_board;
    int8_t a = 0;
    int8_t b = 0;

    a = i;
    b = j;
    while (a != 0 && b != 0) {
        if (board_state[a - 1][b - 1] != EMPTY) {
            if (is_white(board_state[a - 1][b - 1]))
                moves.emplace_back(a - 1, b - 1);
            break;
        }
        moves.emplace_back(a - 1, b - 1);
        a -= 1;
        b -= 1;
    }
    a = i;
    b = j;
    while (a != 7 && b != 7) {
        if (board_state[a + 1][b + 1] != EMPTY) {
            if (is_white(board_state[a + 1][b + 1]))
                moves.emplace_back(a + 1, b + 1);
            break;
        }
        moves.emplace_back(a + 1, b + 1);
        a += 1;
        b += 1;
    }
    a = i;
    b = j;
    while (a != 0 && b != 7) {
        if (board_state[a - 1][b + 1] != EMPTY) {
            if (is_white(board_state[a - 1][b + 1]))
                moves.emplace_back(a - 1, b + 1);
            break;
        }
        moves.emplace_back(a - 1, b + 1);
        a -= 1;
        b += 1;
    }
    a = i;
    b = j;
    while (a != 7 && b != 0) {
        if (board_state[a + 1][b - 1] != EMPTY) {
            if (is_white(board_state[a + 1][b - 1]))
                moves.emplace_back(a + 1, b - 1);
            break;
        }
        moves.emplace_back(a + 1, b - 1);
        a += 1;
        b -= 1;
    }
}

void Chess::move_rules_B(int8_t i, int8_t j, PosList &moves) const {
    const auto &board_state = current_board;
    int8_t a = 0;
    int8_t b = 0;

    a = i;
    b = j;
    while (a != 0 && b != 0) {
        if (board_state[a - 1][b - 1] != EMPTY) {
            if (is_black(board_state[a - 1][b - 1]))
                moves.emplace_back(a - 1, b - 1);
            break;
        }
        moves.emplace_back(a - 1, b - 1);
        a -= 1;
        b -= 1;
    }
    a = i;
    b = j;
    while (a != 7 && b != 7) {
        if (board_state[a + 1][b + 1] != EMPTY) {
            if (is_black(board_state[a + 1][b + 1]))
                moves.emplace_back(a + 1, b + 1);
            break;
        }
        moves.emplace_back(a + 1, b + 1);
        a += 1;
        b += 1;
    }
    a = i;
    b = j;
    while (a != 0 && b != 7) {
        if (board_state[a - 1][b + 1] != EMPTY) {
            if (is_black(board_state[a - 1][b + 1]))
                moves.emplace_back(a - 1, b + 1);
            break;
        }
        moves.emplace_back(a - 1, b + 1);
        a -= 1;
        b += 1;
    }
    a = i;
    b = j;
    while (a != 7 && b != 0) {
        if (board_state[a + 1][b - 1] != EMPTY) {
            if (is_black(board_state[a + 1][b - 1]))
                moves.emplace_back(a + 1, b - 1);
            break;
        }
        moves.emplace_back(a + 1, b - 1);
        a += 1;
        b -= 1;
    }
}

void Chess::move_rules_q(int8_t i, int8_t j, PosList &moves) const {
    move_rules_r(i, j, moves);
    move_rules_b(i, j, moves);
}

void Chess::move_rules_Q(int8_t i, int8_t j, PosList &moves) const {
    move_rules_R(i, j, moves);
    move_rules_B(i, j, moves);
}

void Chess::move_rules_n(int8_t i, int8_t j, PosList &moves) const {
    const auto &board_state = current_board;
    const pos_t steps[] = {{i + 2, j - 1}, {i + 2, j + 1}, {i + 1, j - 2}, {i - 1, j - 2},
                           {i - 2, j + 1}, {i - 2, j - 1}, {i - 1, j + 2}, {i + 1, j + 2}};
    for (auto p : steps) {
        auto [a, b] = p;
        if (0 <= a && a <= 7 && 0 <= b && b <= 7) {
            if (is_white(board_state[a][b]) || board_state[a][b] == EMPTY) {
                moves.emplace_back(a, b);
            }
        }
    }
}

void Chess::move_rules_N(int8_t i, int8_t j, PosList &moves) const {
    const auto &board_state = current_board;
    const pos_t steps[] = {{i + 2, j - 1}, {i + 2, j + 1}, {i + 1, j - 2}, {i - 1, j - 2},
                           {i - 2, j + 1}, {i - 2, j - 1}, {i - 1, j + 2}, {i + 1, j + 2}};
    for (auto p : steps) {
        auto [a, b] = p;
        if (0 <= a && a <= 7 && 0 <= b && b <= 7) {
            if (is_black(board_state[a][b]) || board_state[a][b] == EMPTY) {
                moves.emplace_back(a, b);
            }
        }
    }
}

void Chess::move_rules_k(PosList &moves) const {
    int i = -1;
    int j = -1;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            if (current_board[r][c] == B_KING) {
                i = r;
                j = c;
                break;
            }
        }
        if (i != -1)
            break;
    }
    if (i == -1)
        return;
    const pos_t steps[] = {{i + 1, j},     {i - 1, j},     {i, j + 1},     {i, j - 1},
                           {i + 1, j + 1}, {i - 1, j - 1}, {i + 1, j - 1}, {i - 1, j + 1}};
    for (auto p : steps) {
        auto [a, b] = p;
        if (0 <= a && a <= 7 && 0 <= b && b <= 7) {
            auto sq = current_board[a][b];
            bool ok = false;
            switch (sq) {
            case EMPTY:
            case W_QUEEN:
            case W_BISHOP:
            case W_KNIGHT:
            case W_PAWN:
            case W_ROOK:
                ok = true;
                break;
            default:
                break;
            }
            if (ok) {
                moves.emplace_back(a, b);
            }
        }
    }
    if (can_castle(0) && !check_status())
        moves.emplace_back(0, 2);
    if (can_castle(1) && !check_status())
        moves.emplace_back(0, 6);
}

void Chess::move_rules_K(PosList &moves) const {
    int i = -1;
    int j = -1;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            if (current_board[r][c] == W_KING) {
                i = r;
                j = c;
                break;
            }
        }
        if (i != -1)
            break;
    }
    if (i == -1)
        return;
    const pos_t steps[] = {{i + 1, j},     {i - 1, j},     {i, j + 1},     {i, j - 1},
                           {i + 1, j + 1}, {i - 1, j - 1}, {i + 1, j - 1}, {i - 1, j + 1}};
    for (auto p : steps) {
        auto [a, b] = p;
        if (0 <= a && a <= 7 && 0 <= b && b <= 7) {
            auto sq = current_board[a][b];
            bool ok = false;
            switch (sq) {
            case EMPTY:
            case B_QUEEN:
            case B_BISHOP:
            case B_KNIGHT:
            case B_PAWN:
            case B_ROOK:
                ok = true;
                break;
            default:
                break;
            }
            if (ok) {
                moves.emplace_back(a, b);
            }
        }
    }
    if (can_castle(0) && !check_status())
        moves.emplace_back(7, 2);
    if (can_castle(1) && !check_status())
        moves.emplace_back(7, 6);
}

void Chess::move_piece(int r1, int c1, int r2, int c2, int promoted_piece) {
    // Snapshot progress (pawn move or capture) before anything is mutated below -
    // both branches read/clear these exact squares, so this check is the same
    // regardless of which player is moving. En-passant capture is the one case where
    // the destination square is empty but a piece is still captured (mirrors the
    // exact condition each branch below uses to remove the captured pawn).
    bool moved_pawn = (current_board[r1][c1] == W_PAWN || current_board[r1][c1] == B_PAWN);
    bool is_capture = current_board[r2][c2] != EMPTY ||
                      (moved_pawn && std::abs(c1 - c2) == 1 && current_board[r2][c2] == EMPTY);

    // Push the PRE-move position into the history-stacked-encoder window (see
    // history_boards' comment in chess.hpp) before anything below mutates
    // current_board. repetition_count - 1 converts "occurrences including
    // itself" (this class's existing FIDE-facing counter) to "occurrences
    // before this one" (what the encoder wants), matching engine-zoo's
    // current_repetitions_before semantics; clamped to [0,2] since that's all
    // the encoder's two repetition-flag planes distinguish.
    for (int i = kMaxHistoryFrames - 1; i > 0; --i) {
        history_boards[i] = history_boards[i - 1];
        history_repetitions_before[i] = history_repetitions_before[i - 1];
    }
    history_boards[0] = current_board;
    history_repetitions_before[0] =
        static_cast<int8_t>(std::min(2, std::max(0, repetition_count - 1)));
    history_count = std::min(history_count + 1, kMaxHistoryFrames);

    if (player == 0) {
        auto promoted = false;
        auto piece = current_board[r1][c1];
        current_board[r1][c1] = EMPTY;
        if (piece == W_ROOK && r1 == 7 && c1 == 0)
            R1_move_count++;
        if (piece == W_ROOK && r1 == 7 && c1 == 7)
            R2_move_count++;
        if (piece == W_KING) {
            K_move_count++;
            // Castling is encoded as a plain 2-square king move (see move_rules_K) -
            // the corresponding rook has to be relocated by hand here since it isn't
            // part of the (r1,c1)->(r2,c2) move being applied.
            if (c2 - c1 == 2) {
                current_board[7][7] = EMPTY;
                current_board[7][5] = W_ROOK;
                R2_move_count++;
            } else if (c2 - c1 == -2) {
                current_board[7][0] = EMPTY;
                current_board[7][3] = W_ROOK;
                R1_move_count++;
            }
        }
        if (piece == W_PAWN) {
            if (std::abs(r1 - r2) > 1) {
                en_passant = c1;
                en_passant_move = move_count;
            }
            if (std::abs(c1 - c2) == 1 && current_board[r2][c2] == EMPTY)
                current_board[r2 + 1][c2] = EMPTY;
            if (r2 == 0) {
                bool is_valid_promo = false;
                switch (promoted_piece) {
                case W_ROOK:
                case W_BISHOP:
                case W_KNIGHT:
                case W_QUEEN:
                    is_valid_promo = true;
                    break;
                default:
                    break;
                }
                if (is_valid_promo) {
                    current_board[r2][c2] = promoted_piece;
                    promoted = true;
                }
            }
        }
        if (!promoted)
            current_board[r2][c2] = piece;
        player = 1;
        move_count++;
    } else {
        bool promoted = false;
        auto piece = current_board[r1][c1];
        current_board[r1][c1] = EMPTY;
        if (piece == B_ROOK && r1 == 0 && c1 == 0)
            r1_move_count++;
        if (piece == B_ROOK && r1 == 0 && c1 == 7)
            r2_move_count++;
        if (piece == B_KING) {
            k_move_count++;
            // Castling is encoded as a plain 2-square king move (see move_rules_k) -
            // the corresponding rook has to be relocated by hand here since it isn't
            // part of the (r1,c1)->(r2,c2) move being applied.
            if (c2 - c1 == 2) {
                current_board[0][7] = EMPTY;
                current_board[0][5] = B_ROOK;
                r2_move_count++;
            } else if (c2 - c1 == -2) {
                current_board[0][0] = EMPTY;
                current_board[0][3] = B_ROOK;
                r1_move_count++;
            }
        }
        if (piece == B_PAWN) {
            if (std::abs(r1 - r2) > 1) {
                en_passant = c1;
                en_passant_move = move_count;
            }
            if (std::abs(c1 - c2) == 1 && current_board[r2][c2] == EMPTY)
                current_board[r2 - 1][c2] = EMPTY;
            if (r2 == 7) {
                bool is_valid_promo = false;
                switch (promoted_piece) {
                case B_ROOK:
                case B_BISHOP:
                case B_KNIGHT:
                case B_QUEEN:
                    is_valid_promo = true;
                    break;
                default:
                    break;
                }
                if (is_valid_promo) {
                    current_board[r2][c2] = promoted_piece;
                    promoted = true;
                }
            }
        }
        if (!promoted)
            current_board[r2][c2] = piece;
        player = 0;
        move_count++;
    }

    halfmove_clock = (moved_pawn || is_capture) ? 0 : static_cast<int16_t>(halfmove_clock + 1);

    uint64_t new_hash = compute_position_hash();
    position_history.push_back(new_hash);
    // Recomputed once per move (not once per is_terminal()/is_threefold_repetition()
    // call - see the comment on repetition_count in chess.hpp). O(history size),
    // bounded by however many plies this game has run so far.
    int8_t count = 0;
    for (uint64_t h : position_history)
        if (h == new_hash)
            ++count;
    repetition_count = count;
}

bool Chess::can_castle(int side) const {
    if (player == 0 && K_move_count == 0) {
        if (side == 0 && R1_move_count == 0 && current_board[7][1] == EMPTY &&
            current_board[7][2] == EMPTY && current_board[7][3] == EMPTY) {
            return true;
        }
        if (side == 1 && R2_move_count == 0 && current_board[7][5] == EMPTY &&
            current_board[7][6] == EMPTY) {
            return true;
        }
    }
    if (player == 1 && k_move_count == 0) {
        if (side == 0 && r1_move_count == 0 && current_board[0][1] == EMPTY &&
            current_board[0][2] == EMPTY && current_board[0][3] == EMPTY) {
            return true;
        }
        if (side == 1 && r2_move_count == 0 && current_board[0][5] == EMPTY &&
            current_board[0][6] == EMPTY) {
            return true;
        }
    }
    return false;
}

bool Chess::check_status() const {
    return bitboard::is_attacked(player, current_board);
}

ActionList<> Chess::actions(bool stop_early) const {
    ActionList<> actss;
    auto b = *this;
    PosList moves;

    if (b.player == 0) {
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                auto p = b.current_board[i][j];
                if (is_white(p)) {
                    moves.clear();
                    switch (p) {
                    case W_ROOK:
                        b.move_rules_R(i, j, moves);
                        break;
                    case W_KNIGHT:
                        b.move_rules_N(i, j, moves);
                        break;
                    case W_BISHOP:
                        b.move_rules_B(i, j, moves);
                        break;
                    case W_QUEEN:
                        b.move_rules_Q(i, j, moves);
                        break;
                    case W_KING:
                        b.move_rules_K(moves);
                        break;
                    case W_PAWN:
                        b.move_rules_P(i, j, moves);
                        break;
                    default:
                        std::abort();
                    }

                    for (int mi = 0; mi < moves.size(); ++mi) {
                        auto f = moves.list[mi];
                        auto r1 = static_cast<int8_t>(i);
                        auto c1 = static_cast<int8_t>(j);
                        if (p == W_KING) {
                            r1 = b.K_move_count > 0 ? i : 7;
                            c1 = b.K_move_count > 0 ? j : 4;
                            for (int rr = 0; rr < 8; ++rr)
                                for (int cc = 0; cc < 8; ++cc)
                                    if (b.current_board[rr][cc] == W_KING) {
                                        r1 = rr;
                                        c1 = cc;
                                    }
                        }
                        auto copy = b.current_board;
                        copy[f.first][f.second] = p;
                        copy[r1][c1] = 0;

                        // Handle Castling
                        if (p == W_KING && std::abs(f.second - c1) == 2) {
                            if (f.second == 6) { // Kingside
                                copy[7][5] = copy[7][7];
                                copy[7][7] = 0;
                            } else if (f.second == 2) { // Queenside
                                copy[7][3] = copy[7][0];
                                copy[7][0] = 0;
                            }
                        }
                        // Handle En Passant
                        if (p == W_PAWN && c1 != f.second &&
                            b.current_board[f.first][f.second] == 0) {
                            copy[r1][f.second] = 0;
                        }

                        if (!bitboard::is_attacked(b.player, copy)) {
                            if (p == W_PAWN && f.first == 0) {
                                actss.emplace_back(r1, c1, f.first, f.second, 1);
                                actss.emplace_back(r1, c1, f.first, f.second, 2);
                                actss.emplace_back(r1, c1, f.first, f.second, 3);
                                actss.emplace_back(r1, c1, f.first, f.second, 4);
                            } else {
                                actss.emplace_back(r1, c1, f.first, f.second, 0);
                            }
                            if (stop_early && !actss.empty())
                                return actss;
                        }
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                auto p = b.current_board[i][j];
                if (is_black(p)) {
                    moves.clear();
                    switch (p) {
                    case B_ROOK:
                        b.move_rules_r(i, j, moves);
                        break;
                    case B_KNIGHT:
                        b.move_rules_n(i, j, moves);
                        break;
                    case B_BISHOP:
                        b.move_rules_b(i, j, moves);
                        break;
                    case B_QUEEN:
                        b.move_rules_q(i, j, moves);
                        break;
                    case B_KING:
                        b.move_rules_k(moves);
                        break;
                    case B_PAWN:
                        b.move_rules_p(i, j, moves);
                        break;
                    default:
                        std::abort();
                    }

                    for (int mi = 0; mi < moves.size(); ++mi) {
                        auto f = moves.list[mi];
                        auto r1 = static_cast<int8_t>(i);
                        auto c1 = static_cast<int8_t>(j);
                        if (p == B_KING) {
                            r1 = b.k_move_count > 0 ? i : 0;
                            c1 = b.k_move_count > 0 ? j : 4;
                            for (int rr = 0; rr < 8; ++rr)
                                for (int cc = 0; cc < 8; ++cc)
                                    if (b.current_board[rr][cc] == B_KING) {
                                        r1 = rr;
                                        c1 = cc;
                                    }
                        }
                        auto copy = b.current_board;
                        copy[f.first][f.second] = p;
                        copy[r1][c1] = 0;

                        // Handle Castling
                        if (p == B_KING && std::abs(f.second - c1) == 2) {
                            if (f.second == 6) { // Kingside
                                copy[0][5] = copy[0][7];
                                copy[0][7] = 0;
                            } else if (f.second == 2) { // Queenside
                                copy[0][3] = copy[0][0];
                                copy[0][0] = 0;
                            }
                        }
                        // Handle En Passant
                        if (p == B_PAWN && c1 != f.second &&
                            b.current_board[f.first][f.second] == 0) {
                            copy[r1][f.second] = 0;
                        }

                        if (!bitboard::is_attacked(b.player, copy)) {
                            if (p == B_PAWN && f.first == 7) {
                                actss.emplace_back(r1, c1, f.first, f.second, 1);
                                actss.emplace_back(r1, c1, f.first, f.second, 2);
                                actss.emplace_back(r1, c1, f.first, f.second, 3);
                                actss.emplace_back(r1, c1, f.first, f.second, 4);
                            } else {
                                actss.emplace_back(r1, c1, f.first, f.second, 0);
                            }
                            if (stop_early && !actss.empty())
                                return actss;
                        }
                    }
                }
            }
        }
    }

    return actss;
}
