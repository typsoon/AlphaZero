// Original source: https://github.com/geochri/AlphaZero_Chess/blob/master/src/chess_board.py
#include "chess.hpp"
#include "bitboard.hpp"
#include <cmath>
#include <iostream>

Chess::Chess() {
    Chess::reset();
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
}

bool Chess::is_terminal() const {
    return actions(true).empty();
}

int Chess::get_current_player() const {
    return player;
}

float Chess::reward() const {
    if (!is_terminal())
        return 0.0f;
    if (check_status())
        return -1.0f;
    return 0.0f;
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

void Chess::write_canonical_state(float *out_buffer) const {
    std::fill(out_buffer, out_buffer + 19 * 64, 0.0f); // NOLINT

    bool p1_k_castle = (player == 0) ? (k_move_count == 0 && r2_move_count == 0)
                                     : (K_move_count == 0 && R2_move_count == 0);
    bool p1_q_castle = (player == 0) ? (k_move_count == 0 && r1_move_count == 0)
                                     : (K_move_count == 0 && R1_move_count == 0);
    bool p2_k_castle = (player == 0) ? (K_move_count == 0 && R2_move_count == 0)
                                     : (k_move_count == 0 && r2_move_count == 0);
    bool p2_q_castle = (player == 0) ? (K_move_count == 0 && R1_move_count == 0)
                                     : (k_move_count == 0 && r1_move_count == 0);

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int r = (player == 0) ? i : (7 - i);
            auto p = current_board[r][j];

            if (p != EMPTY) {
                bool is_p1_piece = (player == 0 && p > 0) || (player == 1 && p < 0);
                int plane = (is_p1_piece ? 0 : 6) + std::abs(p) - 1;
                out_buffer[plane * 64 + i * 8 + j] = 1.0f;
            }

            out_buffer[12 * 64 + i * 8 + j] = (player == 0) ? 1.0f : 0.0f;
            out_buffer[13 * 64 + i * 8 + j] = static_cast<float>(move_count);
            out_buffer[14 * 64 + i * 8 + j] = p1_k_castle ? 1.0f : 0.0f;
            out_buffer[15 * 64 + i * 8 + j] = p1_q_castle ? 1.0f : 0.0f;
            out_buffer[16 * 64 + i * 8 + j] = p2_k_castle ? 1.0f : 0.0f;
            out_buffer[17 * 64 + i * 8 + j] = p2_q_castle ? 1.0f : 0.0f;
            out_buffer[18 * 64 + i * 8 + j] = (en_passant != -1 && en_passant == j) ? 1.0f : 0.0f;
        }
    }
}

std::vector<int64_t> Chess::get_state_shape() const {
    return {19, 8, 8};
}

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
    if (player == 0) {
        auto promoted = false;
        auto piece = current_board[r1][c1];
        current_board[r1][c1] = EMPTY;
        if (piece == W_ROOK && r1 == 7 && c1 == 0)
            R1_move_count++;
        if (piece == W_ROOK && r1 == 7 && c1 == 7)
            R2_move_count++;
        if (piece == W_KING)
            K_move_count++;
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
        if (piece == B_KING)
            k_move_count++;
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
                        auto copy = current_board;
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
                            current_board[f.first][f.second] == 0) {
                            copy[r1][f.second] = 0;
                        }

                        if (!bitboard::is_attacked(player, copy)) {
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
                        auto copy = current_board;
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
                            current_board[f.first][f.second] == 0) {
                            copy[r1][f.second] = 0;
                        }

                        if (!bitboard::is_attacked(player, copy)) {
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
