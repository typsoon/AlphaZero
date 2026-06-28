// Original source: https://github.com/geochri/AlphaZero_Chess/blob/master/src/chess_board.py
#include "chess.hpp"
#include "bitboard.hpp"
#include <cmath>
#include <iostream>
#include <set>

Chess::Chess() {
    reset();
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
bool Chess::is_empty(int8_t p) {
    return p == 0;
}

int Chess::encode_action(const ChessAction<> &a) const {
    int from = (a.r1 * 8) + a.c1;
    int to = (a.r2 * 8) + a.c2;
    return (from * 64 + to) * 5 + a.promotion;
}

ChessAction<> Chess::decode_action(int a) const {
    ChessAction<> act;
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
    std::fill(out_buffer, out_buffer + 19 * 64, 0.0f);

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

std::pair<std::vector<Chess::pos_t>, std::vector<Chess::pos_t>>
Chess::move_rules_P(int8_t i, int8_t j) const {
    std::vector<pos_t> next_positions;
    std::vector<pos_t> threats;
    const auto &board_state = current_board;
    if (i - 1 >= 0 && i - 1 <= 7 && j + 1 >= 0 && j + 1 <= 7)
        threats.emplace_back(i - 1, j + 1);
    if (i - 1 >= 0 && i - 1 <= 7 && j - 1 >= 0 && j - 1 <= 7)
        threats.emplace_back(i - 1, j - 1);
    if (i == 6) {
        if (board_state[i - 1][j] == EMPTY) {
            next_positions.emplace_back(i - 1, j);
            if (board_state[i - 2][j] == EMPTY)
                next_positions.emplace_back(i - 2, j);
        }
    } else if (i == 3 && en_passant != -1) {
        if (j - 1 == en_passant && std::abs(en_passant_move - move_count) == 1)
            next_positions.emplace_back(i - 1, j - 1);
        else if (j + 1 == en_passant && std::abs(en_passant_move - move_count) == 1)
            next_positions.emplace_back(i - 1, j + 1);
    }
    if ((i == 1 || i == 2 || i == 3 || i == 4 || i == 5) && board_state[i - 1][j] == EMPTY)
        next_positions.emplace_back(i - 1, j);
    if (j == 0 && is_black(board_state[i - 1][j + 1]))
        next_positions.emplace_back(i - 1, j + 1);
    else if (j == 7 && is_black(board_state[i - 1][j - 1]))
        next_positions.emplace_back(i - 1, j - 1);
    else if (j >= 1 && j <= 6) {
        if (is_black(board_state[i - 1][j + 1]))
            next_positions.emplace_back(i - 1, j + 1);
        if (is_black(board_state[i - 1][j - 1]))
            next_positions.emplace_back(i - 1, j - 1);
    }
    return {next_positions, threats};
}

std::pair<std::vector<Chess::pos_t>, std::vector<Chess::pos_t>>
Chess::move_rules_p(int8_t i, int8_t j) const {
    std::vector<pos_t> next_positions;
    std::vector<pos_t> threats;
    const auto &board_state = current_board;
    if (i + 1 >= 0 && i + 1 <= 7 && j + 1 >= 0 && j + 1 <= 7)
        threats.emplace_back(i + 1, j + 1);
    if (i + 1 >= 0 && i + 1 <= 7 && j - 1 >= 0 && j - 1 <= 7)
        threats.emplace_back(i + 1, j - 1);
    if (i == 1) {
        if (board_state[i + 1][j] == EMPTY) {
            next_positions.emplace_back(i + 1, j);
            if (board_state[i + 2][j] == EMPTY)
                next_positions.emplace_back(i + 2, j);
        }
    } else if (i == 4 && en_passant != -1) {
        if (j - 1 == en_passant && std::abs(en_passant_move - move_count) == 1)
            next_positions.emplace_back(i + 1, j - 1);
        else if (j + 1 == en_passant && std::abs(en_passant_move - move_count) == 1)
            next_positions.emplace_back(i + 1, j + 1);
    }
    if ((i == 2 || i == 3 || i == 4 || i == 5 || i == 6) && board_state[i + 1][j] == EMPTY)
        next_positions.emplace_back(i + 1, j);
    if (j == 0 && is_white(board_state[i + 1][j + 1]))
        next_positions.emplace_back(i + 1, j + 1);
    else if (j == 7 && is_white(board_state[i + 1][j - 1]))
        next_positions.emplace_back(i + 1, j - 1);
    else if (j >= 1 && j <= 6) {
        if (is_white(board_state[i + 1][j + 1]))
            next_positions.emplace_back(i + 1, j + 1);
        if (is_white(board_state[i + 1][j - 1]))
            next_positions.emplace_back(i + 1, j - 1);
    }
    return {next_positions, threats};
}

std::vector<Chess::pos_t> Chess::move_rules_r(int8_t i, int8_t j) const {
    std::vector<pos_t> next_positions;
    const auto &board_state = current_board;
    int a = i;
    while (a != 0) {
        if (board_state[a - 1][j] != EMPTY) {
            if (is_white(board_state[a - 1][j]))
                next_positions.emplace_back(a - 1, j);
            break;
        }
        next_positions.emplace_back(a - 1, j);
        a -= 1;
    }
    a = i;
    while (a != 7) {
        if (board_state[a + 1][j] != EMPTY) {
            if (is_white(board_state[a + 1][j]))
                next_positions.emplace_back(a + 1, j);
            break;
        }
        next_positions.emplace_back(a + 1, j);
        a += 1;
    }
    a = j;
    while (a != 7) {
        if (board_state[i][a + 1] != EMPTY) {
            if (is_white(board_state[i][a + 1]))
                next_positions.emplace_back(i, a + 1);
            break;
        }
        next_positions.emplace_back(i, a + 1);
        a += 1;
    }
    a = j;
    while (a != 0) {
        if (board_state[i][a - 1] != EMPTY) {
            if (is_white(board_state[i][a - 1]))
                next_positions.emplace_back(i, a - 1);
            break;
        }
        next_positions.emplace_back(i, a - 1);
        a -= 1;
    }
    return next_positions;
}

std::vector<Chess::pos_t> Chess::move_rules_R(int8_t i, int8_t j) const {
    std::vector<pos_t> next_positions;
    const auto &board_state = current_board;
    int a = i;
    while (a != 0) {
        if (board_state[a - 1][j] != EMPTY) {
            if (is_black(board_state[a - 1][j]))
                next_positions.emplace_back(a - 1, j);
            break;
        }
        next_positions.emplace_back(a - 1, j);
        a -= 1;
    }
    a = i;
    while (a != 7) {
        if (board_state[a + 1][j] != EMPTY) {
            if (is_black(board_state[a + 1][j]))
                next_positions.emplace_back(a + 1, j);
            break;
        }
        next_positions.emplace_back(a + 1, j);
        a += 1;
    }
    a = j;
    while (a != 7) {
        if (board_state[i][a + 1] != EMPTY) {
            if (is_black(board_state[i][a + 1]))
                next_positions.emplace_back(i, a + 1);
            break;
        }
        next_positions.emplace_back(i, a + 1);
        a += 1;
    }
    a = j;
    while (a != 0) {
        if (board_state[i][a - 1] != EMPTY) {
            if (is_black(board_state[i][a - 1]))
                next_positions.emplace_back(i, a - 1);
            break;
        }
        next_positions.emplace_back(i, a - 1);
        a -= 1;
    }
    return next_positions;
}

std::vector<Chess::pos_t> Chess::move_rules_b(int8_t i, int8_t j) const {
    std::vector<pos_t> next_positions;
    const auto &board_state = current_board;
    int a = i, b = j;
    while (a != 0 && b != 0) {
        if (board_state[a - 1][b - 1] != EMPTY) {
            if (is_white(board_state[a - 1][b - 1]))
                next_positions.emplace_back(a - 1, b - 1);
            break;
        }
        next_positions.emplace_back(a - 1, b - 1);
        a -= 1;
        b -= 1;
    }
    a = i;
    b = j;
    while (a != 7 && b != 7) {
        if (board_state[a + 1][b + 1] != EMPTY) {
            if (is_white(board_state[a + 1][b + 1]))
                next_positions.emplace_back(a + 1, b + 1);
            break;
        }
        next_positions.emplace_back(a + 1, b + 1);
        a += 1;
        b += 1;
    }
    a = i;
    b = j;
    while (a != 0 && b != 7) {
        if (board_state[a - 1][b + 1] != EMPTY) {
            if (is_white(board_state[a - 1][b + 1]))
                next_positions.emplace_back(a - 1, b + 1);
            break;
        }
        next_positions.emplace_back(a - 1, b + 1);
        a -= 1;
        b += 1;
    }
    a = i;
    b = j;
    while (a != 7 && b != 0) {
        if (board_state[a + 1][b - 1] != EMPTY) {
            if (is_white(board_state[a + 1][b - 1]))
                next_positions.emplace_back(a + 1, b - 1);
            break;
        }
        next_positions.emplace_back(a + 1, b - 1);
        a += 1;
        b -= 1;
    }
    return next_positions;
}

std::vector<Chess::pos_t> Chess::move_rules_B(int8_t i, int8_t j) const {
    std::vector<pos_t> next_positions;
    const auto &board_state = current_board;
    int a = i, b = j;
    while (a != 0 && b != 0) {
        if (board_state[a - 1][b - 1] != EMPTY) {
            if (is_black(board_state[a - 1][b - 1]))
                next_positions.emplace_back(a - 1, b - 1);
            break;
        }
        next_positions.emplace_back(a - 1, b - 1);
        a -= 1;
        b -= 1;
    }
    a = i;
    b = j;
    while (a != 7 && b != 7) {
        if (board_state[a + 1][b + 1] != EMPTY) {
            if (is_black(board_state[a + 1][b + 1]))
                next_positions.emplace_back(a + 1, b + 1);
            break;
        }
        next_positions.emplace_back(a + 1, b + 1);
        a += 1;
        b += 1;
    }
    a = i;
    b = j;
    while (a != 0 && b != 7) {
        if (board_state[a - 1][b + 1] != EMPTY) {
            if (is_black(board_state[a - 1][b + 1]))
                next_positions.emplace_back(a - 1, b + 1);
            break;
        }
        next_positions.emplace_back(a - 1, b + 1);
        a -= 1;
        b += 1;
    }
    a = i;
    b = j;
    while (a != 7 && b != 0) {
        if (board_state[a + 1][b - 1] != EMPTY) {
            if (is_black(board_state[a + 1][b - 1]))
                next_positions.emplace_back(a + 1, b - 1);
            break;
        }
        next_positions.emplace_back(a + 1, b - 1);
        a += 1;
        b -= 1;
    }
    return next_positions;
}

std::vector<Chess::pos_t> Chess::move_rules_n(int8_t i, int8_t j) const {
    std::vector<pos_t> next_positions;
    const auto &board_state = current_board;
    std::vector<pos_t> steps = {{i + 2, j - 1}, {i + 2, j + 1}, {i + 1, j - 2}, {i - 1, j - 2},
                                {i - 2, j + 1}, {i - 2, j - 1}, {i - 1, j + 2}, {i + 1, j + 2}};
    for (auto p : steps) {
        int a = p.first, b = p.second;
        if (0 <= a && a <= 7 && 0 <= b && b <= 7) {
            if (is_white(board_state[a][b]) || board_state[a][b] == EMPTY) {
                next_positions.emplace_back(a, b);
            }
        }
    }
    return next_positions;
}

std::vector<Chess::pos_t> Chess::move_rules_N(int8_t i, int8_t j) const {
    std::vector<pos_t> next_positions;
    const auto &board_state = current_board;
    std::vector<pos_t> steps = {{i + 2, j - 1}, {i + 2, j + 1}, {i + 1, j - 2}, {i - 1, j - 2},
                                {i - 2, j + 1}, {i - 2, j - 1}, {i - 1, j + 2}, {i + 1, j + 2}};
    for (auto p : steps) {
        int a = p.first, b = p.second;
        if (0 <= a && a <= 7 && 0 <= b && b <= 7) {
            if (is_black(board_state[a][b]) || board_state[a][b] == EMPTY) {
                next_positions.emplace_back(a, b);
            }
        }
    }
    return next_positions;
}

std::vector<Chess::pos_t> Chess::move_rules_q(int8_t i, int8_t j) const {
    auto moves = move_rules_r(i, j);
    auto diag = move_rules_b(i, j);
    moves.insert(moves.end(), diag.begin(), diag.end());
    return moves;
}

std::vector<Chess::pos_t> Chess::move_rules_Q(int8_t i, int8_t j) const {
    auto moves = move_rules_R(i, j);
    auto diag = move_rules_B(i, j);
    moves.insert(moves.end(), diag.begin(), diag.end());
    return moves;
}

std::vector<Chess::pos_t> Chess::possible_W_moves(bool threats) const {
    std::vector<pos_t> c_list;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            auto p = current_board[i][j];
            std::vector<pos_t> moves;
            switch (p) {
            case W_ROOK:
                moves = move_rules_R(i, j);
                break;
            case W_KNIGHT:
                moves = move_rules_N(i, j);
                break;
            case W_BISHOP:
                moves = move_rules_B(i, j);
                break;
            case W_QUEEN:
                moves = move_rules_Q(i, j);
                break;
            case W_PAWN: {
                auto res = move_rules_P(i, j);
                moves = threats ? res.second : res.first;
                break;
            }
            default:
                break;
            }
            c_list.insert(c_list.end(), moves.begin(), moves.end());
        }
    }
    return c_list;
}

std::vector<Chess::pos_t> Chess::possible_B_moves(bool threats) const {
    std::vector<pos_t> c_list;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            auto p = current_board[i][j];
            std::vector<pos_t> moves;
            switch (p) {
            case B_ROOK:
                moves = move_rules_r(i, j);
                break;
            case B_KNIGHT:
                moves = move_rules_n(i, j);
                break;
            case B_BISHOP:
                moves = move_rules_b(i, j);
                break;
            case B_QUEEN:
                moves = move_rules_q(i, j);
                break;
            case B_PAWN: {
                auto res = move_rules_p(i, j);
                moves = threats ? res.second : res.first;
                break;
            }
            default:
                break;
            }
            c_list.insert(c_list.end(), moves.begin(), moves.end());
        }
    }
    return c_list;
}

std::vector<Chess::pos_t> Chess::move_rules_k() const {
    int i = -1, j = -1;
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
        return {};
    std::vector<pos_t> next_positions;
    auto c_list = possible_W_moves(true);
    std::set<std::pair<int, int>> c_set(c_list.begin(), c_list.end());
    std::vector<pos_t> steps = {{i + 1, j},     {i - 1, j},     {i, j + 1},     {i, j - 1},
                                {i + 1, j + 1}, {i - 1, j - 1}, {i + 1, j - 1}, {i - 1, j + 1}};
    for (auto p : steps) {
        int a = p.first, b = p.second;
        if (0 <= a && a <= 7 && 0 <= b && b <= 7) {
            int sq = current_board[a][b];
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
            }
            if (ok && c_set.find({a, b}) == c_set.end()) {
                next_positions.emplace_back(a, b);
            }
        }
    }
    if (const_cast<Chess *>(this)->castle(0, false) == true && check_status() == false)
        next_positions.emplace_back(0, 2);
    if (const_cast<Chess *>(this)->castle(1, false) == true && check_status() == false)
        next_positions.emplace_back(0, 6);
    return next_positions;
}

std::vector<Chess::pos_t> Chess::move_rules_K() const {
    int i = -1, j = -1;
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
        return {};
    std::vector<pos_t> next_positions;
    auto c_list = possible_B_moves(true);
    std::set<std::pair<int, int>> c_set(c_list.begin(), c_list.end());
    std::vector<pos_t> steps = {{i + 1, j},     {i - 1, j},     {i, j + 1},     {i, j - 1},
                                {i + 1, j + 1}, {i - 1, j - 1}, {i + 1, j - 1}, {i - 1, j + 1}};
    for (auto p : steps) {
        int a = p.first, b = p.second;
        if (0 <= a && a <= 7 && 0 <= b && b <= 7) {
            int sq = current_board[a][b];
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
            }
            if (ok && c_set.find({a, b}) == c_set.end()) {
                next_positions.emplace_back(a, b);
            }
        }
    }
    if (const_cast<Chess *>(this)->castle(0, false) == true && check_status() == false)
        next_positions.emplace_back(7, 2);
    if (const_cast<Chess *>(this)->castle(1, false) == true && check_status() == false)
        next_positions.emplace_back(7, 6);
    return next_positions;
}

void Chess::move_piece(int r1, int c1, int r2, int c2, int promoted_piece) {
    if (player == 0) {
        bool promoted = false;
        int piece = current_board[r1][c1];
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
        int piece = current_board[r1][c1];
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

bool Chess::castle(int side, bool inplace) {
    if (player == 0 && K_move_count == 0) {
        if (side == 0 && R1_move_count == 0 && current_board[7][1] == EMPTY &&
            current_board[7][2] == EMPTY && current_board[7][3] == EMPTY) {
            if (inplace) {
                current_board[7][0] = EMPTY;
                current_board[7][3] = W_ROOK;
                current_board[7][4] = EMPTY;
                current_board[7][2] = W_KING;
                K_move_count++;
                player = 1;
            }
            return true;
        } else if (side == 1 && R2_move_count == 0 && current_board[7][5] == EMPTY &&
                   current_board[7][6] == EMPTY) {
            if (inplace) {
                current_board[7][7] = EMPTY;
                current_board[7][5] = W_ROOK;
                current_board[7][4] = EMPTY;
                current_board[7][6] = W_KING;
                K_move_count++;
                player = 1;
            }
            return true;
        }
    }
    if (player == 1 && k_move_count == 0) {
        if (side == 0 && r1_move_count == 0 && current_board[0][1] == EMPTY &&
            current_board[0][2] == EMPTY && current_board[0][3] == EMPTY) {
            if (inplace) {
                current_board[0][0] = EMPTY;
                current_board[0][3] = B_ROOK;
                current_board[0][4] = EMPTY;
                current_board[0][2] = B_KING;
                k_move_count++;
                player = 0;
            }
            return true;
        } else if (side == 1 && r2_move_count == 0 && current_board[0][5] == EMPTY &&
                   current_board[0][6] == EMPTY) {
            if (inplace) {
                current_board[0][7] = EMPTY;
                current_board[0][5] = B_ROOK;
                current_board[0][4] = EMPTY;
                current_board[0][6] = B_KING;
                k_move_count++;
                player = 0;
            }
            return true;
        }
    }
    return false;
}

bool Chess::check_status() const {
    return bitboard::is_attacked(player, current_board);
}

void Chess::backup() {
    copy_board = current_board;
    move_count_copy = move_count;
    en_passant_copy = en_passant;
    en_passant_move_copy = en_passant_move;
    r1_move_count_copy = r1_move_count;
    r2_move_count_copy = r2_move_count;
    k_move_count_copy = k_move_count;
    R1_move_count_copy = R1_move_count;
    R2_move_count_copy = R2_move_count;
    K_move_count_copy = K_move_count;
}

void Chess::restore() {
    current_board = copy_board;
    move_count = move_count_copy;
    en_passant = en_passant_copy;
    en_passant_move = en_passant_move_copy;
    r1_move_count = r1_move_count_copy;
    r2_move_count = r2_move_count_copy;
    k_move_count = k_move_count_copy;
    R1_move_count = R1_move_count_copy;
    R2_move_count = R2_move_count_copy;
    K_move_count = K_move_count_copy;
}

std::vector<ChessAction<>> Chess::actions(bool stop_early) const {
    std::vector<ChessAction<>> acts;
    auto b = *this;
    if (b.player == 0) {
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                auto p = b.current_board[i][j];
                if (is_white(p)) {
                    std::vector<pos_t> moves;
                    switch (p) {
                    case W_ROOK:
                        moves = b.move_rules_R(i, j);
                        break;
                    case W_KNIGHT:
                        moves = b.move_rules_N(i, j);
                        break;
                    case W_BISHOP:
                        moves = b.move_rules_B(i, j);
                        break;
                    case W_QUEEN:
                        moves = b.move_rules_Q(i, j);
                        break;
                    case W_KING:
                        moves = b.move_rules_K();
                        break;
                    case W_PAWN:
                        moves = b.move_rules_P(i, j).first;
                        break;
                    default:
                        std::abort();
                    }

                    for (auto &f : moves) {
                        auto r1 = static_cast<int8_t>(i);
                        auto c1 = static_cast<int8_t>(j);
                        if (p == W_PAWN && f.first == 0) {
                            acts.emplace_back(r1, c1, f.first, f.second, 1); // Q
                            acts.emplace_back(r1, c1, f.first, f.second, 2); // R
                            acts.emplace_back(r1, c1, f.first, f.second, 3); // N
                            acts.emplace_back(r1, c1, f.first, f.second, 4); // B
                        } else {
                            acts.emplace_back(r1, c1, f.first, f.second, 0);
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
                    std::vector<pos_t> moves;
                    switch (p) {
                    case B_ROOK:
                        moves = b.move_rules_r(i, j);
                        break;
                    case B_KNIGHT:
                        moves = b.move_rules_n(i, j);
                        break;
                    case B_BISHOP:
                        moves = b.move_rules_b(i, j);
                        break;
                    case B_QUEEN:
                        moves = b.move_rules_q(i, j);
                        break;
                    case B_KING:
                        moves = b.move_rules_k();
                        break;
                    case B_PAWN:
                        moves = b.move_rules_p(i, j).first;
                        break;
                    default:
                        std::abort();
                    }

                    for (auto &f : moves) {
                        auto r1 = static_cast<int8_t>(i);
                        auto c1 = static_cast<int8_t>(j);
                        if (p == B_PAWN && f.first == 7) {
                            acts.emplace_back(r1, c1, f.first, f.second, 1); // q
                            acts.emplace_back(r1, c1, f.first, f.second, 2); // r
                            acts.emplace_back(r1, c1, f.first, f.second, 3); // n
                            acts.emplace_back(r1, c1, f.first, f.second, 4); // b
                        } else {
                            acts.emplace_back(r1, c1, f.first, f.second, 0);
                        }
                    }
                }
            }
        }
    }

    std::vector<ChessAction<>> actss;
    for (auto &act : acts) {
        std::array<std::array<int8_t, 8>, 8> copy = current_board;
        int8_t p = copy[act.r1][act.c1];
        copy[act.r2][act.c2] = p;
        copy[act.r1][act.c1] = 0;

        // Handle Castling
        if ((p == W_KING || p == B_KING) && std::abs(act.c2 - act.c1) == 2) {
            if (act.c2 == 6) { // Kingside
                copy[act.r2][5] = copy[act.r2][7];
                copy[act.r2][7] = 0;
            } else if (act.c2 == 2) { // Queenside
                copy[act.r2][3] = copy[act.r2][0];
                copy[act.r2][0] = 0;
            }
        }
        // Handle En Passant
        if ((p == W_PAWN || p == B_PAWN) && act.c1 != act.c2 &&
            current_board[act.r2][act.c2] == 0) {
            copy[act.r1][act.c2] = 0;
        }

        if (!bitboard::is_attacked(player, copy)) {
            actss.push_back(act);
            if (stop_early)
                return actss;
        }
    }
    return actss;
}
