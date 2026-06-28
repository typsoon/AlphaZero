#include "bitboard.hpp"
#include <iostream>

namespace bitboard {

bool initialized = false;

uint64_t knight_attacks[64];
uint64_t king_attacks[64];
uint64_t pawn_attacks[2][64]; // [0=White, 1=Black][square]

void init() {
    if (initialized)
        return;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            int sq = r * 8 + c;
            uint64_t n_mask = 0;
            int dr[] = {-2, -2, -1, -1, 1, 1, 2, 2};
            int dc[] = {-1, 1, -2, 2, -2, 2, -1, 1};
            for (int i = 0; i < 8; ++i) {
                int nr = r + dr[i];
                int nc = c + dc[i];
                if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
                    n_mask |= (1ULL << (nr * 8 + nc));
                }
            }
            knight_attacks[sq] = n_mask;

            uint64_t k_mask = 0;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    if (i == 0 && j == 0)
                        continue;
                    int nr = r + i, nc = c + j;
                    if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
                        k_mask |= (1ULL << (nr * 8 + nc));
                    }
                }
            }
            king_attacks[sq] = k_mask;

            // White pawn attacks (move up, so row - 1)
            uint64_t wp_mask = 0;
            if (r - 1 >= 0) {
                if (c - 1 >= 0)
                    wp_mask |= (1ULL << ((r - 1) * 8 + (c - 1)));
                if (c + 1 < 8)
                    wp_mask |= (1ULL << ((r - 1) * 8 + (c + 1)));
            }
            pawn_attacks[0][sq] = wp_mask;

            // Black pawn attacks (move down, so row + 1)
            uint64_t bp_mask = 0;
            if (r + 1 < 8) {
                if (c - 1 >= 0)
                    bp_mask |= (1ULL << ((r + 1) * 8 + (c - 1)));
                if (c + 1 < 8)
                    bp_mask |= (1ULL << ((r + 1) * 8 + (c + 1)));
            }
            pawn_attacks[1][sq] = bp_mask;
        }
    }
    initialized = true;
}

static uint64_t get_sliding_attacks(int sq, uint64_t occupied, bool is_rook, bool is_bishop) {
    uint64_t attacks = 0;
    int r = sq / 8;
    int c = sq % 8;

    if (is_rook) {
        // Up (negative rank)
        for (int i = r - 1; i >= 0; --i) {
            uint64_t b = 1ULL << (i * 8 + c);
            attacks |= b;
            if (occupied & b)
                break;
        }
        // Down (positive rank)
        for (int i = r + 1; i < 8; ++i) {
            uint64_t b = 1ULL << (i * 8 + c);
            attacks |= b;
            if (occupied & b)
                break;
        }
        // Left (negative col)
        for (int i = c - 1; i >= 0; --i) {
            uint64_t b = 1ULL << (r * 8 + i);
            attacks |= b;
            if (occupied & b)
                break;
        }
        // Right (positive col)
        for (int i = c + 1; i < 8; ++i) {
            uint64_t b = 1ULL << (r * 8 + i);
            attacks |= b;
            if (occupied & b)
                break;
        }
    }
    if (is_bishop) {
        // Up-Left
        for (int i = r - 1, j = c - 1; i >= 0 && j >= 0; --i, --j) {
            uint64_t b = 1ULL << (i * 8 + j);
            attacks |= b;
            if (occupied & b)
                break;
        }
        // Up-Right
        for (int i = r - 1, j = c + 1; i >= 0 && j < 8; --i, ++j) {
            uint64_t b = 1ULL << (i * 8 + j);
            attacks |= b;
            if (occupied & b)
                break;
        }
        // Down-Left
        for (int i = r + 1, j = c - 1; i < 8 && j >= 0; ++i, --j) {
            uint64_t b = 1ULL << (i * 8 + j);
            attacks |= b;
            if (occupied & b)
                break;
        }
        // Down-Right
        for (int i = r + 1, j = c + 1; i < 8 && j < 8; ++i, ++j) {
            uint64_t b = 1ULL << (i * 8 + j);
            attacks |= b;
            if (occupied & b)
                break;
        }
    }
    return attacks;
}

bool is_attacked(int player, const std::array<std::array<int8_t, 8>, 8> &board) {
    if (!initialized)
        init();

    uint64_t my_king = 0;
    uint64_t opp_pawns = 0;
    uint64_t opp_knights = 0;
    uint64_t opp_bishops = 0;
    uint64_t opp_rooks = 0;
    uint64_t opp_queens = 0;
    uint64_t opp_kings = 0;
    uint64_t occupied = 0;

    int8_t k_val = (player == 0) ? 6 : -6;
    int8_t opp_p = (player == 0) ? -1 : 1;
    int8_t opp_n = (player == 0) ? -2 : 2;
    int8_t opp_b = (player == 0) ? -3 : 3;
    int8_t opp_r = (player == 0) ? -4 : 4;
    int8_t opp_q = (player == 0) ? -5 : 5;
    int8_t opp_k = (player == 0) ? -6 : 6;

    int k_sq = -1;

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            int8_t p = board[r][c];
            if (p == 0)
                continue;
            uint64_t sq_mask = 1ULL << (r * 8 + c);
            occupied |= sq_mask;

            if (p == k_val) {
                my_king |= sq_mask;
                k_sq = r * 8 + c;
            } else if (p == opp_p) {
                opp_pawns |= sq_mask;
            } else if (p == opp_n) {
                opp_knights |= sq_mask;
            } else if (p == opp_b) {
                opp_bishops |= sq_mask;
            } else if (p == opp_r) {
                opp_rooks |= sq_mask;
            } else if (p == opp_q) {
                opp_queens |= sq_mask;
            } else if (p == opp_k) {
                opp_kings |= sq_mask;
            }
        }
    }

    if (k_sq == -1)
        return false; // Should not happen in valid chess state

    // 1. Check pawns (our pawn attacks from king position would hit their pawns)
    if (pawn_attacks[player][k_sq] & opp_pawns)
        return true;

    // 2. Check knights
    if (knight_attacks[k_sq] & opp_knights)
        return true;

    // 3. Check kings
    if (king_attacks[k_sq] & opp_kings)
        return true;

    // 4. Check sliders
    uint64_t bq = opp_bishops | opp_queens;
    uint64_t rq = opp_rooks | opp_queens;

    if (bq || rq) {
        uint64_t bishop_attacks = get_sliding_attacks(k_sq, occupied, false, true);
        if (bishop_attacks & bq)
            return true;

        uint64_t rook_attacks = get_sliding_attacks(k_sq, occupied, true, false);
        if (rook_attacks & rq)
            return true;
    }

    return false;
}

} // namespace bitboard
