#include "bitboard.hpp"
#include <cstring>

namespace bitboard {

bool is_attacked(int player, const std::array<std::array<int8_t, 8>, 8> &board) {
    int8_t my_king = (player == 0) ? 6 : -6;
    int8_t opp_pawn = (player == 0) ? -1 : 1;
    int8_t opp_knight = (player == 0) ? -2 : 2;
    int8_t opp_bishop = (player == 0) ? -3 : 3;
    int8_t opp_rook = (player == 0) ? -4 : 4;
    int8_t opp_queen = (player == 0) ? -5 : 5;
    int8_t opp_king = (player == 0) ? -6 : 6;

    const int8_t *ptr = board[0].data();
    const auto *k_ptr = static_cast<const int8_t *>(memchr(ptr, my_king, 64));
    if (!k_ptr)
        return false;

    int k_sq = k_ptr - ptr;
    int r = k_sq / 8;
    int c = k_sq % 8;

    // 1. Check pawns
    if (player == 0) {
        // White king, check black pawns (which attack downwards, so we look up for them)
        if (r - 1 >= 0) {
            if (c - 1 >= 0 && board[r - 1][c - 1] == opp_pawn)
                return true;
            if (c + 1 < 8 && board[r - 1][c + 1] == opp_pawn)
                return true;
        }
    } else {
        // Black king, check white pawns (look down)
        if (r + 1 < 8) {
            if (c - 1 >= 0 && board[r + 1][c - 1] == opp_pawn)
                return true;
            if (c + 1 < 8 && board[r + 1][c + 1] == opp_pawn)
                return true;
        }
    }

    // 2. Check knights
    static const int kr[] = {-2, -2, -1, -1, 1, 1, 2, 2};
    static const int kc[] = {-1, 1, -2, 2, -2, 2, -1, 1};
    for (int i = 0; i < 8; ++i) {
        int nr = r + kr[i];
        int nc = c + kc[i];
        if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
            if (board[nr][nc] == opp_knight)
                return true;
        }
    }

    // 3. Check kings
    static const int k_dr[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static const int k_dc[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    for (int i = 0; i < 8; ++i) {
        int nr = r + k_dr[i];
        int nc = c + k_dc[i];
        if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
            if (board[nr][nc] == opp_king)
                return true;
        }
    }

    // 4. Check straight sliders (Rook/Queen)
    // Up
    for (int i = r - 1; i >= 0; --i) {
        int8_t p = board[i][c];
        if (p != 0) {
            if (p == opp_rook || p == opp_queen)
                return true;
            break;
        }
    }
    // Down
    for (int i = r + 1; i < 8; ++i) {
        int8_t p = board[i][c];
        if (p != 0) {
            if (p == opp_rook || p == opp_queen)
                return true;
            break;
        }
    }
    // Left
    for (int i = c - 1; i >= 0; --i) {
        int8_t p = board[r][i];
        if (p != 0) {
            if (p == opp_rook || p == opp_queen)
                return true;
            break;
        }
    }
    // Right
    for (int i = c + 1; i < 8; ++i) {
        int8_t p = board[r][i];
        if (p != 0) {
            if (p == opp_rook || p == opp_queen)
                return true;
            break;
        }
    }

    // 5. Check diagonal sliders (Bishop/Queen)
    // Up-Left
    for (int i = r - 1, j = c - 1; i >= 0 && j >= 0; --i, --j) {
        int8_t p = board[i][j];
        if (p != 0) {
            if (p == opp_bishop || p == opp_queen)
                return true;
            break;
        }
    }
    // Up-Right
    for (int i = r - 1, j = c + 1; i >= 0 && j < 8; --i, ++j) {
        int8_t p = board[i][j];
        if (p != 0) {
            if (p == opp_bishop || p == opp_queen)
                return true;
            break;
        }
    }
    // Down-Left
    for (int i = r + 1, j = c - 1; i < 8 && j >= 0; ++i, --j) {
        int8_t p = board[i][j];
        if (p != 0) {
            if (p == opp_bishop || p == opp_queen)
                return true;
            break;
        }
    }
    // Down-Right
    for (int i = r + 1, j = c + 1; i < 8 && j < 8; ++i, ++j) {
        int8_t p = board[i][j];
        if (p != 0) {
            if (p == opp_bishop || p == opp_queen)
                return true;
            break;
        }
    }

    return false;
}

} // namespace bitboard
