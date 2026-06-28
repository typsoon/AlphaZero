#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace bitboard {

extern bool initialized;
void init();

// Returns true if the king of player `player` (0=White, 1=Black) is attacked by the opponent.
// `board` is the 8x8 array: 1..6 for White, -1..-6 for Black.
bool is_attacked(int player, const std::array<std::array<int8_t, 8>, 8> &board);

struct Move {
    int r1, c1, r2, c2, promo;
};

// Generates all legal moves for `player` (0=White, 1=Black).
std::vector<Move> generate_legal_moves(int player,
                                       const std::array<std::array<int8_t, 8>, 8> &board,
                                       int en_passant_c, bool w_kingside, bool w_queenside,
                                       bool b_kingside, bool b_queenside);

} // namespace bitboard
