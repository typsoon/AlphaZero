#pragma once
#include <array>
#include <cstdint>

namespace bitboard {

// Returns true if the king of player `player` (0=White, 1=Black) is attacked by the opponent.
// `board` is the 8x8 array: 1..6 for White, -1..-6 for Black.
bool is_attacked(int player, const std::array<std::array<int8_t, 8>, 8> &board);

} // namespace bitboard
