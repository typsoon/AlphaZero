#include "connect4_encoder.hpp"

#include "connect4.hpp"

// Verbatim move of the old Connect4::write_canonical_state; member accesses now
// go through `game`, reached via friendship (declared in connect4.hpp).
void Connect4Encoder::write_canonical_state(const GameState &state, float *out_buffer) const {
    const Connect4 &game = static_cast<const Connect4 &>(state);
    int idx = 0;
    for (int row = 0; row < Connect4::ROWS; row++) {
        for (int col = 0; col < Connect4::COLS; col++) {
            out_buffer[idx++] = static_cast<float>(game.board[row][col] * game.currentPlayer);
        }
    }
}

std::vector<int64_t> Connect4Encoder::state_shape() const {
    return {std::get<0>(Connect4::state_dim), std::get<1>(Connect4::state_dim),
            std::get<2>(Connect4::state_dim)};
}
