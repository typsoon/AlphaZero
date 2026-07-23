#include "chess_encoder.hpp"

#include "chess.hpp"
#include <algorithm>
#include <cmath>

// NB: this is a verbatim move of the logic that lived in
// Chess::write_canonical_state - kept identical on purpose (see the golden test
// in tests/test_state_encoder.cpp). Only the member accesses are now qualified
// through `game`, reached via friendship (declared in chess.hpp).
void ChessEncoderV1::write_canonical_state(const GameState &state, float *out_buffer) const {
    const Chess &game = static_cast<const Chess &>(state);
    std::fill(out_buffer, out_buffer + 19 * 64, 0.0f); // NOLINT

    bool p1_k_castle = (game.player == 0)
                           ? (game.k_move_count == 0 && game.r2_move_count == 0)
                           : (game.K_move_count == 0 && game.R2_move_count == 0);
    bool p1_q_castle = (game.player == 0)
                           ? (game.k_move_count == 0 && game.r1_move_count == 0)
                           : (game.K_move_count == 0 && game.R1_move_count == 0);
    bool p2_k_castle = (game.player == 0)
                           ? (game.K_move_count == 0 && game.R2_move_count == 0)
                           : (game.k_move_count == 0 && game.r2_move_count == 0);
    bool p2_q_castle = (game.player == 0)
                           ? (game.K_move_count == 0 && game.R1_move_count == 0)
                           : (game.k_move_count == 0 && game.r1_move_count == 0);

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int r = (game.player == 0) ? i : (7 - i);
            auto p = game.current_board[r][j];

            if (p != EMPTY) {
                bool is_p1_piece = (game.player == 0 && p > 0) || (game.player == 1 && p < 0);
                int plane = (is_p1_piece ? 0 : 6) + std::abs(p) - 1;
                out_buffer[plane * 64 + i * 8 + j] = 1.0f;
            }

            out_buffer[12 * 64 + i * 8 + j] = (game.player == 0) ? 1.0f : 0.0f;
            out_buffer[13 * 64 + i * 8 + j] = static_cast<float>(game.move_count);
            out_buffer[14 * 64 + i * 8 + j] = p1_k_castle ? 1.0f : 0.0f;
            out_buffer[15 * 64 + i * 8 + j] = p1_q_castle ? 1.0f : 0.0f;
            out_buffer[16 * 64 + i * 8 + j] = p2_k_castle ? 1.0f : 0.0f;
            out_buffer[17 * 64 + i * 8 + j] = p2_q_castle ? 1.0f : 0.0f;
            // En-passant plane, gated on freshness: only the immediately
            // preceding double push is a live EP target. Without this gate two
            // positions identical except for EP freshness produce the same
            // tensor while having different legal actions, breaking the
            // inference cache's identical-tensor => identical-legal-actions
            // assumption (see inference_cache.hpp).
            out_buffer[18 * 64 + i * 8 + j] =
                (game.en_passant != -1 && game.en_passant == j &&
                 std::abs(game.en_passant_move - game.move_count) == 1)
                    ? 1.0f
                    : 0.0f;
        }
    }
}

std::vector<int64_t> ChessEncoderV1::state_shape() const {
    return {Chess::state_dim[0], Chess::state_dim[1], Chess::state_dim[2]};
}
