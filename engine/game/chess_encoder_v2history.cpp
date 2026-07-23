#include "chess_encoder_v2history.hpp"

#include "chess.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
constexpr int kHistoryPlanes = 14;
constexpr int kAuxiliaryPlanes = 7;

// Encodes one frame's 14 piece/repetition planes into out[0..14*64), using
// `perspective` (always the CURRENT side to move, per the reference: every
// historical frame is colored/oriented from the mover's CURRENT perspective,
// not that frame's own side to move) for both the own/opponent split and the
// row orientation. Mirrors ChessEncoderV1's row-flip convention (see the
// class comment on why that, not the reference's opposite flip, was kept).
void encode_frame(const Chess::board_t &board, int8_t perspective, int8_t repetitions_before,
                  float *out) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int r = (perspective == 0) ? i : (7 - i);
            auto p = board[r][j];
            if (p != EMPTY) {
                bool is_own_piece = (perspective == 0 && p > 0) || (perspective == 1 && p < 0);
                int plane = (is_own_piece ? 0 : 6) + std::abs(p) - 1;
                out[plane * 64 + i * 8 + j] = 1.0f;
            }
        }
    }
    if (repetitions_before >= 1) {
        std::fill(out + 12 * 64, out + 13 * 64, 1.0f);
    }
    if (repetitions_before >= 2) {
        std::fill(out + 13 * 64, out + 14 * 64, 1.0f);
    }
}
} // namespace

ChessEncoderV2History::ChessEncoderV2History(int history) : history_(history) {
    if (history != 1 && history != 4 && history != 8) {
        throw std::invalid_argument("ChessEncoderV2History: history must be 1, 4, or 8, got " +
                                    std::to_string(history));
    }
}

void ChessEncoderV2History::write_canonical_state(const GameState &state, float *out_buffer) const {
    const Chess &game = static_cast<const Chess &>(state);
    int total_planes = kHistoryPlanes * history_ + kAuxiliaryPlanes;
    std::fill(out_buffer, out_buffer + total_planes * 64, 0.0f);

    int8_t perspective = game.player;

    // Frame 0 is always the current position (mirrors the reference's
    // `positions()` iterator, which yields `current` before `previous[..]`).
    int8_t current_reps_before =
        static_cast<int8_t>(std::min(2, std::max(0, game.repetition_count - 1)));
    encode_frame(game.current_board, perspective, current_reps_before, out_buffer);

    // Frames 1..history-1 come from the rolling history window; frames past
    // what Chess has actually recorded (history_count) stay all-zero, matching
    // the reference's Option<ChessPosition>::None for "before the game started".
    for (int frame = 1; frame < history_; ++frame) {
        int hist_idx = frame - 1;
        if (hist_idx >= game.history_count)
            break;
        encode_frame(game.history_boards[hist_idx], perspective,
                    game.history_repetitions_before[hist_idx], out_buffer + frame * 14 * 64);
    }

    // 7 auxiliary planes for the CURRENT position, appended after all history
    // frames. Castling-rights derivation matches ChessEncoderV1's exactly.
    int base = history_ * kHistoryPlanes * 64;
    bool own_k_castle = (game.player == 0)
                            ? (game.k_move_count == 0 && game.r2_move_count == 0)
                            : (game.K_move_count == 0 && game.R2_move_count == 0);
    bool own_q_castle = (game.player == 0)
                            ? (game.k_move_count == 0 && game.r1_move_count == 0)
                            : (game.K_move_count == 0 && game.R1_move_count == 0);
    bool opp_k_castle = (game.player == 0)
                            ? (game.K_move_count == 0 && game.R2_move_count == 0)
                            : (game.k_move_count == 0 && game.r2_move_count == 0);
    bool opp_q_castle = (game.player == 0)
                            ? (game.K_move_count == 0 && game.R1_move_count == 0)
                            : (game.k_move_count == 0 && game.r1_move_count == 0);

    std::fill(out_buffer + base, out_buffer + base + 64, (game.player == 0) ? 1.0f : 0.0f);
    std::fill(out_buffer + base + 64, out_buffer + base + 128, own_k_castle ? 1.0f : 0.0f);
    std::fill(out_buffer + base + 128, out_buffer + base + 192, own_q_castle ? 1.0f : 0.0f);
    std::fill(out_buffer + base + 192, out_buffer + base + 256, opp_k_castle ? 1.0f : 0.0f);
    std::fill(out_buffer + base + 256, out_buffer + base + 320, opp_q_castle ? 1.0f : 0.0f);
    float halfmove_frac = std::min(1.0f, static_cast<float>(game.halfmove_clock) / 100.0f);
    std::fill(out_buffer + base + 320, out_buffer + base + 384, halfmove_frac);
    // move_count is this engine's ply counter (0 at game start, incremented once
    // per move_piece() call), matching the reference's `ply` exactly.
    float fullmove_frac =
        std::min(1.0f, static_cast<float>(game.move_count / 2 + 1) / 200.0f);
    std::fill(out_buffer + base + 384, out_buffer + base + 448, fullmove_frac);
}

std::vector<int64_t> ChessEncoderV2History::state_shape() const {
    return {kHistoryPlanes * history_ + kAuxiliaryPlanes, 8, 8};
}
