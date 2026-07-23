#ifndef CHESS_ENCODER_HPP
#define CHESS_ENCODER_HPP

#include "state_encoder.hpp"

// The original 19-plane chess encoding: 12 piece planes (6 own + 6 opponent,
// board row-flipped so the side to move always advances toward row 0), a
// side-to-move plane, a raw move-count plane, 4 castling-rights planes, and an
// en-passant plane (gated on freshness - only the immediately-preceding double
// push counts, which the inference transposition cache relies on; see
// inference_cache.hpp).
//
// This is byte-for-byte identical to the historical Chess::write_canonical_state
// (enforced by the golden test in engine/tests/test_state_encoder.cpp): every
// existing chess checkpoint was trained on exactly this layout, so it must not
// drift. A different encoding (e.g. history-stacked planes for a newer network
// generation) gets its own StateEncoder subclass, with no change to Chess.
class ChessEncoderV1 : public StateEncoder {
  public:
    // `state` must be a Chess (guaranteed: get_canonical_state() returns the
    // Game itself). Reads Chess's position via friendship.
    void write_canonical_state(const GameState &state, float *out) const override;
    std::vector<int64_t> state_shape() const override;
};

#endif // CHESS_ENCODER_HPP
