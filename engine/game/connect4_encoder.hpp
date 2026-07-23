#ifndef CONNECT4_ENCODER_HPP
#define CONNECT4_ENCODER_HPP

#include "state_encoder.hpp"

// Single-plane Connect4 encoding: each cell times the side to move, so the
// tensor is always from the mover's perspective. Byte-identical to the old
// Connect4::write_canonical_state.
class Connect4Encoder : public StateEncoder {
  public:
    void write_canonical_state(const GameState &state, float *out) const override;
    std::vector<int64_t> state_shape() const override;
};

#endif // CONNECT4_ENCODER_HPP
