#ifndef STATE_ENCODER_HPP
#define STATE_ENCODER_HPP

#include "game.hpp"
#include <cstdint>
#include <vector>

// Turns a game position into the neural-network input tensor. Deliberately
// decoupled from the Game/GameState classes (which own game *rules*, not the
// network's input *representation*): a single game can then support multiple
// encodings, selected at runtime, without subclassing the game or bolting
// encoding flags onto it - e.g. different chess encodings for different network
// generations.
//
// The chosen encoder is threaded to BOTH places encoding happens - the
// inference factory (engine/inference) and self-play trajectory recording
// (training/self_play.cpp) - since the network must be trained on exactly what
// it is served at inference.
//
// The GameState& passed to write_canonical_state is always the concrete Game
// (Game::get_canonical_state() returns shared_from_this(); see game.hpp), so a
// concrete encoder static_casts it to the game type it knows how to read.
struct StateEncoder {
    virtual ~StateEncoder() = default;

    // Writes the encoding of `state` into `out` (row-major planes). `out` must
    // have room for the product of state_shape().
    virtual void write_canonical_state(const GameState &state, float *out) const = 0;

    // The shape {channels, height, width} this encoder produces - the network's
    // input dimensions. Replaces the old GameState::get_state_shape().
    virtual std::vector<int64_t> state_shape() const = 0;
};

#endif // STATE_ENCODER_HPP
