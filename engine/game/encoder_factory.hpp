#ifndef ENCODER_FACTORY_HPP
#define ENCODER_FACTORY_HPP

#include "state_encoder.hpp"
#include <memory>

// Picks the default StateEncoder for a game when none was explicitly supplied,
// so existing call sites (inference server, arena, standalone MCTS) that don't
// care about encoder variants keep working with no changes, while a caller that
// wants a specific encoding (e.g. a future history-stacked chess encoder) can
// still pass one explicitly. Dispatches on the concrete game type.
std::shared_ptr<StateEncoder> default_encoder_for(const GameState &state);

#endif // ENCODER_FACTORY_HPP
