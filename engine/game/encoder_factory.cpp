#include "encoder_factory.hpp"

#include "chess.hpp"
#include "chess_encoder.hpp"
#include "connect4.hpp"
#include "connect4_encoder.hpp"
#include <stdexcept>

std::shared_ptr<StateEncoder> default_encoder_for(const GameState &state) {
    if (dynamic_cast<const Chess *>(&state) != nullptr) {
        return std::make_shared<ChessEncoderV1>();
    }
    if (dynamic_cast<const Connect4 *>(&state) != nullptr) {
        return std::make_shared<Connect4Encoder>();
    }
    throw std::runtime_error("default_encoder_for: no default StateEncoder for this game type");
}
