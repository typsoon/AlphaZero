#ifndef CHESS_ENCODER_V2HISTORY_HPP
#define CHESS_ENCODER_V2HISTORY_HPP

#include "state_encoder.hpp"

// History-stacked chess encoding, ported to match the friend's engine-zoo
// reference (crates/algorithms/src/alphazero/network/chess_v2.rs +
// crates/games/src/chess/az.rs `ChessAzState<HISTORY>::encode_state`)
// plane-for-plane: `history` frames (current position first, most recent
// history next, oldest last; frames before the game started are all-zero) of
// 14 planes each - 6 own-piece + 6 opponent-piece planes (P,N,B,R,Q,K order,
// matching this engine's own Piece enum) plus 2 repetition-flag planes
// (>=1 / >=2 prior occurrences of that exact frame's position, within the
// retained window - independent of Chess's full-game FIDE repetition
// tracking) - followed by 7 auxiliary planes for the CURRENT position: side
// to move, own/opponent kingside/queenside castling rights, halfmove clock
// (/100, clamped), fullmove number (/200, clamped). Total channels =
// 14*history + 7 (63 for the reference's default history=4).
//
// One deliberate, called-out deviation from the reference: engine-zoo flips
// the board for WHITE to move (row = 7-rank) and leaves Black unflipped, while
// this class DEFAULTS to reusing THIS engine's existing convention from
// ChessEncoderV1 (White unflipped, Black flipped) for every frame, since that
// convention is already baked into this repo's tested Gumbel-policy
// action-map machinery (python/network.py's canonical-frame index maps) and
// the two conventions are otherwise equivalent for a network THIS repo trains
// itself - both are internally consistent, arbitrary choices from a learning
// standpoint. All plane CONTENT, ordering, and the 7 auxiliary planes match
// the reference exactly regardless of `flip_white`.
//
// `flip_white`: pass true to reproduce the reference's OWN convention exactly
// (White flipped, Black unflipped) instead of this repo's default. Required
// when serving weights TRANSPLANTED from the reference (e.g. via
// convert_safetensors_v2.py) - those weights were trained to interpret the
// reference's row orientation, not this engine's; feeding them the default
// (opposite) orientation silently corrupts the input for every position.
// Must be paired with a network built with the matching
// ChessAzV2Network(action_convention="engine_zoo") action-index maps (see
// python/network.py) - the encoder's row orientation and the network's
// canonical-to-engine action translation must agree on which color's frame is
// "canonical". Never needed for a network THIS repo trained itself.
//
// `history` must be 1, 4, or 8 (matching ChessAzV2Config::validate in the
// reference); default 4 matches the reference's default. Reading more than
// `Chess::kMaxHistoryFrames` (8) is not possible - Chess only retains that many
// prior board snapshots (see its history_boards comment).
class ChessEncoderV2History : public StateEncoder {
  public:
    explicit ChessEncoderV2History(int history = 4, bool flip_white = false);

    void write_canonical_state(const GameState &state, float *out) const override;
    std::vector<int64_t> state_shape() const override;

    int history() const { return history_; }
    bool flip_white() const { return flip_white_; }

  private:
    int history_;
    bool flip_white_;
};

#endif // CHESS_ENCODER_V2HISTORY_HPP
