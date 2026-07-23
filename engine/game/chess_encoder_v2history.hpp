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
// the board for WHITE to move (row = 7-rank) and leaves Black unflipped: this
// class instead reuses THIS engine's existing convention from ChessEncoderV1
// (White unflipped, Black flipped) for every frame, since that convention is
// already baked into this repo's tested Gumbel-policy action-map machinery
// (python/network.py's canonical-frame index maps) and the two conventions
// are otherwise equivalent - both are internally consistent, arbitrary
// choices from a learning standpoint. All plane CONTENT, ordering, and the 7
// auxiliary planes match the reference exactly.
//
// `history` must be 1, 4, or 8 (matching ChessAzV2Config::validate in the
// reference); default 4 matches the reference's default. Reading more than
// `Chess::kMaxHistoryFrames` (8) is not possible - Chess only retains that many
// prior board snapshots (see its history_boards comment).
class ChessEncoderV2History : public StateEncoder {
  public:
    explicit ChessEncoderV2History(int history = 4);

    void write_canonical_state(const GameState &state, float *out) const override;
    std::vector<int64_t> state_shape() const override;

    int history() const { return history_; }

  private:
    int history_;
};

#endif // CHESS_ENCODER_V2HISTORY_HPP
