#ifndef CHESS_ENCODER_V2HISTORY_HPP
#define CHESS_ENCODER_V2HISTORY_HPP

#include "state_encoder.hpp"

// History-stacked chess encoding, ported to match the friend's engine-zoo
// reference (crates/alphazero/src/representation/chess_v2.rs, `cell()` +
// `encode_state()`) plane-for-plane: `history` frames (current position
// first, most recent history next, oldest last; frames before the game
// started are all-zero) of 14 planes each - 6 own-piece + 6 opponent-piece
// planes (P,N,B,R,Q,K order, matching this engine's own Piece enum) plus 2
// repetition-flag planes (>=1 / >=2 prior occurrences of that exact frame's
// position, within the retained window - independent of Chess's full-game
// FIDE repetition tracking) - followed by 7 auxiliary planes for the CURRENT
// position: side to move, own/opponent kingside/queenside castling rights,
// halfmove clock (/100, clamped), fullmove number (/200, clamped). Total
// channels = 14*history + 7 (63 for the reference's default history=4).
//
// CORRECTED 2026-08-05 (was wrong below until now - see [[chess-v2-flip-white-bugfix]]):
// the DEFAULT (flip_white=false) already reproduces the reference's row
// orientation exactly, verified by tracing both engines' real row arithmetic
// down to concrete squares rather than trusting either side's comments.
// Reference: for the mover's own frame, row = 7 - rank_index (rank "1" = 0,
// per engine-zoo's own chess crate's Rank::to_index()); for the opponent's
// frame, row = rank_index (`crates/alphazero/src/representation/chess_v2.rs`
// `cell()`). This engine's `encode_frame` (see the .cpp) produces the
// identical mapping under flip_white=false: perspective-to-move frames
// unflipped (i = board_row = 7-rank_index), the other side's frames flipped
// (i = rank_index) - confirmed with a concrete worked example (White king on
// e1, Black king on e8, White to move -> both engines place the White king's
// plane bit on tensor row 7 and Black's on row 0 under this default). All
// plane CONTENT, ordering, and the 7 auxiliary planes match the reference
// exactly regardless of `flip_white`.
//
// `flip_white`: pass true for the OPPOSITE of the reference's row
// orientation - a full top/bottom mirror of every position relative to what
// engine-zoo actually produces. NOT needed (and actively wrong) when serving
// weights TRANSPLANTED from the reference (e.g. via convert_safetensors_v2.py)
// - use the default `false`. Kept only as a knob for exploring the mirrored
// orientation; there is currently no known use case that wants it.
// Must be paired with a network built with the matching
// ChessAzV2Network(action_convention="engine_zoo") action-index maps (see
// python/network.py) for the plane-ORDER correction (knight-move/
// underpromotion geometry) - that is a separate, independent mechanism from
// this row-orientation flip (the row-flip assignment is the same for both
// action_convention values; see _build_chess_v2_action_maps's docstring).
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
