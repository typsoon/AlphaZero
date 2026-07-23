// Original source: https://github.com/geochri/AlphaZero_Chess/blob/master/src/chess_board.py
#ifndef CHESS_HPP
#define CHESS_HPP

#include "game.hpp"
#include <array>
#include <cstdint>
#include <spdlog/spdlog.h>
#include <vector>

// NOLINTNEXTLINE(cppcoreguidelines-use-enum-class)
enum Piece : int8_t {
    EMPTY = 0,
    W_PAWN = 1,
    W_KNIGHT = 2,
    W_BISHOP = 3,
    W_ROOK = 4,
    W_QUEEN = 5,
    W_KING = 6,
    B_PAWN = -1,
    B_KNIGHT = -2,
    B_BISHOP = -3,
    B_ROOK = -4,
    B_QUEEN = -5,
    B_KING = -6
};

template <typename T = int8_t> struct ChessAction {
    T r1, c1;
    T r2, c2;
    T promotion; // 0=None, 1=queen, 2=rook, 3=knight, 4=bishop

    ChessAction() = default;
    ChessAction(T r1, T c1, T r2, T c2, T promotion)
        : r1(r1), c1(c1), r2(r2), c2(c2), promotion(promotion) {}
};

// Capacity: the documented theoretical maximum for a *legal* chess position is
// 218 moves (e.g. "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1"), and
// this generator's own promotion handling (emplacing all 4 promotion choices
// per promoting pawn move, in Chess::actions()) is already folded into that
// figure's usual derivation. 320 keeps a comfortable margin above that bound -
// self-play with an early/untrained network can reach unusual multi-queen
// positions well outside "sensible" play, and list/_size are adjacent fields
// in this struct with no bounds check on push_back()/emplace_back() below, so
// silently overflowing 256 here (the previous capacity, only ~1.17x the
// documented bound) would write straight through _size into whatever memory
// follows it - undefined behavior that previously manifested as legal_actions
// entries decoding to structurally-impossible squares (e.g. row 8 on an 8x8
// board) during a real training run. See push_back()/emplace_back()'s own
// bounds check below for the second, independent layer of defense against
// this - this capacity bump alone is not assumed sufficient on its own.
template <typename T = int8_t> struct ActionList {
    static constexpr int kCapacity = 320;
    std::array<ChessAction<T>, kCapacity> list{};
    int _size = 0;

    // Bounds-checked: unlike a raw `list[_size++] = ...`, this can never write
    // past `list` even if some future rule change (or an unanticipated
    // position) pushes the true legal-move count past kCapacity - silently
    // dropping the excess is far preferable to corrupting whatever field or
    // stack memory happens to follow `list`. Logged once per overflow (not
    // every dropped entry) since this is called from a hot path and is only
    // ever expected to fire on a genuine bug or a not-yet-anticipated
    // position, not in steady-state operation.
    void push_back(const ChessAction<T> &a) {
        if (_size >= kCapacity) {
            spdlog::error("ActionList overflow: dropping action beyond capacity={}", kCapacity);
            return;
        }
        list[_size++] = a;
    }
    void emplace_back(T r1, T c1, T r2, T c2, T promo) { push_back({r1, c1, r2, c2, promo}); }
    const ChessAction<T> *begin() const { return list.data(); }
    const ChessAction<T> *end() const { return list.data() + _size; }
    ChessAction<T> *begin() { return list.data(); }
    ChessAction<T> *end() { return list.data() + _size; }
    bool empty() const { return _size == 0; }
    int size() const { return _size; }
};

// Capacity: bounds the pseudo-legal destination squares for a single piece
// (moves.clear() below resets this once per piece in Chess::actions()), not
// the whole side's legal-move list (see ActionList above for that). A queen's
// real maximum is 27 (13 diagonal + 14 orthogonal from the center of an empty
// board); 32 already carries a margin over that, but push_back()/emplace_back()
// below is bounds-checked the same way as ActionList's, for the same reason -
// no code path here should ever come close to exceeding a piece's real move
// count, but silently overflowing into whatever follows `list` in memory is a
// worse failure mode than dropping an entry ever would be.
struct PosList {
    static constexpr int kCapacity = 32;
    std::array<std::pair<int8_t, int8_t>, kCapacity> list;
    int _size = 0;
    void emplace_back(int8_t r, int8_t c) {
        if (_size >= kCapacity) {
            spdlog::error("PosList overflow: dropping move beyond capacity={}", kCapacity);
            return;
        }
        list[_size++] = {r, c};
    }
    const std::pair<int8_t, int8_t> *begin() const { return list.data(); }
    const std::pair<int8_t, int8_t> *end() const { return list.data() + _size; }
    int size() const { return _size; }
    void clear() { _size = 0; }
};

class Chess : public Game2D<8, 8> {
    // Reads Chess's private position (board, castling counts, en passant, move
    // count, side to move) to build the 19-plane input tensor. Kept out of the
    // Chess class itself so alternative encodings need no changes here. See
    // engine/game/chess_encoder.hpp.
    friend class ChessEncoderV1;
    // Reads the rolling board/repetition history below to build the
    // engine-zoo-style history-stacked input tensor. See
    // engine/game/chess_encoder_v2history.hpp.
    friend class ChessEncoderV2History;

  private:
    // Fields are grouped by alignment (8-byte vector, then 2-byte ints, then
    // 1-byte ints, then the align-1 board array last) rather than declared in
    // logical/usage order, to minimize padding - Chess is cloned via
    // std::shared_ptr<Game> once per MCTS leaf explored (up to hundreds of times
    // per search), so its size multiplies directly into self-play's allocation
    // traffic.

    // One Zobrist hash per position reached so far in this game (including the
    // starting position), in order. Used for threefold-repetition detection - see
    // is_threefold_repetition(). Grows by one entry per ply; bounded in practice by
    // whatever max_moves the caller enforces (self-play/training default 512).
    std::vector<uint64_t> position_history;

    int16_t move_count{};
    int16_t en_passant_move{};
    int16_t r1_move_count{}, r2_move_count{}, k_move_count{};
    int16_t R1_move_count{}, R2_move_count{}, K_move_count{};
    // Plies since the last pawn move or capture. FIDE's fifty-move rule is 50 full
    // moves (100 plies) without progress by either side - see is_fifty_move_draw().
    int16_t halfmove_clock{};

    int8_t player{}; // 0 for white, 1 for black
    int8_t en_passant{};
    // How many times the *current* (last-pushed) position has occurred so far,
    // including itself. Recomputed once per move in move_piece() rather than
    // rescanned on every is_terminal()/reward() call, since is_terminal() in
    // particular is called very frequently (once per MCTS tree node visited, once
    // per self-play ply) relative to how often the position actually changes.
    int8_t repetition_count{1};

    // Rolling window of PRE-move board snapshots (most recent at index 0),
    // pushed in move_piece() before the board is mutated, plus each snapshot's
    // own "repetitions before" count. Exists purely to feed
    // ChessEncoderV2History's history-stacked planes (mirrors engine-zoo's
    // ChessAzState<HISTORY>, which keeps the identical data directly on its
    // game-state struct) - unrelated to and independent from
    // position_history/repetition_count above, which serve FIDE draw
    // detection over the *entire* game rather than a bounded window. Sized
    // for the largest history length ChessEncoderV2History supports (8);
    // history_count tracks how many entries are actually valid (0 right
    // after reset()/set_custom_state(), growing by one per move_piece() call,
    // capped at kMaxHistoryFrames) so encoders can zero-fill the rest, matching
    // engine-zoo's Option<ChessPosition> "no history yet" semantics.
    static constexpr int kMaxHistoryFrames = 8;
    std::array<board_t, kMaxHistoryFrames> history_boards{};
    std::array<int8_t, kMaxHistoryFrames> history_repetitions_before{};
    int history_count{0};

    board_t current_board{};

  public:
    static constexpr int action_dim = 64 * 64 * 5;
    static constexpr std::array<int, 3> state_dim = {19, 8, 8};

    void set_custom_state(const board_t &board, int8_t active_player, int8_t en_passant_col = -1,
                          int16_t k_mc = 0, int16_t r1_mc = 0, int16_t r2_mc = 0, int16_t K_mc = 0,
                          int16_t R1_mc = 0, int16_t R2_mc = 0);

    Chess();

    void reset() override;
    int getActionSize() const override;
    std::vector<int> get_legal_actions() const override;
    void step(int action) override;
    bool is_terminal() const override;
    int get_current_player() const override;
    float reward() const override;
    std::shared_ptr<const GameState> get_canonical_state() const override;
    std::shared_ptr<Game> clone() const override;
    void render() const override;
    board_t get_board_state() const override;

    // FIDE's fifty-move rule: automatic draw once 100 plies (50 full moves) have
    // passed with no pawn move and no capture by either side.
    bool is_fifty_move_draw() const;
    // Automatic draw once the current position (same piece placement, same side to
    // move, same castling rights, same en-passant capture availability) has occurred
    // for the third time - see position_history/repetition_count above.
    bool is_threefold_repetition() const;

  private:
    using pos_t = std::pair<int8_t, int8_t>;

    // Zobrist hash of the current position (board + side to move + castling rights +
    // en-passant file, if currently capturable) - see chess.cpp for the key table.
    uint64_t compute_position_hash() const;

    //  Helper functions
    static bool is_white(int8_t p);
    static bool is_black(int8_t p);
    static bool is_empty(int8_t p);

    void move_rules_P(int8_t i, int8_t j, PosList &moves) const;
    void move_rules_p(int8_t i, int8_t j, PosList &moves) const;
    void move_rules_r(int8_t i, int8_t j, PosList &moves) const;
    void move_rules_R(int8_t i, int8_t j, PosList &moves) const;
    void move_rules_n(int8_t i, int8_t j, PosList &moves) const;
    void move_rules_N(int8_t i, int8_t j, PosList &moves) const;
    void move_rules_b(int8_t i, int8_t j, PosList &moves) const;
    void move_rules_B(int8_t i, int8_t j, PosList &moves) const;
    void move_rules_q(int8_t i, int8_t j, PosList &moves) const;
    void move_rules_Q(int8_t i, int8_t j, PosList &moves) const;
    void move_rules_k(PosList &moves) const;
    void move_rules_K(PosList &moves) const;

  public:
    void move_piece(int r1, int c1, int r2, int c2, int promoted_piece = W_QUEEN);
    bool can_castle(int side) const;
    bool castle(int side, bool inplace = false); // side: 0=queenside, 1=kingside
    bool check_status() const;
    ActionList<> actions(bool stop_early = false) const;

    static int encode_action(const ChessAction<> &a);
    static ChessAction<> decode_action(int a);
};

#endif
