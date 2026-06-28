// Original source: https://github.com/geochri/AlphaZero_Chess/blob/master/src/chess_board.py
#ifndef CHESS_HPP
#define CHESS_HPP

#include "game.hpp"
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

class Chess : public Game2D<8, 8> {
  private:
    board_t current_board;
    int8_t player; // 0 for white, 1 for black
    int16_t move_count;
    int8_t en_passant;
    int16_t en_passant_move;
    int16_t r1_move_count, r2_move_count, k_move_count;
    int16_t R1_move_count, R2_move_count, K_move_count;

    board_t copy_board;
    int8_t en_passant_copy;
    int16_t r1_move_count_copy, r2_move_count_copy, k_move_count_copy;
    int16_t R1_move_count_copy, R2_move_count_copy, K_move_count_copy;
    int16_t move_count_copy;
    int16_t en_passant_move_copy;

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
    void write_canonical_state(float *out_buffer) const override;
    std::vector<int64_t> get_state_shape() const override;

  private:
    using pos_t = std::pair<int8_t, int8_t>;

    //  Helper functions
    static bool is_white(int8_t p);
    static bool is_black(int8_t p);
    static bool is_empty(int8_t p);

    std::pair<std::vector<pos_t>, std::vector<pos_t>> move_rules_P(int8_t i, int8_t j) const;
    std::pair<std::vector<pos_t>, std::vector<pos_t>> move_rules_p(int8_t i, int8_t j) const;
    std::vector<pos_t> move_rules_r(int8_t i, int8_t j) const;
    std::vector<pos_t> move_rules_R(int8_t i, int8_t j) const;
    std::vector<pos_t> move_rules_n(int8_t i, int8_t j) const;
    std::vector<pos_t> move_rules_N(int8_t i, int8_t j) const;
    std::vector<pos_t> move_rules_b(int8_t i, int8_t j) const;
    std::vector<pos_t> move_rules_B(int8_t i, int8_t j) const;
    std::vector<pos_t> move_rules_q(int8_t i, int8_t j) const;
    std::vector<pos_t> move_rules_Q(int8_t i, int8_t j) const;
    std::vector<pos_t> move_rules_k() const;
    std::vector<pos_t> move_rules_K() const;

    std::vector<pos_t> possible_W_moves(bool threats = false) const;
    std::vector<pos_t> possible_B_moves(bool threats = false) const;

  public:
    void move_piece(int r1, int c1, int r2, int c2, int promoted_piece = W_QUEEN);
    bool castle(int side, bool inplace = false); // side: 0=queenside, 1=kingside
    bool check_status() const;
    std::vector<ChessAction<>> actions(bool stop_early = false) const;

    int encode_action(const ChessAction<> &a) const;
    ChessAction<> decode_action(int a) const;

    void backup();
    void restore();
};

#endif
