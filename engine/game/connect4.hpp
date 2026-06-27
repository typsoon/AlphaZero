#ifndef CONNECT_4_HPP
#define CONNECT_4_HPP

#include "game.hpp"
#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <memory>
#include <torch/torch.h>
#include <vector>

using std::vector;

class Connect4 : public Game2D<6, 7> {
  public:
    static constexpr int action_dim = COLS;
    static constexpr auto state_dim = std::make_tuple(1, ROWS, COLS);

    // Static utilities for board evaluation (testable and reusable)
    static bool hasWin(const board_t &board, int player);
    static bool hasWin(const std::vector<std::vector<int>> &board, int player);
    static bool isBoardFull(const board_t &board);
    static bool isBoardFull(const std::vector<std::vector<int>> &board);

    Connect4(torch::Device device = torch::Device("cpu"));
    Connect4(const board_t &initial_board, torch::Device device = torch::Device("cpu"));
    Connect4(const std::vector<std::vector<int>> &initial_board,
             torch::Device device = torch::Device("cpu"));
    ~Connect4() override = default;

    void reset() override;
    int getActionSize() const override;
    int get_current_player() const override;
    std::vector<int> get_legal_actions() const override;
    void step(int action) override;
    bool is_terminal() const override;
    float reward() const override;
    std::vector<std::vector<int>> get_board_state() const override;
    GameState get_canonical_state() const override;
    void write_canonical_state(float *out_buffer) const override;
    std::unique_ptr<Game> clone() const override;
    void render() const override;

  private:
    void reset_initial_state();
    torch::Device device;

    bool checkWin(int row, int col) const;
    bool checkDirection(int row, int col, int dRow, int dCol) const;

    board_t board;
    int currentPlayer;
    bool finished;
    float _reward;
};

#endif // CONNECT_4_HPP
