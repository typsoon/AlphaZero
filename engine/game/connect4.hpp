#ifndef CONNECT_4_HPP
#define CONNECT_4_HPP

#include "game.hpp"
#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <memory>
#include <torch/torch.h>
#include <vector>

using std::vector;

class Connect4 : public Game {
  public:
    static constexpr int ROWS = 6;
    static constexpr int COLS = 7;
    static constexpr int action_dim = 7;
    static constexpr auto state_dim = std::make_tuple(1, 6, 7);

    Connect4(torch::Device device = torch::Device("cpu"));
    Connect4(const std::vector<std::vector<int>>& initial_board, 
             torch::Device device = torch::Device("cpu"));
    ~Connect4() override = default;

    void reset() override;
    int getActionSize() const override;
    int get_current_player() const override;
    vector<int> get_legal_actions() const override;
    void step(int action) override;
    bool is_terminal() const override;
    float reward() const override;
    std::vector<std::vector<int>> get_board_state() const override;
    GameState get_canonical_state() const override;
    std::unique_ptr<Game> clone() const override;
    void render() const override;

  private:
    const torch::Device device;

    bool checkWin(int row, int col) const;
    bool checkDirection(int row, int col, int dRow, int dCol) const;

    vector<vector<int>> board;
    int currentPlayer;
    bool finished;
    float _reward;
};

#endif // CONNECT_4_HPP
