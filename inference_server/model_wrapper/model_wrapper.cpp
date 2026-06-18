#include "model_wrapper.hpp"

#include <chrono>
#include <connect4.hpp>
#include <mcts.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <torch/torch.h>
#include <utility>
#include <vector>

namespace {

class Connect4ModelWrapper final : public ModelWrapper {
  torch::Device device;
  MCTS mcts;

public:
  Connect4ModelWrapper(std::string network_path, std::string device)
      : device(torch::Device(std::move(device))),
        mcts(std::move(network_path), this->device) {}

  std::string predict(const std::string &request_payload) override {
    const auto payload_json = nlohmann::json::parse(request_payload);
    const auto &board_json = payload_json.at("board");

    std::vector<std::vector<int>> board;
    board.reserve(board_json.size());
    for (const auto &row_json : board_json) {
      std::vector<int> row;
      row.reserve(row_json.size());
      for (const auto &cell_json : row_json) {
        int value = cell_json.get<int>();
        if (value == 2) {
          value = -1;
        }
        row.push_back(value);
      }
      board.push_back(std::move(row));
    }

    Connect4 game(board, device);

    auto start = std::chrono::high_resolution_clock::now();
    const auto policy = mcts.search(game);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // TODO: Make this log better
    spdlog::info("Elapsed time: {} ms", duration.count());

    return nlohmann::json{{"policy", policy}}.dump();
  }
};

} // namespace

std::shared_ptr<ModelWrapper>
create_connect4_model_wrapper(const std::string &network_path,
                              const std::string &device) {
  return std::make_shared<Connect4ModelWrapper>(network_path, device);
}
