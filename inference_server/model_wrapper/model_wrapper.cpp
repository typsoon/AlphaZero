#include "model_wrapper.hpp"

#include <chrono>
#include <connect4.hpp>
#include <mcts.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <torch/torch.h>
#include <utility>

namespace {

class Connect4ModelWrapper final : public ModelWrapper {
    torch::Device device;
    MCTS mcts;
    int search_depth;
    int batch_size;

  public:
    Connect4ModelWrapper(std::string network_path, std::string device, int search_depth,
                         int batch_size)
        : device(torch::Device(std::move(device))), mcts(std::move(network_path), this->device),
          search_depth(search_depth), batch_size(batch_size) {}

    std::string predict(const std::string &request_payload) override {
        const auto payload_json = nlohmann::json::parse(request_payload);
        const auto &board_json = payload_json.at("board");

        Connect4::board_t board = {};
        for (int r = 0; r < Connect4::ROWS; r++) {
            for (int c = 0; c < Connect4::COLS; c++) {
                int value = board_json[r][c].get<int>();
                if (value == 2) {
                    value = -1;
                }
                board[r][c] = value; // NOLINT
            }
        }

        Connect4 game(board);

        auto start = std::chrono::high_resolution_clock::now();
        const auto [policy, value] = mcts.search(game, search_depth, batch_size);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        spdlog::info("Connect4 prediction finished in {} ms (batch size: {})", duration.count(),
                     board_json.size());

        return nlohmann::json{{"policy", policy}, {"value", value}}.dump();
    }
};

} // namespace

std::shared_ptr<ModelWrapper> create_connect4_model_wrapper(const std::string &network_path,
                                                            const std::string &device,
                                                            int mcts_search_depth,
                                                            int mcts_batch_size) {
    return std::make_shared<Connect4ModelWrapper>(network_path, device, mcts_search_depth,
                                                  mcts_batch_size);
}
