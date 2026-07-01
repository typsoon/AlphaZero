#include "model_wrapper.hpp"

#include <chess.hpp>
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

    std::string encode_payload(const std::vector<float> &policy, float value) override {
        return nlohmann::json{{"policy", policy}, {"value", value}}.dump();
    }
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

        return encode_payload(policy, value);
    }
};

class ChessModelWrapper final : public ModelWrapper {
    torch::Device device;
    MCTS mcts;
    int search_depth;
    int batch_size;

  public:
    ChessModelWrapper(std::string network_path, std::string device, int search_depth,
                      int batch_size)
        : device(torch::Device(std::move(device))), mcts(std::move(network_path), this->device),
          search_depth(search_depth), batch_size(batch_size) {}

    std::string encode_payload(const std::vector<float> &policy, float value) override {
        nlohmann::json sparse_policy = nlohmann::json::array();
        for (size_t i = 0; i < policy.size(); ++i) {
            if (policy[i] > 1e-6) {
                sparse_policy.push_back({{"index", i}, {"value", policy[i]}});
            }
        }
        return nlohmann::json{{"policy", sparse_policy}, {"value", value}}.dump();
    }
    std::string predict(const std::string &request_payload) override {
        const auto payload_json = nlohmann::json::parse(request_payload);
        const auto &board_json = payload_json.at("board");

        Chess::board_t board = {};
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                board[r][c] = board_json[r][c].get<int>();
            }
        }

        int8_t player = payload_json.at("player").get<int>();
        int8_t en_passant = payload_json.at("en_passant").get<int>();
        auto castling = payload_json.at("castling").get<std::vector<int>>();

        Chess game;
        game.set_custom_state(board, player, en_passant, castling[0], castling[1], castling[2],
                              castling[3], castling[4], castling[5]);

        auto start = std::chrono::high_resolution_clock::now();
        const auto [policy, value] = mcts.search(game, search_depth, batch_size);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        spdlog::info("Chess prediction finished in {} ms (batch size: {})", duration.count(),
                     board_json.size());

        return encode_payload(policy, value);
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

std::shared_ptr<ModelWrapper> create_chess_model_wrapper(const std::string &network_path,
                                                         const std::string &device,
                                                         int mcts_search_depth,
                                                         int mcts_batch_size) {
    return std::make_shared<ChessModelWrapper>(network_path, device, mcts_search_depth,
                                               mcts_batch_size);
}
