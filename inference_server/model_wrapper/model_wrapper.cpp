#include "model_wrapper.hpp"

#include <chess.hpp>
#include <chess_encoder_v2history.hpp>
#include <chrono>
#include <connect4.hpp>
#include <mcts.hpp>
#include <nlohmann/json.hpp>
#include <random>
#include <spdlog/spdlog.h>
#include <state_encoder.hpp>
#include <torch/torch.h>
#include <utility>

namespace {

// Runs one MCTS search reproducing training_params/*.json's self-play search
// config: a full_search_probability-weighted coin flip between the full
// mcts_search_depth and fast_mcts_simulations, then either plain-PUCT
// search() or Gumbel-Top-k search_gumbel() depending on use_gumbel_search.
// Both paths return a plain (policy, value) pair, matching what this file's
// two predict() methods already expect - search_gumbel()'s gumbel_result
// carries additional policy-improvement-guaranteed move info
// (chosen_action) that mcts.hpp documents as the "correct" move to play,
// but the wire response format here only ever encoded a policy distribution
// + value (the same contract search() has always had), so this returns pi
// unchanged rather than widening that contract.
std::pair<std::vector<float>, float>
run_search(MCTS &mcts, const Game &game, int mcts_search_depth, int mcts_batch_size,
          bool use_gumbel_search, int max_num_considered_actions, float full_search_probability,
          int fast_mcts_simulations) {
    int num_simulations = mcts_search_depth;
    if (full_search_probability < 1.0f) {
        // thread_local: predict() may be called from multiple Crow worker
        // threads concurrently; a shared std::mt19937 would need external
        // locking, this doesn't.
        thread_local std::mt19937 rng(std::random_device{}());
        thread_local std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        if (unit(rng) >= full_search_probability) {
            num_simulations = fast_mcts_simulations > 0 ? fast_mcts_simulations : mcts_search_depth;
        }
    }

    if (use_gumbel_search) {
        auto result =
            mcts.search_gumbel(game, num_simulations, mcts_batch_size, max_num_considered_actions);
        return {std::move(result.pi), result.root_value};
    }
    return mcts.search(game, num_simulations, mcts_batch_size);
}

} // namespace

namespace {

class Connect4ModelWrapper final : public ModelWrapper {
    torch::Device device;
    MCTS mcts;
    int search_depth;
    int batch_size;
    bool use_gumbel_search;
    int max_num_considered_actions;
    float full_search_probability;
    int fast_mcts_simulations;

  public:
    Connect4ModelWrapper(std::string network_path, std::string device, int search_depth,
                         int batch_size, bool use_gumbel_search, int max_num_considered_actions,
                         float full_search_probability, int fast_mcts_simulations,
                         float dirichlet_epsilon)
        : device(torch::Device(std::move(device))),
          mcts(std::move(network_path), this->device, 1.25f, 19652.0f, dirichlet_epsilon),
          search_depth(search_depth), batch_size(batch_size), use_gumbel_search(use_gumbel_search),
          max_num_considered_actions(max_num_considered_actions),
          full_search_probability(full_search_probability),
          fast_mcts_simulations(fast_mcts_simulations) {}

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
        const auto [policy, value] =
            run_search(mcts, game, search_depth, batch_size, use_gumbel_search,
                      max_num_considered_actions, full_search_probability, fast_mcts_simulations);
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
    bool use_gumbel_search;
    int max_num_considered_actions;
    float full_search_probability;
    int fast_mcts_simulations;

  public:
    // chess_encoder_history: 0 => default 19-plane ChessEncoderV1 (unchanged
    // behavior); N in {1,4,8} => ChessEncoderV2History(N) so a history-encoder
    // net is fed its own input (e.g. puzzle-testing the chess-v2 history net).
    ChessModelWrapper(std::string network_path, std::string device, int search_depth,
                      int batch_size, int chess_encoder_history, bool use_gumbel_search,
                      int max_num_considered_actions, float full_search_probability,
                      int fast_mcts_simulations, float dirichlet_epsilon)
        : device(torch::Device(std::move(device))),
          mcts(std::move(network_path), this->device, 1.25f, 19652.0f, dirichlet_epsilon, 0.3f,
               default_arena_size_in_bytes, 0.0f,
               [chess_encoder_history]() -> std::shared_ptr<StateEncoder> {
                   if (chess_encoder_history > 0) {
                       return std::make_shared<ChessEncoderV2History>(chess_encoder_history);
                   }
                   return nullptr;
               }()),
          search_depth(search_depth), batch_size(batch_size), use_gumbel_search(use_gumbel_search),
          max_num_considered_actions(max_num_considered_actions),
          full_search_probability(full_search_probability),
          fast_mcts_simulations(fast_mcts_simulations) {}

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
        const auto [policy, value] =
            run_search(mcts, game, search_depth, batch_size, use_gumbel_search,
                      max_num_considered_actions, full_search_probability, fast_mcts_simulations);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        spdlog::info("Chess prediction finished in {} ms (batch size: {})", duration.count(),
                     board_json.size());

        return encode_payload(policy, value);
    }
};

} // namespace

std::shared_ptr<ModelWrapper> create_connect4_model_wrapper(
    const std::string &network_path, const std::string &device, int mcts_search_depth,
    int mcts_batch_size, bool use_gumbel_search, int max_num_considered_actions,
    float full_search_probability, int fast_mcts_simulations, float dirichlet_epsilon) {
    return std::make_shared<Connect4ModelWrapper>(
        network_path, device, mcts_search_depth, mcts_batch_size, use_gumbel_search,
        max_num_considered_actions, full_search_probability, fast_mcts_simulations,
        dirichlet_epsilon);
}

std::shared_ptr<ModelWrapper> create_chess_model_wrapper(
    const std::string &network_path, const std::string &device, int mcts_search_depth,
    int mcts_batch_size, int chess_encoder_history, bool use_gumbel_search,
    int max_num_considered_actions, float full_search_probability, int fast_mcts_simulations,
    float dirichlet_epsilon) {
    return std::make_shared<ChessModelWrapper>(
        network_path, device, mcts_search_depth, mcts_batch_size, chess_encoder_history,
        use_gumbel_search, max_num_considered_actions, full_search_probability,
        fast_mcts_simulations, dirichlet_epsilon);
}
