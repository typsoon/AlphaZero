#include "self_play.hpp"
#include "game/game.hpp"

#include "game/game.hpp"
#include "inference/basic_inferer.hpp"
#include "mcts.hpp"
#include "mcts/mcts_factory.hpp"
#include "replay_buffer.hpp"
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <memory>
#include <omp.h>
#include <random>
#include <spdlog/spdlog.h>
#include <string>
#include <utility>
#include <vector>

static torch::Tensor vector_to_tensor(std::vector<float> &data) {
    // Optionally specify size if reshaping is needed
    return torch::from_blob(data.data(), {static_cast<long>(data.size())}, torch::kFloat);
}

static void play_game(std::shared_ptr<Game> game, MCTS &mcts, ReplayBuffer &replay_buffer,
                      int mcts_num_simulations, int mcts_batch_size, int max_moves) {
    game->reset();
    std::vector<Transition> trajectory;

    while (!game->is_terminal() && trajectory.size() < max_moves) {
        auto shape = game->get_state_shape();
        torch::Tensor game_state_tensor = torch::empty(shape, torch::kFloat32);
        game->write_canonical_state(game_state_tensor.data_ptr<float>());
        game_state_tensor = game_state_tensor.unsqueeze(0);
        auto [policy, root_value] = mcts.search(*game, mcts_num_simulations, mcts_batch_size);

        // Temperature scaling: tau=1 for first 30 moves, tau->0 (argmax) afterwards
        int action = -1;
        if (trajectory.size() < 30) {
            std::discrete_distribution<int> dist(policy.begin(), policy.end());
            static thread_local std::mt19937 rng(std::random_device{}());
            action = dist(rng);
        } else {
            action = std::distance(policy.begin(), std::max_element(policy.begin(), policy.end()));
        }

        trajectory.emplace_back(game_state_tensor, vector_to_tensor(policy).clone(), 0);
        game->step(action);
    }

    float value = game->reward();

    for (int i = static_cast<int>(trajectory.size()) - 1; i >= 0; --i) {
        trajectory[i].reward = value;
        value = -value;
    }

    replay_buffer.add(trajectory);
}

void self_play(std::shared_ptr<Game> initial_game, std::string network_path,
               ReplayBuffer &replay_buffer, int num_games, int thread_count,
               int mcts_num_simulations, int mcts_batch_size, int max_moves) {
    auto device = torch::Device(torch::cuda::is_available() ? "cuda" : "cpu");
    // std::cerr << device << '\n';

    // Set wait_for_count to something more reasonable like 2-4 threads worth of batches, and
    // timeout to 2ms to prevent CPU threads from starving.
    int wait_for_count = std::min(thread_count, 4) * mcts_batch_size;
    int timeout_ms = 2;
    auto inferer_factory = NetworkInfererFactory(network_path, device, wait_for_count, timeout_ms);
    MCTSFactory mcts_factory(inferer_factory);

    std::atomic<int> games_finished{0};

    // One MCTS (and its arena) per OpenMP thread, reused across every game that
    // thread picks up, instead of a fresh one per game. MCTS::search() already
    // calls pool.release() as its first line on every call - move 1 of a new game
    // resets the arena exactly the same way move 2 of an ongoing game does - so
    // reusing the same MCTS across games is already safe by construction; this only
    // avoids repeating the ~256MB allocation + first-touch page-fault cost once per
    // game instead of once per thread.
    std::vector<std::unique_ptr<MCTS>> thread_mcts;
    thread_mcts.reserve(thread_count);
    for (int t = 0; t < thread_count; t++) {
        thread_mcts.push_back(mcts_factory.get_mcts());
    }

#pragma omp parallel for schedule(dynamic) num_threads(thread_count)
    for (int i = 0; i < num_games; i++) { // NOLINT
        auto &mcts = thread_mcts[omp_get_thread_num()];
        play_game(initial_game->clone(), *mcts, replay_buffer, mcts_num_simulations,
                  mcts_batch_size, max_moves);

        auto current_finished = ++games_finished;

        spdlog::info("Games played: {}/{}", current_finished, num_games);
    }
}
