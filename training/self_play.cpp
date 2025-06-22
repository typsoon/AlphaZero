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
#include <random>
#include <vector>

using std::string;

static torch::Tensor vector_to_tensor(std::vector<float> &data) {
    // Optionally specify size if reshaping is needed
    return torch::from_blob(data.data(), {static_cast<long>(data.size())},
                            torch::kFloat);
}

static void play_game(std::unique_ptr<Game> game, std::unique_ptr<MCTS> mcts,
                      ReplayBuffer &replay_buffer) {
    game->reset();
    std::vector<Transition> trajectory;

    while (!game->is_terminal()) {
        auto canonical_state = game->get_canonical_state();
        auto policy = mcts->search(*game);

        // Pick action based on policy probability distribution
        std::discrete_distribution<int> dist(policy.begin(), policy.end());
        static std::mt19937 rng(std::random_device{}());
        int action = dist(rng);

        trajectory.emplace_back(std::move(canonical_state),
                                vector_to_tensor(policy).clone(), 0);
        game->step(action);
    }

    float value = game->reward();

    for (size_t i = 0; i < trajectory.size(); ++i) {
        // update last element in tuple with value or -value depending on i
        // parity
        (trajectory[i]).reward = (i % 2 == 0) ? value : -value;
    }

    replay_buffer.add(trajectory);
}

void self_play(std::shared_ptr<Game> initial_game, string network_path,
               ReplayBuffer &replay_buffer, int num_games, int thread_count) {
    auto device = torch::Device(torch::cuda::is_available() ? "cuda" : "cpu");
    // std::cerr << device << '\n';

    auto game = std::make_unique<Connect4>(device);

    auto inferer_factory = NetworkInfererFactory(network_path, device);
    MCTSFactory mcts_factory(inferer_factory);

    std::atomic<int> games_played{0};
    std::mutex cout_mutex;

    auto worker = [&]() {
        while (true) {
            int current = games_played.fetch_add(1);
            if (current >= num_games)
                break;

            auto mcts = mcts_factory.get_mcts();
            play_game(game->clone(), std::move(mcts), replay_buffer);
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "Games played: " << current + 1 << "/" << num_games
                          << "\n";
            }
        }
    };

    // worker();

    std::vector<std::thread> threads;
    for (int i = 1; i < thread_count; ++i) {
        threads.emplace_back(worker);
    }

    worker();

    for (auto &t : threads) {
        t.join();
    }
}
