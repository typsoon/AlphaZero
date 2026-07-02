#include "../../engine/game/chess.hpp"
#include "../../engine/game/connect4.hpp"
#include "../../engine/utils/replay_buffer.hpp"
#include "../../training/self_play.hpp"
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <torch/csrc/autograd/profiler_kineto.h>

int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <game> <network_path> <num_games> <thread_count> [max_moves] "
                     "[kineto_out] [mcts_num_simulations] [mcts_batch_size]\n"
                  << "  Pass an empty string (\"\") for kineto_out to skip kineto profiling "
                     "while still setting mcts_num_simulations/mcts_batch_size.\n";
        return 1;
    }

    std::string game_name = argv[1];
    std::string network_path = argv[2];
    int num_games = std::stoi(argv[3]);
    int thread_count = std::stoi(argv[4]);
    int max_moves = 512;
    if (argc >= 6) {
        max_moves = std::stoi(argv[5]);
    }

    std::string kineto_out_file = (argc >= 7) ? argv[6] : "";
    bool use_kineto = !kineto_out_file.empty();
    int mcts_num_simulations = (argc >= 8) ? std::stoi(argv[7]) : 800;
    int mcts_batch_size = (argc >= 9) ? std::stoi(argv[8]) : 32;

    if (use_kineto) {
        torch::profiler::impl::ProfilerConfig config(torch::profiler::impl::ProfilerState::KINETO,
                                                     false, // report_input_shapes
                                                     false, // profile_memory
                                                     false, // with_stack
                                                     false, // with_flops
                                                     false  // with_modules
        );
        std::set<torch::profiler::impl::ActivityType> activities = {
            torch::profiler::impl::ActivityType::CPU, torch::profiler::impl::ActivityType::CUDA};
        torch::autograd::profiler::prepareProfiler(config, activities);
        torch::autograd::profiler::enableProfiler(config, activities);
    }

    std::shared_ptr<Game> initial_game;
    if (game_name == "connect4") {
        initial_game = std::make_shared<Connect4>();
    } else if (game_name == "chess") {
        initial_game = std::make_shared<Chess>();
    } else {
        std::cerr << "Unknown game: " << game_name << '\n';
        return 1;
    }

    ReplayBuffer replay_buffer(1000000);

    std::cout << "Starting self play profiling with " << num_games << " games on " << thread_count
              << " threads... (max_moves=" << max_moves
              << ", mcts_num_simulations=" << mcts_num_simulations
              << ", mcts_batch_size=" << mcts_batch_size << ")" << '\n';

    self_play(initial_game, network_path, replay_buffer, num_games, thread_count,
              mcts_num_simulations, mcts_batch_size, max_moves);

    if (use_kineto) {
        auto profiler_result = torch::autograd::profiler::disableProfiler();
        profiler_result->save(kineto_out_file);
        std::cout << "Kineto profile saved to " << kineto_out_file << '\n';
    }

    std::cout << "Profiling completed." << '\n';
    return 0;
}
