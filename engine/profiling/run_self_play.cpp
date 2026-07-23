#include "../../engine/game/chess.hpp"
#include "../../engine/game/chess_encoder_v2history.hpp"
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
                     "[kineto_out] [mcts_num_simulations] [mcts_batch_size] "
                     "[fast_mcts_num_simulations] [full_search_probability] "
                     "[use_gumbel_search] [max_num_considered_actions] "
                     "[transposition_cache_entries] [chess_encoder_history] "
                     "[value_network_path]\n"
                  << "  Pass an empty string (\"\") for kineto_out to skip kineto profiling "
                     "while still setting mcts_num_simulations/mcts_batch_size.\n"
                  << "  Pass full_search_probability=1 to disable playout cap randomization "
                     "(every move gets the full search).\n"
                  << "  Pass use_gumbel_search=1 to use MCTS::search_gumbel() instead of "
                     "MCTS::search() for every move.\n";
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
    int fast_mcts_num_simulations = (argc >= 10) ? std::stoi(argv[9]) : 100;
    float full_search_probability = (argc >= 11) ? std::stof(argv[10]) : 0.25f;
    bool use_gumbel_search = (argc >= 12) ? (std::stoi(argv[11]) != 0) : false;
    int max_num_considered_actions = (argc >= 13) ? std::stoi(argv[12]) : 16;
    size_t transposition_cache_entries = (argc >= 14) ? std::stoull(argv[13]) : 1000000;
    int chess_encoder_history = (argc >= 15) ? std::stoi(argv[14]) : 0;
    std::string value_network_path = (argc >= 16) ? argv[15] : "";

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

    ReplayBuffer replay_buffer(1000000, initial_game->getActionSize());

    std::cout << "Starting self play profiling with " << num_games << " games on " << thread_count
              << " threads... (max_moves=" << max_moves
              << ", mcts_num_simulations=" << mcts_num_simulations
              << ", mcts_batch_size=" << mcts_batch_size
              << ", fast_mcts_num_simulations=" << fast_mcts_num_simulations
              << ", full_search_probability=" << full_search_probability
              << ", use_gumbel_search=" << use_gumbel_search
              << ", max_num_considered_actions=" << max_num_considered_actions
              << ", transposition_cache_entries=" << transposition_cache_entries
              << ", chess_encoder_history=" << chess_encoder_history << ")" << '\n';

    std::shared_ptr<StateEncoder> encoder = nullptr;
    if (game_name == "chess" && chess_encoder_history > 0) {
        encoder = std::make_shared<ChessEncoderV2History>(chess_encoder_history);
    }

    self_play(initial_game, network_path, replay_buffer, num_games, thread_count,
              mcts_num_simulations, mcts_batch_size, max_moves, fast_mcts_num_simulations,
              full_search_probability, transposition_cache_entries, use_gumbel_search,
              max_num_considered_actions, false, -0.95f, 3, 60, 0.1f, 0.0f, encoder, encoder,
              value_network_path);

    if (use_kineto) {
        auto profiler_result = torch::autograd::profiler::disableProfiler();
        profiler_result->save(kineto_out_file);
        std::cout << "Kineto profile saved to " << kineto_out_file << '\n';
    }

    std::cout << "Profiling completed." << '\n';
    return 0;
}
