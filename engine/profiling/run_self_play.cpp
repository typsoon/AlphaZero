#include "../../engine/game/connect4.hpp"
#include "../../engine/utils/replay_buffer.hpp"
#include "../../training/self_play.hpp"
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <network_path> <num_games> <thread_count>\n";
        return 1;
    }

    std::string network_path = argv[1];
    int num_games = std::stoi(argv[2]);
    int thread_count = std::stoi(argv[3]);

    auto device = torch::Device(torch::cuda::is_available() ? "cuda" : "cpu");
    auto initial_game = std::make_shared<Connect4>(device);

    ReplayBuffer replay_buffer(1000000);

    std::cout << "Starting self play profiling with " << num_games << " games on " << thread_count
              << " threads..." << '\n';

    self_play(initial_game, network_path, replay_buffer, num_games, thread_count);

    std::cout << "Profiling completed." << '\n';
    return 0;
}
