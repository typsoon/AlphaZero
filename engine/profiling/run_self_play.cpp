#include "../../engine/game/chess.hpp"
#include "../../engine/game/connect4.hpp"
#include "../../engine/utils/replay_buffer.hpp"
#include "../../training/self_play.hpp"
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <game> <network_path> <num_games> <thread_count> [max_moves]\n";
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

    std::shared_ptr<Game> initial_game;
    if (game_name == "connect4") {
        auto device = torch::Device(torch::cuda::is_available() ? "cuda" : "cpu");
        initial_game = std::make_shared<Connect4>(device);
    } else if (game_name == "chess") {
        initial_game = std::make_shared<Chess>();
    } else {
        std::cerr << "Unknown game: " << game_name << '\n';
        return 1;
    }

    ReplayBuffer replay_buffer(1000000);

    std::cout << "Starting self play profiling with " << num_games << " games on " << thread_count
              << " threads... (max_moves=" << max_moves << ")" << '\n';

    self_play(initial_game, network_path, replay_buffer, num_games, thread_count, 800, 32,
              max_moves);

    std::cout << "Profiling completed." << '\n';
    return 0;
}
