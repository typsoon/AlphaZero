#ifndef SELF_PLAY_HPP
#define SELF_PLAY_HPP

#include "game/game.hpp"

#include "replay_buffer.hpp"
#include <memory>
#include <string>
#include <thread>

void self_play(std::shared_ptr<Game> game, std::string network_path, ReplayBuffer &replay_buf,
               int num_games = 100, int thread_count = std::thread::hardware_concurrency(),
               int mcts_num_simulations = 800, int mcts_batch_size = 32);

// Assuming Game, MCTS, ReplayBuffer, InfererFactory, MCTSFactory are defined
// somewhere And you have torch or your own tensor type if needed

#endif // !SELF_PLAY_HPP
