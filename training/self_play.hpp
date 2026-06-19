#ifndef SELF_PLAY_HPP
#define SELF_PLAY_HPP

#include "game/game.hpp"

#include "replay_buffer.hpp"
#include <memory>
#include <thread>

using std::string;

void self_play(std::shared_ptr<Game> game, string network_path, ReplayBuffer &replay_buf,
               int num_games = 100, int thread_count = std::thread::hardware_concurrency());

// Assuming Game, MCTS, ReplayBuffer, InfererFactory, MCTSFactory are defined
// somewhere And you have torch or your own tensor type if needed

#endif // !SELF_PLAY_HPP
