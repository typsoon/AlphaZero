#ifndef SELF_PLAY_HPP
#define SELF_PLAY_HPP

#include "game/game.hpp"
#include "game/state_encoder.hpp"

#include "replay_buffer.hpp"
#include <memory>
#include <string>
#include <thread>

// fast_mcts_num_simulations/full_search_probability implement playout cap
// randomization: with probability full_search_probability a move gets the full
// mcts_num_simulations search and is recorded as a training example; otherwise it
// gets the cheaper fast_mcts_num_simulations search purely to advance the game and
// is not recorded (its visit distribution isn't accurate enough to train on).
// use_gumbel_search selects MCTS::search_gumbel() (Gumbel-Top-k root selection +
// sequential halving) instead of the default MCTS::search() (Dirichlet-noised
// PUCT) for every move, full or fast search alike - see mcts.hpp for details.
// max_num_considered_actions (m, only used when use_gumbel_search is set) is the
// full search's Gumbel-Top-k candidate-set size: raise it (e.g. 32) while the
// network's policy prior is still too weak to be trusted to rank the top moves,
// and drop back toward the paper's 16 default once it isn't - sequential halving
// can never recover a move the initial Top-m cut excluded. Fast searches derive
// their own smaller m from it (min(m, 8)) since their ~100-simulation budget
// can't support wide candidate sets - see play_game() in self_play.cpp.
// transposition_cache_entries bounds the shared NN-inference transposition
// cache (keyed by canonical-state-tensor hash, scoped to this call's network -
// see engine/inference/inference_cache.hpp); 0 disables caching.
// encoder selects the input encoding used for TRAJECTORY RECORDING (the tensor
// stored in the replay buffer, i.e. what the TRAINEE network learns from); null
// (the default) derives it from game's type via default_encoder_for()
// (engine/game/encoder_factory.hpp) - e.g. ChessEncoderV1 for Chess. Pass a
// specific encoder (e.g. ChessEncoderV2History) to train a network against a
// non-default encoding; the trainee network's input_channels must match
// encoder->state_shape()[0].
// self_play_encoder selects the encoding fed to the SELF-PLAY / INFERENCE
// network at network_path (the net that actually chooses the moves). Null (the
// default) means "same as encoder" - correct for ordinary self-play, where the
// move-generating net and the trainee are the same network with the same
// encoding. It differs only for a frozen-generator bootstrap whose generator
// expects a DIFFERENT encoding than the trainee: e.g. distilling a fresh
// ChessEncoderV2History (63-plane) trainee from a legacy/v1 19-plane champion,
// where the generator must be fed its own 19-plane ChessEncoderV1 input while
// the trajectory is still recorded in the trainee's 63-plane encoding. The
// self-play network's input_channels must match self_play_encoder's shape.
// The resignation_* parameters end decided games early instead of grinding out
// the dead-lost tail: when resignation_enabled, a game is scored as a loss for
// the side to move once that side's search root value has read below
// resignation_threshold on resignation_consecutive_moves of its own successive
// turns, never before ply resignation_min_ply. A random
// resignation_disable_probability fraction of games ignores resignation
// entirely so false resignations stay observable (see ResignationTracker in
// engine/utils/resignation.hpp). Defaults follow AlphaGo Zero's setup;
// resignation_enabled defaults to false so existing callers keep their
// play-to-the-end behavior.
void self_play(std::shared_ptr<Game> game, std::string network_path, ReplayBuffer &replay_buf,
               int num_games = 100, int thread_count = std::thread::hardware_concurrency(),
               int mcts_num_simulations = 800, int mcts_batch_size = 32, int max_moves = 512,
               int fast_mcts_num_simulations = 100, float full_search_probability = 0.25f,
               size_t transposition_cache_entries = 1000000, bool use_gumbel_search = false,
               int max_num_considered_actions = 16, bool resignation_enabled = false,
               float resignation_threshold = -0.95f, int resignation_consecutive_moves = 3,
               int resignation_min_ply = 60, float resignation_disable_probability = 0.1f,
               float fpu_reduction = 0.0f, std::shared_ptr<StateEncoder> encoder = nullptr,
               std::shared_ptr<StateEncoder> self_play_encoder = nullptr,
               std::string value_network_path = "",
               std::shared_ptr<StateEncoder> value_network_encoder = nullptr);

// Assuming Game, MCTS, ReplayBuffer, InfererFactory, MCTSFactory are defined
// somewhere And you have torch or your own tensor type if needed

#endif // !SELF_PLAY_HPP
