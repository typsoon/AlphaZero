#include "self_play.hpp"
#include "game/encoder_factory.hpp"
#include "game/game.hpp"

#include "game/game.hpp"
#include "inference/basic_inferer.hpp"
#include "inference/dual_inferer.hpp"
#include "mcts.hpp"
#include "mcts/mcts_factory.hpp"
#include "replay_buffer.hpp"
#include "resignation.hpp"
#include <algorithm>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <memory>
#include <omp.h>
#include <random>
#include <spdlog/spdlog.h>
#include <string>
#include <utility>
#include <vector>

// MCTS::search() returns a dense policy (one entry per possible encoded action,
// zero everywhere it didn't visit) - pulling out just the nonzero entries here
// is what lets ReplayBuffer store a handful of (index, value) pairs per
// position instead of the full dense vector. torch::tensor() copies the data
// in, so the returned tensors stay valid after `policy` (a local vector) goes
// out of scope.
static std::pair<torch::Tensor, torch::Tensor> sparsify_policy(const std::vector<float> &policy) {
    std::vector<int64_t> indices;
    std::vector<float> values;
    for (size_t a = 0; a < policy.size(); a++) {
        if (policy[a] != 0.0f) {
            indices.push_back(static_cast<int64_t>(a));
            values.push_back(policy[a]);
        }
    }
    return {torch::tensor(indices, torch::kInt64), torch::tensor(values, torch::kFloat32)};
}

// Playout cap randomization (PCR): most moves use a cheap fast_mcts_num_simulations
// search just to pick a move and advance the game, and only a full_search_probability
// fraction of moves pay for the full mcts_num_simulations search - whose visit
// distribution is the only one accurate enough to be worth recording as a training
// target. This cuts average per-move search cost a lot (e.g. ~4x at the 100/800,
// 0.25 defaults below) for a training set that's the same size as always, since
// full-search moves are still the ones kept.
// Returns true when the game ended by resignation (used by self_play() only to
// aggregate a diagnostic count - resigned games still produce ordinary
// trajectories).
static bool play_game(std::shared_ptr<Game> game, MCTS &mcts, ReplayBuffer &replay_buffer,
                      int mcts_num_simulations, int fast_mcts_num_simulations,
                      float full_search_probability, int mcts_batch_size, int max_moves,
                      bool use_gumbel_search, int max_num_considered_actions,
                      bool resignation_enabled, float resignation_threshold,
                      int resignation_consecutive_moves, int resignation_min_ply,
                      float resignation_disable_probability, const StateEncoder &encoder,
                      float temperature, int temperature_plies) {
    game->reset();
    std::vector<Transition> trajectory;
    // Ply index (0-based move number) each trajectory entry was recorded at -
    // needed below because fast-search moves are skipped, so consecutive
    // trajectory entries are no longer necessarily one ply apart.
    std::vector<int> recorded_plies;
    static thread_local std::mt19937 rng(std::random_device{}());
    std::bernoulli_distribution full_search_dist(full_search_probability);

    // Resignation cuts the dead-lost tail off decided games so the replay
    // buffer holds contested positions instead of long K+Q-vs-K grinds. A
    // resignation_disable_probability fraction of games keeps playing to the
    // natural end regardless (AlphaGo Zero's guard): those games are the only
    // ones that could reveal the network resigning positions it would in fact
    // have saved, so the false-resignation rate stays observable in the data.
    std::bernoulli_distribution resignation_disable_dist(resignation_disable_probability);
    bool resignation_active = resignation_enabled && !resignation_disable_dist(rng);
    ResignationTracker resignation_tracker{resignation_threshold, resignation_consecutive_moves,
                                           resignation_min_ply};
    bool resigned = false;

    // MCTS::search() only expands a node (making its children visible to
    // selection) at the end of a round, once evaluate_batch() runs - so a search
    // whose batch_size is >= its num_simulations collects everything into one
    // round and never expands past the root's immediate children, degenerating
    // into a single-ply breadth-only pass with no real tree depth. mcts_batch_size
    // is sized for the full search's much larger simulation count (e.g. 128 vs.
    // 800), so reusing it unmodified for the fast search would trigger exactly
    // that: fast_mcts_num_simulations defaults to 100, well under a typical 128
    // batch size. Capping the fast search's own batch size to a quarter of its
    // simulation budget guarantees it still gets a handful of sequential rounds
    // (so it explores more than one ply deep) regardless of how mcts_batch_size is
    // configured for the full search.
    int fast_mcts_batch_size =
        std::max(1, std::min(mcts_batch_size, fast_mcts_num_simulations / 4));

    int move_idx = 0;
    while (!game->is_terminal() && move_idx < max_moves) {
        bool full_search = full_search_dist(rng);
        int simulations = full_search ? mcts_num_simulations : fast_mcts_num_simulations;
        int batch_size = full_search ? mcts_batch_size : fast_mcts_batch_size;
        std::vector<float> policy;
        // The search's raw network value at the root (side-to-move
        // perspective), fed to the resignation tracker below - fast and full
        // searches alike, so a losing side's streak isn't reset by the cheap
        // moves in between full searches.
        float root_value = 0.0f;
        // Gumbel search returns the action to play explicitly: the
        // sequential-halving winner (argmax of g + logit + sigma(completedQ))
        // is what carries the paper's policy-improvement guarantee, and it is
        // *not* recoverable from the returned pi, which deliberately drops the
        // Gumbel noise term (see MCTS::search_gumbel in mcts.hpp).
        int gumbel_action = -1;
        if (use_gumbel_search) {
            // Fast searches get a smaller Gumbel candidate set (capped at 8)
            // than the full search's max_num_considered_actions: with
            // fast_mcts_num_simulations=100 and m=16 the phase-0 budget is ~1
            // visit per candidate (n/(ceil(log2(m))*m)), so the
            // sequential-halving cuts would be based on near-noise Q
            // estimates. The cap doubles the visits behind every cut at the
            // same simulation cost; the full search's 800-sim budget has no
            // such problem, so it uses the configured m unmodified.
            int max_considered =
                full_search ? max_num_considered_actions : std::min(max_num_considered_actions, 8);
            auto result = mcts.search_gumbel(*game, simulations, batch_size, max_considered);
            policy = std::move(result.pi);
            gumbel_action = result.chosen_action;
            root_value = result.root_value;
        } else {
            auto result = mcts.search(*game, simulations, batch_size);
            policy = std::move(result.first);
            root_value = result.second;
        }

        // Temperature scaling: tau=`temperature` for the first
        // `temperature_plies` moves - sampled from the search's own policy,
        // reweighted by policy[a]^(1/tau) when tau != 1 (AlphaZero's standard
        // softening formula; std::discrete_distribution normalizes the
        // weights itself, so the powered vector doesn't need to sum to 1).
        // tau<=0 falls through to the greedy branches below regardless of
        // ply, same as the tau->0 phase past temperature_plies. The sampling
        // phase applies to both search variants (for Gumbel it adds opening
        // diversity on top of the Gumbel draws' own randomness, same as it
        // does on top of Dirichlet noise for PUCT); the tau->0 phase plays
        // the search's own best action - argmax of visit counts for PUCT,
        // the sequential-halving winner for Gumbel.
        int action = -1;
        if (move_idx < temperature_plies && temperature > 0.0f) {
            if (temperature != 1.0f) {
                std::vector<float> scaled(policy.size());
                for (size_t a = 0; a < policy.size(); a++) {
                    scaled[a] = policy[a] > 0.0f ? std::pow(policy[a], 1.0f / temperature) : 0.0f;
                }
                std::discrete_distribution<int> dist(scaled.begin(), scaled.end());
                action = dist(rng);
            } else {
                std::discrete_distribution<int> dist(policy.begin(), policy.end());
                action = dist(rng);
            }
        } else if (use_gumbel_search) {
            action = gumbel_action;
        } else {
            action = std::distance(policy.begin(), std::max_element(policy.begin(), policy.end()));
        }

        if (full_search) {
            auto shape = encoder.state_shape();
            torch::Tensor game_state_tensor = torch::empty(shape, torch::kFloat32);
            encoder.write_canonical_state(*game, game_state_tensor.data_ptr<float>());
            game_state_tensor = game_state_tensor.unsqueeze(0);
            auto [policy_indices, policy_values] = sparsify_policy(policy);
            trajectory.emplace_back(game_state_tensor, policy_indices, policy_values, 0);
            recorded_plies.push_back(move_idx);
        }

        // Checked before stepping: on a resignation the resigner is the player
        // to move at the *current* ply, so breaking here leaves move_idx (and
        // the just-recorded position, if this was a full search) pointing at
        // the position the game was resigned in.
        if (resignation_active && resignation_tracker.observe(move_idx, root_value)) {
            resigned = true;
            break;
        }

        game->step(action);
        move_idx++;
    }

    // game->reward() is expressed from the perspective of whoever is "to move" at
    // the now-terminal state, i.e. ply `move_idx` (see Chess::reward()/
    // Connect4::reward()). A recorded state at ply `recorded_plies[i]` shares that
    // same "to move" player exactly when the two plies have the same parity (each
    // intervening move, recorded or not, flips whose turn it is), so the sign flip
    // is driven by the ply gap's parity rather than by counting trajectory entries -
    // that stays correct even though fast-search moves in between were never added
    // to trajectory.
    // A resigned game follows the same convention: the resigner is the player to
    // move at ply move_idx (the loop broke before stepping), and the game is
    // scored as a loss (-1) from their perspective.
    float terminal_value = resigned ? -1.0f : game->reward();
    int terminal_ply = move_idx;

    for (int i = static_cast<int>(trajectory.size()) - 1; i >= 0; --i) {
        bool same_player = (terminal_ply - recorded_plies[i]) % 2 == 0;
        trajectory[i].reward = same_player ? terminal_value : -terminal_value;
    }

    replay_buffer.add(trajectory);
    return resigned;
}

void self_play(std::shared_ptr<Game> initial_game, std::string network_path,
               ReplayBuffer &replay_buffer, int num_games, int thread_count,
               int mcts_num_simulations, int mcts_batch_size, int max_moves,
               int fast_mcts_num_simulations, float full_search_probability,
               size_t transposition_cache_entries, bool use_gumbel_search,
               int max_num_considered_actions, bool resignation_enabled,
               float resignation_threshold, int resignation_consecutive_moves,
               int resignation_min_ply, float resignation_disable_probability, float fpu_reduction,
               std::shared_ptr<StateEncoder> encoder,
               std::shared_ptr<StateEncoder> self_play_encoder, std::string value_network_path,
               std::shared_ptr<StateEncoder> value_network_encoder, float dirichlet_epsilon,
               float temperature, int temperature_plies) {
    auto device = torch::Device(torch::cuda::is_available() ? "cuda" : "cpu");
    // std::cerr << device << '\n';

    // CUDA's default sync policy (cudaDeviceScheduleAuto) has every thread
    // blocked in cudaStreamSynchronize() busy-spin on the CPU while it waits for
    // the GPU, rather than sleeping - profiling showed this costing 65-84% of
    // total CPU/CUDA-API time during self-play, since every one of thread_count
    // OS threads spins at 100% CPU each time its MCTS batch is in flight,
    // directly starving the other self-play threads that could otherwise be
    // doing tree-search work. cudaDeviceScheduleBlockingSync trades a small
    // amount of added wake-up latency per sync for actually yielding the core.
    // Must be set before this process's first real CUDA context-creating call
    // (NetworkInfererFactory's torch::jit::load() below is the first one on this
    // path) - once a context exists, the call below is a no-op and returns
    // cudaErrorSetOnActiveProcess, which happens whenever self_play() is invoked
    // from Python training (python/__main__.py's model.to(device) already
    // created the process's CUDA context earlier), so this only takes effect for
    // this binary's own standalone callers (e.g. the profiling harness).
    if (device.is_cuda()) {
        cudaError_t flag_status = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        if (flag_status != cudaSuccess) {
            spdlog::warn("cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync) failed ({}): a CUDA "
                         "context already existed for this process, so the default spin-wait sync "
                         "policy is still in effect.",
                         cudaGetErrorString(flag_status));
        }
    }

    // wait_for_count must exceed mcts_batch_size by a real margin: a single thread's
    // one round of a full search submits up to mcts_batch_size states in one
    // DynamicBatcher::submit() call (see MCTS::evaluate_batch()), so setting
    // wait_for_count == mcts_batch_size would let that one submission satisfy the
    // threshold by itself - the worker would fire before any other thread's
    // concurrent submissions get a chance to add to the same batch, defeating the
    // point of batching across threads in the first place. Empirically, tuning this
    // (and timeout_ms) against the PCR self-play workload didn't move GPU idle time
    // meaningfully either way (see training/self_play.cpp git history), so this
    // keeps the original 2-threads-worth-of-full-search-batches sizing.
    int wait_for_count = std::min(thread_count, 1) * mcts_batch_size;
    int timeout_ms = 2;
    // `encoder` is the TRAINEE encoding used for trajectory recording (what
    // lands in the replay buffer). Falls back to the game type's default (e.g.
    // ChessEncoderV1 for Chess) when the caller didn't pass one.
    if (!encoder) {
        encoder = default_encoder_for(*initial_game);
    }
    // The inference factory feeds the move-generating network at network_path,
    // which may expect a DIFFERENT encoding than the trainee - e.g. a frozen
    // 19-plane legacy generator distilling into a 63-plane history-encoder
    // trainee (see self_play.hpp). Use self_play_encoder for the factory when
    // given; otherwise the generator IS the trainee (ordinary self-play) and
    // one encoder serves both.
    auto factory_encoder = self_play_encoder ? self_play_encoder : encoder;
    auto inferer_factory = NetworkInfererFactory(network_path, device, wait_for_count, timeout_ms,
                                                 transposition_cache_entries, factory_encoder);

    std::optional<NetworkInfererFactory> value_inferer_factory;
    if (!value_network_path.empty()) {
        auto val_encoder = value_network_encoder ? value_network_encoder : factory_encoder;
        value_inferer_factory.emplace(value_network_path, device, wait_for_count, timeout_ms,
                                      transposition_cache_entries, val_encoder);
    }

    struct MaybeDualFactory : InfererFactory {
        NetworkInfererFactory &policy_fac;
        NetworkInfererFactory *value_fac;
        MaybeDualFactory(NetworkInfererFactory &p, NetworkInfererFactory *v)
            : policy_fac(p), value_fac(v) {}
        std::unique_ptr<Inferer> get_inferer() override {
            if (!value_fac)
                return policy_fac.get_inferer();
            return std::make_unique<DualNetworkInferer>(policy_fac.get_inferer(),
                                                        value_fac->get_inferer());
        }
    };
    MaybeDualFactory dual_factory(inferer_factory,
                                  value_inferer_factory ? &*value_inferer_factory : nullptr);

    // Calculate necessary arena size for the MCTS memory pool
    size_t arena_size_bytes = calculate_arena_size(initial_game->getActionSize(), mcts_num_simulations);

    // Pass the PUCT/arena defaults through explicitly so fpu_reduction and
    // dirichlet_epsilon reach the factory; 0.0 fpu_reduction reproduces the
    // original assume-draw FPU, 0.25 dirichlet_epsilon reproduces the original
    // hardcoded root-noise weight.
    MCTSFactory mcts_factory(dual_factory, 1.25f, 19652.0f, dirichlet_epsilon, 0.3f,
                             arena_size_bytes, fpu_reduction);

    std::atomic<int> games_finished{0};
    std::atomic<int> games_resigned{0};

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
        bool resigned =
            play_game(initial_game->clone(), *mcts, replay_buffer, mcts_num_simulations,
                      fast_mcts_num_simulations, full_search_probability, mcts_batch_size,
                      max_moves, use_gumbel_search, max_num_considered_actions, resignation_enabled,
                      resignation_threshold, resignation_consecutive_moves, resignation_min_ply,
                      resignation_disable_probability, *encoder, temperature, temperature_plies);
        if (resigned)
            games_resigned++;

        auto current_finished = ++games_finished;

        spdlog::info("Games played: {}/{}", current_finished, num_games);
    }

    if (resignation_enabled) {
        spdlog::info("Self-play resignations: {}/{} games", games_resigned.load(), num_games);
    }
}
