#include "../../engine/game/chess.hpp"
#include "../../engine/game/chess_encoder.hpp"
#include "../../engine/game/chess_encoder_v2history.hpp"
#include "../../engine/game/connect4.hpp"
#include "../../engine/game/state_encoder.hpp"
#include "../../engine/inference/basic_inferer.hpp"
#include "../../engine/mcts/mcts.hpp"
#include "../../engine/mcts/mcts_factory.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <memory>
#include <omp.h>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

// Plays head-to-head matches between two checkpoints of the same game and
// reports the score - the direct strength comparison (e.g. "is the current
// network actually stronger than the one from 6 hours ago?") that neither the
// puzzle evaluator (absolute, tiny sample) nor the training losses (in-
// distribution only) can answer. Mirrors run_self_play's structure: one MCTS
// pair (and its arenas) per OpenMP thread, games distributed dynamically,
// colors alternated per game index so neither engine gets a first-move-
// advantage edge over the match.
namespace {

struct MatchTally {
    std::atomic<int> a_wins{0};
    std::atomic<int> b_wins{0};
    std::atomic<int> draws{0};
    std::atomic<int> a_wins_as_white{0};
    std::atomic<int> b_wins_as_white{0};
};

// Picks engine A's move for the position in `game`. Gumbel mode plays the
// sequential-halving winner (the action carrying the paper's policy-
// improvement guarantee); PUCT mode plays argmax of the visit distribution.
// Both are the "strongest play" selection rules - no opening temperature
// sampling here, match diversity comes from Gumbel's own noise draws (or, in
// PUCT mode, from nothing, which is why gumbel is the default for arenas).
int pick_move(MCTS &mcts, const Game &game, int num_simulations, int batch_size,
              bool use_gumbel_search, int max_num_considered_actions) {
    if (use_gumbel_search) {
        auto result =
            mcts.search_gumbel(game, num_simulations, batch_size, max_num_considered_actions);
        return result.chosen_action;
    }
    auto [pi, value] = mcts.search(game, num_simulations, batch_size);
    auto best = std::max_element(pi.begin(), pi.end());
    return static_cast<int>(std::distance(pi.begin(), best));
}

} // namespace

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <game> <network_path_a> <network_path_b> [num_games] [thread_count] "
                     "[max_moves] [mcts_num_simulations] [mcts_batch_size] "
                     "[use_gumbel_search] [max_num_considered_actions] "
                     "[transposition_cache_entries] [encoder_history_a] [encoder_history_b]\n"
                  << "  encoder_history_{a,b}: chess only - 0 (default) = 19-plane "
                     "ChessEncoderV1; 1/4/8 = ChessEncoderV2History(N). Set per network to "
                     "match what each checkpoint was trained on (e.g. a history-encoder "
                     "trainee vs the legacy 1432: encoder_history_a=4 encoder_history_b=0).\n"
                  << "  Plays network A vs network B, alternating colors each game, and prints "
                     "the match score plus an Elo-difference estimate (positive = A stronger).\n";
        return 1;
    }

    std::string game_name = argv[1];
    std::string network_path_a = argv[2];
    std::string network_path_b = argv[3];
    int num_games = (argc >= 5) ? std::stoi(argv[4]) : 40;
    int thread_count = (argc >= 6) ? std::stoi(argv[5]) : 8;
    int max_moves = (argc >= 7) ? std::stoi(argv[6]) : 512;
    int mcts_num_simulations = (argc >= 8) ? std::stoi(argv[7]) : 800;
    int mcts_batch_size = (argc >= 9) ? std::stoi(argv[8]) : 64;
    bool use_gumbel_search = (argc >= 10) ? (std::stoi(argv[9]) != 0) : true;
    int max_num_considered_actions = (argc >= 11) ? std::stoi(argv[10]) : 16;
    size_t transposition_cache_entries = (argc >= 12) ? std::stoull(argv[11]) : 1000000;
    int encoder_history_a = (argc >= 13) ? std::stoi(argv[12]) : 0;
    int encoder_history_b = (argc >= 14) ? std::stoi(argv[13]) : 0;

    std::shared_ptr<Game> initial_game;
    if (game_name == "connect4") {
        initial_game = std::make_shared<Connect4>();
    } else if (game_name == "chess") {
        initial_game = std::make_shared<Chess>();
    } else {
        std::cerr << "Unknown game: " << game_name << '\n';
        return 1;
    }

    auto device = torch::Device(torch::cuda::is_available() ? "cuda" : "cpu");

    spdlog::info("Arena: {} games of {} on {} threads (sims={}, batch={}, gumbel={}, m={})",
                 num_games, game_name, thread_count, mcts_num_simulations, mcts_batch_size,
                 use_gumbel_search, max_num_considered_actions);
    spdlog::info("  A: {}", network_path_a);
    spdlog::info("  B: {}", network_path_b);

    // One shared inference factory (network + batcher + transposition cache)
    // per checkpoint, exactly like self_play() shares one across its threads -
    // each engine's evaluations batch across all games in flight. The caches
    // are per-factory, so results can never leak between the two networks.
    int wait_for_count = mcts_batch_size;
    int timeout_ms = 2;
    // Each network is fed the encoding it was trained on. 0 => leave null so the
    // factory derives the game default (ChessEncoderV1); N in {1,4,8} =>
    // ChessEncoderV2History(N). This lets a 63-plane history-encoder net play a
    // 19-plane legacy/v1 net in the same match without either being mis-fed.
    auto encoder_for = [](int history) -> std::shared_ptr<StateEncoder> {
        if (history == 0) {
            return nullptr;
        }
        return std::make_shared<ChessEncoderV2History>(history);
    };
    auto factory_a =
        NetworkInfererFactory(network_path_a, device, wait_for_count, timeout_ms,
                              transposition_cache_entries, encoder_for(encoder_history_a));
    auto factory_b =
        NetworkInfererFactory(network_path_b, device, wait_for_count, timeout_ms,
                              transposition_cache_entries, encoder_for(encoder_history_b));
    // eps=0: no Dirichlet root noise in PUCT mode - an arena wants each
    // engine's best play, not exploration. (search_gumbel ignores eps.)
    MCTSFactory mcts_factory_a(factory_a, 1.25F, 19652.0F, /*eps=*/0.0F);
    MCTSFactory mcts_factory_b(factory_b, 1.25F, 19652.0F, /*eps=*/0.0F);

    std::vector<std::unique_ptr<MCTS>> thread_mcts_a;
    std::vector<std::unique_ptr<MCTS>> thread_mcts_b;
    thread_mcts_a.reserve(thread_count);
    thread_mcts_b.reserve(thread_count);
    for (int t = 0; t < thread_count; t++) {
        thread_mcts_a.push_back(mcts_factory_a.get_mcts());
        thread_mcts_b.push_back(mcts_factory_b.get_mcts());
    }

    MatchTally tally;
    std::atomic<int> games_finished{0};

#pragma omp parallel for schedule(dynamic) num_threads(thread_count)
    for (int g = 0; g < num_games; g++) { // NOLINT
        auto &mcts_a = *thread_mcts_a[omp_get_thread_num()];
        auto &mcts_b = *thread_mcts_b[omp_get_thread_num()];
        bool a_plays_white = (g % 2 == 0);

        auto game = initial_game->clone();
        game->reset();
        int move_idx = 0;
        while (!game->is_terminal() && move_idx < max_moves) {
            bool white_to_move = game->get_current_player() == 0;
            auto &mover = (white_to_move == a_plays_white) ? mcts_a : mcts_b;
            int action = pick_move(mover, *game, mcts_num_simulations, mcts_batch_size,
                                   use_gumbel_search, max_num_considered_actions);
            game->step(action);
            move_idx++;
        }

        // reward() is expressed from the perspective of the side to move at
        // the terminal position (see Chess::reward()): -1 means that side has
        // been beaten, +1 (not produced by chess, but handled for generality)
        // would mean it somehow stands winning, 0 is a draw. A game cut off by
        // max_moves is scored as a draw.
        int winner = -1; // -1 = draw, 0 = white, 1 = black
        if (game->is_terminal()) {
            float r = game->reward();
            int to_move = game->get_current_player();
            if (r < 0.0F) {
                winner = 1 - to_move;
            } else if (r > 0.0F) {
                winner = to_move;
            }
        }

        if (winner == -1) {
            tally.draws++;
        } else {
            bool a_won = (winner == 0) == a_plays_white;
            if (a_won) {
                tally.a_wins++;
                if (winner == 0)
                    tally.a_wins_as_white++;
            } else {
                tally.b_wins++;
                if (winner == 0)
                    tally.b_wins_as_white++;
            }
        }

        auto done = ++games_finished;
        spdlog::info("Arena game {}/{} finished ({} plies): running score A {} - {} B ({} draws)",
                     done, num_games, move_idx, tally.a_wins.load(), tally.b_wins.load(),
                     tally.draws.load());
    }

    int a_wins = tally.a_wins.load();
    int b_wins = tally.b_wins.load();
    int draws = tally.draws.load();
    auto score_a = (a_wins + 0.5 * draws) / num_games;

    spdlog::info("=== ARENA RESULT ===");
    spdlog::info("A wins: {} ({} as white, {} as black)", a_wins, tally.a_wins_as_white.load(),
                 a_wins - tally.a_wins_as_white.load());
    spdlog::info("B wins: {} ({} as white, {} as black)", b_wins, tally.b_wins_as_white.load(),
                 b_wins - tally.b_wins_as_white.load());
    spdlog::info("Draws:  {}", draws);
    spdlog::info("Score for A: {:.1f}%", 100.0 * score_a);
    // The standard logistic-Elo estimate; undefined at 0%/100%, so clamp those
    // to just inside the open interval (they'd print as +/-inf otherwise, and
    // a shutout in a finite match only bounds the true difference anyway).
    auto clamped = std::clamp(score_a, 1e-3, 1.0 - 1e-3);
    auto elo_diff = -400.0 * std::log10(1.0 / clamped - 1.0);
    spdlog::info("Estimated Elo difference (A - B): {:+.0f}{}", elo_diff,
                 (score_a <= 1e-3 || score_a >= 1.0 - 1e-3) ? " (bound - shutout match)" : "");

    return 0;
}
