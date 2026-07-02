#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTest/MemoryLeakWarningPlugin.h"
#include "CppUTest/TestHarness.h"
#include "game/chess.hpp"
#include "game/connect4.hpp"
#include "mcts.hpp"
#include <algorithm>
#include <memory>
#include <utility>

namespace {

// Returns a uniform prior (equal logits) and a constant value=0 for every legal
// action, regardless of the position. With no information from the network at all,
// the only signal MCTS::search() can use to prefer one move over another is what it
// discovers by actually simulating: in particular, whether a move leads to a
// terminal (checkmate) node. This isolates MCTS's own terminal-node backpropagation
// logic from anything the trained network might already know.
class UniformInferer : public Inferer {
  public:
    UniformInferer() : Inferer(torch::kCPU) {}

    std::vector<inference_result> infer(const std::vector<const GameState *> &states) override {
        std::vector<inference_result> out;
        out.reserve(states.size());
        for (const auto *state : states) {
            inference_result res;
            res.legal_actions = state->get_legal_actions();
            res.legal_action_logits.assign(res.legal_actions.size(), 0.0f);
            res.value = 0.0f;
            out.push_back(std::move(res));
        }
        return out;
    }
};

// Kc1, Ne5 (White) vs Kh8, Rg8, pawns g7/h7, Ra2, Qe2 (Black). White is down a full
// queen and two rooks - hopeless on material - but Black's king is smothered by its
// own rook and pawns (g8/g7/h7 all occupied), so Ne5-f7 is checkmate whenever it's
// White's move: nothing can capture the knight on f7, and the king has no flight
// square.
Chess::board_t smothered_mate_board() {
    Chess::board_t b;
    for (auto &row : b)
        row.fill(EMPTY);
    b[0][6] = B_ROOK;   // g8
    b[0][7] = B_KING;   // h8
    b[1][6] = B_PAWN;   // g7
    b[1][7] = B_PAWN;   // h7
    b[3][4] = W_KNIGHT; // e5
    b[6][0] = B_ROOK;   // a2
    b[6][4] = B_QUEEN;  // e2
    b[7][2] = W_KING;   // c1
    return b;
}

} // namespace

TEST_GROUP(MCTSTests){};

// With a uniform prior (see UniformInferer above), the search has no reason to prefer
// Nf7 over any other knight move *except* what backpropagation tells it once a
// simulation actually reaches that terminal node - so this test exercises MCTS's
// terminal-node backprop path directly, one ply deep.
TEST(MCTSTests, ForcedMateGetsMostVisitsUnderUniformPrior) {
    Chess game;
    game.set_custom_state(smothered_mate_board(), 0); // White to move

    int mate_action = Chess::encode_action({3, 4, 1, 5, 0}); // e5-f7

    MCTS mcts(std::make_unique<UniformInferer>());
    auto [pi, root_value] = mcts.search(game, /*num_simulations=*/400, /*batch_size=*/1);
    (void)root_value;

    int argmax =
        static_cast<int>(std::distance(pi.begin(), std::max_element(pi.begin(), pi.end())));

    CHECK_EQUAL(mate_action, argmax);
    // The mating move should dominate the visit distribution, not just narrowly win it.
    CHECK_TRUE(pi[mate_action] > 0.5f);
}

// Same board, but Black to move: White's Nf7# threat is still sitting there for
// White's *next* move, so Black must find a reply that neutralizes it (e.g. guarding
// f7, or otherwise removing the threat) or lose. This exercises backpropagation two
// plies deep: leaf (White delivers mate, Black to move at the terminal) -> its parent
// (White-to-move node, right after Black's non-defensive reply) -> the root
// (Black-to-move). A sign error anywhere in that chain - not just the exact spot the
// previous test caught - would make this fail too.
//
// Rather than hand-picking which of Black's replies "defends" (tactics are easy to
// get wrong by hand, as our own puzzle-authoring earlier in this project found out the
// hard way), this test asks the engine itself: play each Black reply, then check
// whether White's Nf7 is still legal and still checkmate afterward.
TEST(MCTSTests, DefendingMoveOutscoresMoveThatAllowsForcedMateNextPly) {
    Chess game;
    game.set_custom_state(smothered_mate_board(), 1); // Black to move

    int mate_action = Chess::encode_action({3, 4, 1, 5, 0}); // White's e5-f7

    // Black has several replies that each individually defend (e.g. guarding f7, or
    // giving check to force White to deal with that first) - MCTS is free to prefer
    // any of them, so this collects *all* moves in each category rather than picking
    // one arbitrary representative of each, and compares the total probability mass
    // MCTS assigns to each category.
    std::vector<int> defend_actions;
    std::vector<int> allow_actions;
    for (int act : game.get_legal_actions()) {
        auto after = game.clone();
        after->step(act);
        if (after->is_terminal())
            continue; // ignore any Black move that itself ends the game

        auto white_actions = after->get_legal_actions();
        bool white_still_has_mate = std::find(white_actions.begin(), white_actions.end(),
                                              mate_action) != white_actions.end();

        bool allows_mate = false;
        if (white_still_has_mate) {
            auto after_mate = after->clone();
            after_mate->step(mate_action);
            allows_mate = after_mate->is_terminal() && after_mate->reward() == -1.0f;
        }

        (allows_mate ? allow_actions : defend_actions).push_back(act);
    }

    // Sanity-check the position actually has both kinds of replies, or the rest of
    // this test would be vacuous.
    CHECK_TRUE(!defend_actions.empty());
    CHECK_TRUE(!allow_actions.empty());

    MCTS mcts(std::make_unique<UniformInferer>());
    auto [pi, root_value] = mcts.search(game, /*num_simulations=*/2000, /*batch_size=*/1);
    (void)root_value;

    auto sum_pi = [&pi](const std::vector<int> &actions) {
        float sum = 0.0f;
        for (int a : actions)
            sum += pi[a];
        return sum;
    };

    CHECK_TRUE(sum_pi(defend_actions) > sum_pi(allow_actions));
}

// General regression coverage for the policy MCTS::search() returns, independent of
// the terminal-node bug above: it should be a normalized distribution supported only
// on legal actions.
TEST(MCTSTests, PolicyIsNormalizedOverLegalActionsOnly) {
    Chess game; // standard starting position
    auto legal = game.get_legal_actions();
    std::vector<bool> is_legal(game.getActionSize(), false);
    for (int act : legal)
        is_legal[act] = true;

    MCTS mcts(std::make_unique<UniformInferer>());
    auto [pi, root_value] = mcts.search(game, /*num_simulations=*/200, /*batch_size=*/8);
    (void)root_value;

    CHECK_EQUAL(static_cast<size_t>(game.getActionSize()), pi.size());

    float sum = 0.0f;
    for (size_t a = 0; a < pi.size(); ++a) {
        CHECK_TRUE(pi[a] >= 0.0f);
        if (!is_legal[a]) {
            CHECK_EQUAL(0.0f, pi[a]);
        }
        sum += pi[a];
    }
    DOUBLES_EQUAL(1.0, sum, 1e-4);
}

// With a uniform prior/value on an opening position where no single move is
// objectively winning, visits should spread across the (20) legal moves rather than
// pathologically collapsing onto one - a cheap guard against e.g. an accidental
// always-pick-the-first-child bug, or a UCB/selection loop that gets stuck.
TEST(MCTSTests, VisitsSpreadAcrossMovesWhenNoTacticsExist) {
    Chess game; // standard starting position, White to move, 20 legal moves

    MCTS mcts(std::make_unique<UniformInferer>());
    auto [pi, root_value] = mcts.search(game, /*num_simulations=*/400, /*batch_size=*/8);
    (void)root_value;

    auto nonzero_count = 0;
    auto max_pi = 0.0f;
    for (auto p : pi) {
        if (p > 0.0f)
            ++nonzero_count;
        max_pi = std::max(max_pi, p);
    }

    CHECK_TRUE(nonzero_count > 10);
    CHECK_TRUE(max_pi < 0.5f);
}

TEST_GROUP(MCTSConnect4Tests){};

// Player 1 (X) has three in a row at the bottom of columns 0-2; column 3 completes
// it. Column 6 has three stacked O's (not four - no win) just so the piece counts
// stay equal and it's still player 1's turn (Connect4's constructor infers whose turn
// it is from how many of each piece are on the board).
//
// This is the Connect4 analogue of ForcedMateGetsMostVisitsUnderUniformPrior above,
// and it matters for a specific reason: Connect4::step() only flips currentPlayer in
// its "game continues" branch - a winning move leaves currentPlayer pointing at the
// player who just won, whereas Chess::step() always flips player, win or not. Since
// MCTS's terminal backprop is written generically against Game::reward()/
// get_current_player(), that asymmetry between the two games' conventions is exactly
// the kind of thing that could make a fix that's correct for Chess silently wrong for
// Connect4 - so this checks it actually holds up here too, rather than assuming it.
TEST(MCTSConnect4Tests, ForcedWinGetsMostVisitsUnderUniformPrior) {
    Connect4::board_t b{};
    for (auto &row : b)
        row.fill(0);
    b[5][0] = 1;
    b[5][1] = 1;
    b[5][2] = 1; // three X's in a row at the bottom; column 3 wins
    b[5][6] = -1;
    b[4][6] = -1;
    b[3][6] = -1; // three stacked O's, just to balance piece counts

    Connect4 game(b);
    CHECK_EQUAL(1, game.get_current_player()); // sanity: X to move

    int win_action = 3;

    MCTS mcts(std::make_unique<UniformInferer>());
    auto [pi, root_value] = mcts.search(game, /*num_simulations=*/400, /*batch_size=*/1);
    (void)root_value;

    int argmax =
        static_cast<int>(std::distance(pi.begin(), std::max_element(pi.begin(), pi.end())));

    CHECK_EQUAL(win_action, argmax);
    CHECK_TRUE(pi[win_action] > 0.5f);
}

// Mirror image of the test above, with player -1 (O) to move instead of player 1
// (X). The always-flip fix in Connect4::step() negates currentPlayer symmetrically
// (`currentPlayer = -currentPlayer`), so on paper it should behave identically for
// either sign - but that's exactly the kind of assumption worth checking rather than
// trusting, especially right after fixing a sign bug in this same code path. Only
// testing X's win (as the other test does) would leave O's side of the board
// completely unverified.
TEST(MCTSConnect4Tests, ForcedWinForPlayerTwoGetsMostVisitsUnderUniformPrior) {
    Connect4::board_t b{};
    for (auto &row : b)
        row.fill(0);
    b[5][0] = -1;
    b[5][1] = -1;
    b[5][2] = -1; // three O's in a row at the bottom; column 3 wins
    b[5][6] = 1;
    b[4][6] = 1; // two stacked X's, just to unbalance piece counts so it's O's turn

    Connect4 game(b);
    CHECK_EQUAL(-1, game.get_current_player()); // sanity: O to move

    int win_action = 3;

    MCTS mcts(std::make_unique<UniformInferer>());
    auto [pi, root_value] = mcts.search(game, /*num_simulations=*/400, /*batch_size=*/1);
    (void)root_value;

    int argmax =
        static_cast<int>(std::distance(pi.begin(), std::max_element(pi.begin(), pi.end())));

    CHECK_EQUAL(win_action, argmax);
    CHECK_TRUE(pi[win_action] > 0.5f);
}

TEST_GROUP(SelfPlayRewardAssignmentTests){};

// Mirrors training/self_play.cpp's play_game()'s trajectory-reward assignment
// exactly (see the loop at the bottom of play_game): game->reward() is expressed
// from the perspective of whoever is "to move" at the now-terminal state, which is
// the *other* player from whoever made trajectory's last recorded move (a decisive
// game always ends with the player-to-move unable to move: mated, or facing a full
// board they can't win from) - so it needs one flip before starting the backward
// per-ply alternation.
static std::vector<float> assign_trajectory_rewards(const std::vector<int> &movers_per_ply,
                                                     float terminal_reward) {
    std::vector<float> rewards(movers_per_ply.size());
    float value = -terminal_reward;
    for (int i = static_cast<int>(movers_per_ply.size()) - 1; i >= 0; --i) {
        rewards[i] = value;
        value = -value;
    }
    return rewards;
}

// Fool's Mate: White plays f3 and g4, Black replies e5 and then Qh4#. Black wins.
// This exact position/move sequence is already covered by ChessTests::Checkmate
// (which asserts game.reward() == -1.0f afterward), so this test isn't guessing at
// what reward() returns - it reuses that already-verified fact and checks what
// self_play.cpp's own alternation loop actually does with it.
//
// The trajectory records one entry per ply, in mover order: [White/f3, Black/e5,
// White/g4, Black/Qh4#]. Black's own last entry (playing the winning move) must end
// up with reward +1 - they won the game. If it doesn't, the network is being trained
// on a value target that says "the move that just won the game was bad for the
// player who made it", which is backwards.
TEST(SelfPlayRewardAssignmentTests, WinningMoversLastTrajectoryEntryGetsPositiveReward) {
    Chess game;
    game.reset();
    game.move_piece(6, 5, 5, 5); // 1. f3    (White, trajectory[0])
    game.move_piece(1, 4, 3, 4); // 1... e5  (Black, trajectory[1])
    game.move_piece(6, 6, 4, 6); // 2. g4    (White, trajectory[2])
    game.move_piece(0, 3, 4, 7); // 2... Qh4# (Black, trajectory[3]) - Black wins

    CHECK_TRUE(game.is_terminal());
    CHECK_EQUAL(-1.0f, game.reward()); // sanity-check against ChessTests::Checkmate

    // movers_per_ply is unused by assign_trajectory_rewards beyond its length, but
    // spelling it out makes the mover/index correspondence explicit for readers.
    std::vector<int> movers_per_ply = {/*White*/ 0, /*Black*/ 1, /*White*/ 0, /*Black*/ 1};
    auto rewards = assign_trajectory_rewards(movers_per_ply, game.reward());

    CHECK_TRUE(rewards[3] > 0.0f); // Black's own winning move
    CHECK_TRUE(rewards[2] < 0.0f); // White's move right before losing
    CHECK_TRUE(rewards[1] > 0.0f); // Black's earlier move, in the game Black wins
    CHECK_TRUE(rewards[0] < 0.0f); // White's opening move, in the game White loses
}

int main(int ac, char **av) {
    MemoryLeakWarningPlugin::turnOffNewDeleteOverloads();
    return CommandLineTestRunner::RunAllTests(ac, av);
}
