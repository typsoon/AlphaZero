#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTest/MemoryLeakWarningPlugin.h"
#include "CppUTest/TestHarness.h"
#include "game/chess.hpp"
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

int main(int ac, char **av) {
    MemoryLeakWarningPlugin::turnOffNewDeleteOverloads();
    return CommandLineTestRunner::RunAllTests(ac, av);
}
