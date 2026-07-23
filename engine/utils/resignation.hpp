#ifndef RESIGNATION_HPP
#define RESIGNATION_HPP

#include <algorithm>
#include <array>

// Decides when a self-play game should be resigned by the side to move, from
// the sequence of root value estimates the per-move searches produce.
//
// The search's root value is expressed from the perspective of the player to
// move, so it alternates sign every ply in a decided game (the loser reads
// ~-1, the winner ~+1). A single streak counter over raw plies would therefore
// be reset on every one of the winner's turns and could never reach its
// target - the streak has to be tracked per side, counting only that player's
// own consecutive turns. observe() keys the two counters by ply parity, which
// is exactly "whose turn it is" for the alternating two-player games this
// engine supports.
//
// A resignation fires only after min_ply (openings are too noisy to trust a
// single bad read) and only once the same side has read value < threshold on
// consecutive_moves of its successive turns (protection against one-off
// evaluation noise, following AlphaGo Zero's false-resignation guard).
struct ResignationTracker {
    // Defaults mirror self_play()'s (see training/self_play.hpp); every real
    // caller aggregate-initializes all three from its own configuration.
    float threshold{-0.95f};
    int consecutive_moves{3};
    int min_ply{60};

    // Records the search's root value (side-to-move perspective) observed at
    // 0-based ply `ply`; returns true when the side to move should resign the
    // game as lost.
    bool observe(int ply, float value) {
        auto side = static_cast<size_t>(ply % 2);
        if (ply >= min_ply && value < threshold)
            streaks[side]++;
        else
            streaks[side] = 0;
        return streaks[side] >= std::max(1, consecutive_moves);
    }

    // Per-side qualifying-read streaks, indexed by ply parity. Public only so
    // the struct stays an aggregate (brace-initializable from the three
    // configuration fields above); mutate it through observe().
    std::array<int, 2> streaks{0, 0};
};

#endif // !RESIGNATION_HPP
