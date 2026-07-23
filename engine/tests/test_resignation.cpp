#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTest/TestHarness.h"
#include "resignation.hpp"

TEST_GROUP(ResignationTrackerTests){void setup(){} void teardown(){}};

// The core behavior: the same side reading below threshold on
// `consecutive_moves` of its own successive turns fires a resignation, even
// though the other side's (positive) readings arrive on the plies in between.
// This is exactly the alternating-sign pattern of a decided game - the loser
// reads ~-1 on its turns, the winner ~+1 on its - and is why the tracker keys
// its streaks by ply parity instead of counting raw plies.
TEST(ResignationTrackerTests, LosingSideResignsDespiteAlternatingWinnerReadings) {
    ResignationTracker tracker{-0.95f, 3, 60};
    // Plies 60/62: two losing reads for side 0; winner's +0.97 in between must
    // not reset side 0's streak.
    CHECK_FALSE(tracker.observe(60, -0.97f));
    CHECK_FALSE(tracker.observe(61, 0.97f));
    CHECK_FALSE(tracker.observe(62, -0.98f));
    CHECK_FALSE(tracker.observe(63, 0.98f));
    CHECK_TRUE(tracker.observe(64, -0.99f));
}

TEST(ResignationTrackerTests, SingleGoodReadingResetsTheStreak) {
    ResignationTracker tracker{-0.95f, 3, 0};
    CHECK_FALSE(tracker.observe(0, -0.99f));
    CHECK_FALSE(tracker.observe(2, -0.99f));
    // A read at (not below) the threshold does not qualify and resets.
    CHECK_FALSE(tracker.observe(4, -0.95f));
    CHECK_FALSE(tracker.observe(6, -0.99f));
    CHECK_FALSE(tracker.observe(8, -0.99f));
    CHECK_TRUE(tracker.observe(10, -0.99f));
}

// Reads before min_ply must not count toward the streak - openings are too
// noisy to trust - but must not poison it either: the streak may start
// building from min_ply onward.
TEST(ResignationTrackerTests, ReadingsBeforeMinPlyNeverCount) {
    ResignationTracker tracker{-0.95f, 2, 60};
    CHECK_FALSE(tracker.observe(56, -0.99f));
    CHECK_FALSE(tracker.observe(58, -0.99f));
    // Streak is still 0 at min_ply; two qualifying reads from here fire.
    CHECK_FALSE(tracker.observe(60, -0.99f));
    CHECK_TRUE(tracker.observe(62, -0.99f));
}

// The two sides' streaks are fully independent: one side's qualifying reads
// never advance (or reset) the other's.
TEST(ResignationTrackerTests, SidesTrackIndependentStreaks) {
    ResignationTracker tracker{-0.95f, 2, 0};
    CHECK_FALSE(tracker.observe(0, -0.99f)); // side 0, streak 1
    CHECK_FALSE(tracker.observe(1, -0.99f)); // side 1, streak 1
    CHECK_FALSE(tracker.observe(2, 0.5f));   // side 0, reset
    CHECK_TRUE(tracker.observe(3, -0.99f));  // side 1, streak 2 - fires
}

// consecutive_moves below 1 is clamped to 1 (a zero or negative setting must
// not mean "never resign" by making the >= comparison unreachable, nor fire
// spuriously before any qualifying read).
TEST(ResignationTrackerTests, ConsecutiveMovesIsClampedToAtLeastOne) {
    ResignationTracker tracker{-0.95f, 0, 0};
    CHECK_FALSE(tracker.observe(0, 0.0f));
    CHECK_TRUE(tracker.observe(2, -0.99f));
}

int main(int argc, char **argv) {
    return CommandLineTestRunner::RunAllTests(argc, argv);
}
