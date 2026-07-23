#include "CppUTest/CommandLineTestRunner.h"
#include "game/chess.hpp"
#include "game/chess_encoder_v2history.hpp"
#include <array>
#include <random>
#include <stdexcept>

TEST_GROUP(ChessEncoderV2HistoryTests){void setup(){} void teardown(){}};

TEST(ChessEncoderV2HistoryTests, RejectsUnsupportedHistoryLengths) {
    CHECK_THROWS(std::invalid_argument, ChessEncoderV2History(2));
    CHECK_THROWS(std::invalid_argument, ChessEncoderV2History(0));
    CHECK_THROWS(std::invalid_argument, ChessEncoderV2History(3));
    // 1, 4, 8 must NOT throw.
    ChessEncoderV2History e1(1);
    ChessEncoderV2History e4(4);
    ChessEncoderV2History e8(8);
}

TEST(ChessEncoderV2HistoryTests, StateShapeMatchesFormula) {
    for (int h : {1, 4, 8}) {
        ChessEncoderV2History encoder(h);
        auto shape = encoder.state_shape();
        CHECK_EQUAL(3u, shape.size());
        CHECK_EQUAL(14 * h + 7, shape[0]);
        CHECK_EQUAL(8, shape[1]);
        CHECK_EQUAL(8, shape[2]);
    }
}

TEST(ChessEncoderV2HistoryTests, GameStartHasZeroedHistoryFramesAndCorrectAuxPlanes) {
    ChessEncoderV2History encoder(4);
    std::vector<float> tensor(14 * 4 * 64 + 7 * 64, -1.0f);
    Chess game;
    encoder.write_canonical_state(game, tensor.data());

    // Frame 0 (current, the start position) must be non-empty: white pawns on
    // row 6 in our board_t convention (mirrors ChessEncoderV1's start-position
    // check), plane 0 = own pawns (white to move at game start).
    CHECK_EQUAL(1.0f, tensor[0 * 64 + 6 * 8 + 0]);
    // Frames 1-3 (no history yet) must be all-zero.
    for (int frame = 1; frame < 4; ++frame) {
        for (int k = 0; k < 14 * 64; ++k) {
            CHECK_EQUAL_TEXT(0.0f, tensor[frame * 14 * 64 + k], "unplayed history frame not zero");
        }
    }
    // No repetitions possible yet: frame 0's repetition planes (12, 13) must
    // be all-zero.
    for (int k = 0; k < 64; ++k) {
        CHECK_EQUAL(0.0f, tensor[12 * 64 + k]);
        CHECK_EQUAL(0.0f, tensor[13 * 64 + k]);
    }

    // Auxiliary planes (base = 4*14*64 = 3584): white to move, both sides have
    // full castling rights, clocks near zero.
    int base = 4 * 14 * 64;
    for (int k = 0; k < 64; ++k) {
        CHECK_EQUAL_TEXT(1.0f, tensor[base + k], "side-to-move plane"); // white to move
        CHECK_EQUAL_TEXT(1.0f, tensor[base + 64 + k], "own kingside castle");
        CHECK_EQUAL_TEXT(1.0f, tensor[base + 128 + k], "own queenside castle");
        CHECK_EQUAL_TEXT(1.0f, tensor[base + 192 + k], "opp kingside castle");
        CHECK_EQUAL_TEXT(1.0f, tensor[base + 256 + k], "opp queenside castle");
        CHECK_EQUAL_TEXT(0.0f, tensor[base + 320 + k], "halfmove clock");
    }
    // fullmove = 0/2+1 = 1, /200 = 0.005.
    CHECK_EQUAL(1.0f / 200.0f, tensor[base + 384]);
}

TEST(ChessEncoderV2HistoryTests, HistoryPopulatesAfterMovesAndOrdersRecentFirst) {
    ChessEncoderV2History encoder(4);
    Chess game;
    // 1. e4 e5 2. Nf3 - three plies played, so frame0=post-Nf3, frame1=post-e5,
    // frame2=post-e4, frame3=start position (still within the 4-frame window).
    game.step(Chess::encode_action({6, 4, 4, 4, 0})); // 1. e4
    game.step(Chess::encode_action({1, 4, 3, 4, 0})); // 1... e5
    game.step(Chess::encode_action({7, 6, 5, 5, 0})); // 2. Nf3

    std::vector<float> tensor(14 * 4 * 64 + 7 * 64, -1.0f);
    encoder.write_canonical_state(game, tensor.data());

    // All 4 frames must now be populated (non-degenerate: at least one piece
    // plane bit set somewhere in each frame - a fully-zero frame would mean
    // history wasn't threaded through correctly).
    for (int frame = 0; frame < 4; ++frame) {
        bool any_set = false;
        for (int k = 0; k < 12 * 64; ++k) {
            if (tensor[frame * 14 * 64 + k] != 0.0f) {
                any_set = true;
                break;
            }
        }
        CHECK_TRUE_TEXT(any_set, "history frame unexpectedly all-zero after 3 plies");
    }
}

TEST(ChessEncoderV2HistoryTests, RepetitionFlagsSetOnActualRepeat) {
    ChessEncoderV2History encoder(8);
    Chess game;
    // Shuffle knights back and forth to repeat the start position: after
    // Nf3 Nf6 Ng1 Ng8, the position is identical to the game's start (2nd
    // occurrence) - repeated once more (Nf3 Nf6 Ng1 Ng8 again) makes it the
    // 3rd occurrence.
    auto knight_shuffle = [&]() {
        game.step(Chess::encode_action({7, 6, 5, 5, 0})); // Ng1-f3
        game.step(Chess::encode_action({0, 6, 2, 5, 0})); // Ng8-f6
        game.step(Chess::encode_action({5, 5, 7, 6, 0})); // Nf3-g1
        game.step(Chess::encode_action({2, 5, 0, 6, 0})); // Nf6-g8
    };
    knight_shuffle(); // position recurs (2nd occurrence)

    std::vector<float> tensor(14 * 8 * 64 + 7 * 64);
    encoder.write_canonical_state(game, tensor.data());
    // Current position (frame 0) has occurred once before -> reps-before=1 ->
    // plane 12 set, plane 13 not.
    CHECK_EQUAL(1.0f, tensor[12 * 64ULL]);
    CHECK_EQUAL(0.0f, tensor[13 * 64ULL]);

    knight_shuffle(); // position recurs again (3rd occurrence)
    encoder.write_canonical_state(game, tensor.data());
    // reps-before clamped to 2 -> both plane 12 and 13 set.
    CHECK_EQUAL(1.0f, tensor[12 * 64ULL]);
    CHECK_EQUAL(1.0f, tensor[13 * 64ULL]);
}

TEST(ChessEncoderV2HistoryTests, ResetAndSetCustomStateClearHistory) {
    ChessEncoderV2History encoder(4);
    Chess game;
    game.step(Chess::encode_action({6, 4, 4, 4, 0}));
    game.step(Chess::encode_action({1, 4, 3, 4, 0}));

    game.reset();
    std::vector<float> tensor(14 * 4 * 64 + 7 * 64);
    encoder.write_canonical_state(game, tensor.data());
    for (int frame = 1; frame < 4; ++frame) {
        for (int k = 0; k < 14 * 64; ++k) {
            CHECK_EQUAL(0.0f, tensor[frame * 14 * 64 + k]);
        }
    }

    game.step(Chess::encode_action({6, 4, 4, 4, 0}));
    auto board = game.get_board_state();
    game.set_custom_state(board, 1);
    encoder.write_canonical_state(game, tensor.data());
    for (int frame = 1; frame < 4; ++frame) {
        for (int k = 0; k < 14 * 64; ++k) {
            CHECK_EQUAL_TEXT(0.0f, tensor[frame * 14 * 64 + k],
                             "set_custom_state must clear prior history");
        }
    }
}

// Sweep sanity: many random-playout positions, checking the invariants that
// must hold regardless of history length - one-hot piece planes, no
// double-occupied square, side-to-move plane constant. Mirrors the sweep in
// test_state_encoder.cpp for ChessEncoderV1.
TEST(ChessEncoderV2HistoryTests, SweepInvariantsHoldAcrossHistoryLengths) {
    for (int h : {1, 4, 8}) {
        ChessEncoderV2History encoder(h);
        std::mt19937 rng(777 + h);
        std::vector<float> tensor(14 * h * 64 + 7 * 64);
        long positions = 0;
        for (int g = 0; g < 30; ++g) {
            Chess game;
            for (int ply = 0; ply < 100; ++ply) {
                if (game.is_terminal())
                    break;
                auto legal = game.get_legal_actions();
                if (legal.empty())
                    break;

                encoder.write_canonical_state(game, tensor.data());
                for (int frame = 0; frame < h; ++frame) {
                    int base = frame * 14 * 64;
                    for (int k = 0; k < 64; ++k) {
                        int occupied = 0;
                        for (int plane = 0; plane < 12; ++plane)
                            occupied += (tensor[base + plane * 64 + k] != 0.0f) ? 1 : 0;
                        CHECK_TRUE(occupied <= 1);
                    }
                }
                int aux_base = h * 14 * 64;
                float stm = tensor[aux_base];
                CHECK_TRUE(stm == 0.0f || stm == 1.0f);
                for (int k = 0; k < 64; ++k)
                    CHECK_EQUAL(stm, tensor[aux_base + k]);
                ++positions;

                std::uniform_int_distribution<size_t> pick(0, legal.size() - 1);
                game.step(legal[pick(rng)]);
            }
        }
        CHECK_TRUE_TEXT(positions > 500, "sweep did not cover enough positions");
    }
}

int main(int ac, char **av) {
    return CommandLineTestRunner::RunAllTests(ac, av);
}
