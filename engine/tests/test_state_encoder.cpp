#include "CppUTest/CommandLineTestRunner.h"
#include "game/chess.hpp"
#include "game/chess_encoder.hpp"
#include "game/connect4.hpp"
#include "game/connect4_encoder.hpp"
#include "game/encoder_factory.hpp"
#include <array>
#include <random>

TEST_GROUP(StateEncoderTests){void setup(){} void teardown(){}};

// ChessEncoderV1 was extracted verbatim from the (now-removed)
// Chess::write_canonical_state, and was proven byte-for-byte identical to it
// across ~45k random-playout positions before the legacy method was deleted
// (see plan history / commit message). That comparison can no longer run
// (the old method is gone by design - the whole point of the extraction), so
// this test instead sweeps many positions - reaching castling, en passant,
// promotions, and both sides to move - and checks the encoder's own
// invariants directly, standing in as the ongoing regression guard.
TEST(StateEncoderTests, ChessEncoderV1SweepInvariants) {
    ChessEncoderV1 encoder;
    std::mt19937 rng(12345);
    std::array<float, 19 * 64> tensor{};
    long positions = 0;
    for (int g = 0; g < 200; ++g) {
        Chess game;
        for (int ply = 0; ply < 240; ++ply) {
            if (game.is_terminal()) break;
            auto legal = game.get_legal_actions();
            if (legal.empty()) break;

            encoder.write_canonical_state(game, tensor.data());
            // Side-to-move plane (index 12) is constant 1 or 0 across the board.
            float stm = tensor[12 * 64];
            for (int k = 0; k < 64; ++k)
                CHECK_EQUAL(stm, tensor[12 * 64 + k]);
            CHECK_TRUE(stm == 0.0f || stm == 1.0f);
            // Piece planes (0-11) and castling/EP planes (14-18) are one-hot
            // per square: every value is exactly 0 or 1.
            for (int plane : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18}) {
                for (int k = 0; k < 64; ++k) {
                    float v = tensor[plane * 64 + k];
                    CHECK_TRUE(v == 0.0f || v == 1.0f);
                }
            }
            // A square holds at most one piece: the 12 piece planes never
            // overlap at the same square.
            for (int k = 0; k < 64; ++k) {
                int occupied = 0;
                for (int plane = 0; plane < 12; ++plane)
                    occupied += (tensor[plane * 64 + k] != 0.0f) ? 1 : 0;
                CHECK_TRUE(occupied <= 1);
            }
            ++positions;

            std::uniform_int_distribution<size_t> pick(0, legal.size() - 1);
            game.step(legal[pick(rng)]);
        }
    }
    CHECK_TRUE_TEXT(positions > 5000, "sweep did not cover enough positions");
}

TEST(StateEncoderTests, ChessStateShapeMatchesStateDim) {
    ChessEncoderV1 encoder;
    auto shape = encoder.state_shape();
    CHECK_EQUAL(3u, shape.size());
    CHECK_EQUAL(Chess::state_dim[0], shape[0]);
    CHECK_EQUAL(Chess::state_dim[1], shape[1]);
    CHECK_EQUAL(Chess::state_dim[2], shape[2]);
}

TEST(StateEncoderTests, Connect4StateShapeMatchesStateDim) {
    Connect4Encoder encoder;
    auto shape = encoder.state_shape();
    CHECK_EQUAL(3u, shape.size());
    CHECK_EQUAL(std::get<0>(Connect4::state_dim), shape[0]);
    CHECK_EQUAL(std::get<1>(Connect4::state_dim), shape[1]);
    CHECK_EQUAL(std::get<2>(Connect4::state_dim), shape[2]);
}

TEST(StateEncoderTests, DefaultEncoderDispatchesOnGameType) {
    Chess chess;
    auto chess_encoder = default_encoder_for(chess);
    CHECK_TRUE(dynamic_cast<ChessEncoderV1 *>(chess_encoder.get()) != nullptr);

    Connect4 connect4;
    auto connect4_encoder = default_encoder_for(connect4);
    CHECK_TRUE(dynamic_cast<Connect4Encoder *>(connect4_encoder.get()) != nullptr);
}

int main(int ac, char **av) { return CommandLineTestRunner::RunAllTests(ac, av); }
