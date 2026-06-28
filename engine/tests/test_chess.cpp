#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTest/MemoryLeakWarningPlugin.h"
#include "game/chess.hpp"
#include <spdlog/spdlog.h>

TEST_GROUP(ChessTests){void setup(){} void teardown(){}};

TEST(ChessTests, InitialMoves) {
    spdlog::info("Testing initial moves...");
    Chess game;
    auto actions = game.get_legal_actions();
    CHECK_EQUAL(20, actions.size());

    bool found_e4 = false;
    for (int act : actions) {
        ChessAction ca = game.decode_action(act);
        if (ca.r1 == 6 && ca.c1 == 4 && ca.r2 == 4 && ca.c2 == 4) {
            game.step(act);
            found_e4 = true;
            break;
        }
    }
    CHECK_TRUE(found_e4);
    CHECK_EQUAL(W_PAWN, game.get_board_state()[4][4]);
    CHECK_EQUAL(EMPTY, game.get_board_state()[6][4]);
}

TEST(ChessTests, Castling) {
    spdlog::info("Testing castling...");
    Chess game;
    auto b = game.get_board_state();
    b[7][5] = EMPTY;
    b[7][6] = EMPTY;
    game.set_custom_state(b, 0);

    auto actions = game.get_legal_actions();
    bool found_castle = false;
    for (int act : actions) {
        ChessAction ca = game.decode_action(act);
        if (ca.r1 == 7 && ca.c1 == 4 && ca.r2 == 7 && ca.c2 == 6) {
            found_castle = true;
            break;
        }
    }
    CHECK_TRUE(found_castle);
}

TEST(ChessTests, EnPassant) {
    spdlog::info("Testing en passant...");
    Chess game;
    game.reset();
    auto b = game.get_board_state();
    b[6][4] = EMPTY;
    b[3][4] = W_PAWN;
    b[1][3] = B_PAWN;
    game.set_custom_state(b, 1);
    game.move_piece(1, 3, 3, 3);

    auto actions = game.get_legal_actions();
    bool found_ep = false;
    for (int act : actions) {
        ChessAction ca = game.decode_action(act);
        if (ca.r1 == 3 && ca.c1 == 4 && ca.r2 == 2 && ca.c2 == 3) {
            found_ep = true;
            game.step(act);
            break;
        }
    }
    CHECK_TRUE(found_ep);
    CHECK_EQUAL(EMPTY, game.get_board_state()[3][3]);
}

TEST(ChessTests, Promotion) {
    spdlog::info("Testing promotion...");
    Chess game;
    game.reset();
    auto b = game.get_board_state();
    b[1][0] = W_PAWN;
    b[0][0] = EMPTY;
    game.set_custom_state(b, 0);

    auto actions = game.get_legal_actions();
    bool found_promo = false;
    for (int act : actions) {
        ChessAction ca = game.decode_action(act);
        if (ca.r1 == 1 && ca.c1 == 0 && ca.r2 == 0 && ca.c2 == 0 && ca.promotion == 1) {
            found_promo = true;
            game.step(act);
            break;
        }
    }
    CHECK_TRUE(found_promo);
    CHECK_EQUAL(W_QUEEN, game.get_board_state()[0][0]);
}

TEST(ChessTests, Checkmate) {
    spdlog::info("Testing checkmate...");
    Chess game;
    game.reset();
    game.move_piece(6, 5, 5, 5); // 1. f3
    game.move_piece(1, 4, 3, 4); // 1... e5
    game.move_piece(6, 6, 4, 6); // 2. g4
    game.move_piece(0, 3, 4, 7); // 2... Qh4#

    CHECK_TRUE(game.is_terminal());
    CHECK_EQUAL(-1.0f, game.reward());
}

TEST(ChessTests, Stalemate) {
    spdlog::info("Testing stalemate...");
    Chess game;
    Chess::board_t b;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            b[i][j] = EMPTY; // NOLINT
        }
    }
    b[0][0] = B_KING;
    b[1][2] = W_QUEEN;
    b[2][2] = W_KING;
    game.set_custom_state(b, 1, -1, 1, 1, 1);

    CHECK_TRUE(game.is_terminal());
    CHECK_EQUAL(0.0f, game.reward());
}

TEST(ChessTests, CanonicalState) {
    spdlog::info("Testing canonical state...");
    Chess game;
    float buffer[19 * 64];
    game.write_canonical_state(buffer); // NOLINT

    for (int i = 0; i < 64; ++i) {
        CHECK_EQUAL(1.0f, buffer[(12 * 64) + i]); // NOLINT
    }
    CHECK_EQUAL(1.0f, buffer[0 * 64 + 6 * 8 + 0]);
    CHECK_EQUAL(1.0f, buffer[6 * 64 + 1 * 8 + 0]);

    auto board = game.get_board_state();
    game.set_custom_state(board, 1);
    game.write_canonical_state(buffer);

    for (int i = 0; i < 64; ++i) {
        CHECK_EQUAL(0.0f, buffer[12 * 64 + i]);
    }
    CHECK_EQUAL(1.0f, buffer[0 * 64 + 6 * 8 + 0]);
    CHECK_EQUAL(1.0f, buffer[6 * 64 + 1 * 8 + 0]);
}

TEST(ChessTests, KingInCheck) {
    spdlog::info("Testing available moves when king is in check...");
    Chess game;
    Chess::board_t b;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            b[i][j] = EMPTY;
        }
    }
    // White king at e1
    b[7][4] = W_KING;
    // Black rook at e8 checking the king
    b[0][4] = B_ROOK;

    // White pawn at d2
    b[6][3] = W_PAWN;
    // White knight at g1, can move to e2 to block
    b[7][6] = W_KNIGHT;

    game.set_custom_state(b, 0);

    auto actions = game.get_legal_actions();

    int valid_moves = 0;
    for (int act : actions) {
        ChessAction ca = game.decode_action(act);
        // No pawn moves should be valid because d2-d3 or d2-d4 does not block the e-file or capture
        // the rook.
        CHECK_TRUE(ca.r1 != 6 || ca.c1 != 3);

        if (ca.r1 == 7 && ca.c1 == 4) {
            // King can move to d1, f1, f2 (d2 is occupied by white pawn)
            // Wait, e2 is occupied by knight, so it cannot move there either.
            // Let's just ensure no knight moves are present.
            CHECK_TRUE((ca.r2 == 7 && ca.c2 == 3) || (ca.r2 == 7 && ca.c2 == 5) ||
                       (ca.r2 == 6 && ca.c2 == 5));
            valid_moves++;
        }
        if (ca.r1 == 7 && ca.c1 == 6) {
            // Knight moving to e2 to block
            CHECK_TRUE(ca.r2 == 6 && ca.c2 == 4);
            valid_moves++;
        }
    }
    CHECK_EQUAL(4, valid_moves);
    CHECK_EQUAL(4, actions.size());
}

TEST(ChessTests, KingNotInCheckButPiecePinned) {
    spdlog::info("Testing available moves when piece is pinned...");
    Chess game;
    Chess::board_t b;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            b[i][j] = EMPTY;
        }
    }
    // White king at e1
    b[7][4] = W_KING;
    // White knight at e2 (pinned)
    b[6][4] = W_KNIGHT;
    // Black rook at e8 (pinning the knight)
    b[0][4] = B_ROOK;

    // White pawn at d2 (not pinned)
    b[6][3] = W_PAWN;

    game.set_custom_state(b, 0);

    auto actions = game.get_legal_actions();

    for (int act : actions) {
        ChessAction ca = game.decode_action(act);
        // The knight is absolutely pinned by the rook. It cannot move at all.
        if (ca.r1 == 6 && ca.c1 == 4) {
            FAIL("Pinned knight generated a move!");
        }
    }
}

int main(int ac, char **av) {
    MemoryLeakWarningPlugin::turnOffNewDeleteOverloads();
    return CommandLineTestRunner::RunAllTests(ac, av);
}
