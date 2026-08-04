#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTest/MemoryLeakWarningPlugin.h"
#include "game/chess.hpp"
#include "game/chess_encoder.hpp"
#include <algorithm>
#include <spdlog/spdlog.h>
#include <utility>
#include <vector>

TEST_GROUP(ChessTests){void setup(){} void teardown(){}};

namespace {
// Plays (r1,c1)->(r2,c2) via the raw move_piece() mutator, but first asserts the move
// is actually legal per the engine's own move generator - move_piece() itself doesn't
// validate legality (it's a low-level board mutator used throughout these tests for
// hand-constructed positions), so for the long padding sequences in the draw-rule
// tests below, an unverified planning mistake could silently corrupt the board
// instead of failing loudly. This turns that into an immediate, clear test failure.
void play_verified(Chess &game, int r1, int c1, int r2, int c2) {
    int action = Chess::encode_action({static_cast<int8_t>(r1), static_cast<int8_t>(c1),
                                       static_cast<int8_t>(r2), static_cast<int8_t>(c2),
                                       /*promotion=*/0});
    auto legal = game.get_legal_actions();
    bool is_legal = std::find(legal.begin(), legal.end(), action) != legal.end();
    char msg[128];
    // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,cert-err33-c,cppcoreguidelines-pro-bounds-array-to-pointer-decay)
    snprintf(msg, sizeof(msg), "move (%d,%d)->(%d,%d) is not legal (player=%d)", r1, c1, r2, c2,
             game.get_current_player());
    CHECK_TRUE_TEXT(is_legal, msg);
    // NOLINTEND(cppcoreguidelines-pro-type-vararg,cert-err33-c,cppcoreguidelines-pro-bounds-array-to-pointer-decay)
    game.move_piece(r1, c1, r2, c2);
}
} // namespace

TEST(ChessTests, InitialMoves) {
    spdlog::info("Testing initial moves...");
    Chess game;
    auto actions = game.get_legal_actions();
    CHECK_EQUAL(20, actions.size());

    bool found_e4 = false;
    for (int act : actions) {
        ChessAction ca = Chess::decode_action(act);
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
    int castle_action = -1;
    for (int act : actions) {
        ChessAction ca = Chess::decode_action(act);
        if (ca.r1 == 7 && ca.c1 == 4 && ca.r2 == 7 && ca.c2 == 6) {
            castle_action = act;
            break;
        }
    }
    CHECK_TRUE(castle_action != -1);

    game.step(castle_action);
    auto after = game.get_board_state();
    CHECK_EQUAL(W_KING, after[7][6]);
    CHECK_EQUAL(EMPTY, after[7][4]);
    CHECK_EQUAL(W_ROOK, after[7][5]);
    CHECK_EQUAL(EMPTY, after[7][7]);
}

TEST(ChessTests, CastlingQueenside) {
    spdlog::info("Testing queenside castling...");
    Chess game;
    auto b = game.get_board_state();
    b[7][1] = EMPTY;
    b[7][2] = EMPTY;
    b[7][3] = EMPTY;
    game.set_custom_state(b, 0);

    auto actions = game.get_legal_actions();
    int castle_action = -1;
    for (int act : actions) {
        ChessAction ca = Chess::decode_action(act);
        if (ca.r1 == 7 && ca.c1 == 4 && ca.r2 == 7 && ca.c2 == 2) {
            castle_action = act;
            break;
        }
    }
    CHECK_TRUE(castle_action != -1);

    game.step(castle_action);
    auto after = game.get_board_state();
    CHECK_EQUAL(W_KING, after[7][2]);
    CHECK_EQUAL(EMPTY, after[7][4]);
    CHECK_EQUAL(W_ROOK, after[7][3]);
    CHECK_EQUAL(EMPTY, after[7][0]);
}

TEST(ChessTests, CastlingThroughAttackedSquareIsIllegalKingside) {
    spdlog::info("Testing that castling kingside through an attacked square is illegal...");
    Chess game;
    Chess::board_t b{};
    for (auto &row : b)
        row.fill(EMPTY);
    b[7][4] = W_KING; // e1 - not itself in check
    b[7][7] = W_ROOK; // h1
    b[0][7] = B_KING; // h8, kept out of the way
    // Black rook on f8 attacks straight down the empty f-file to f1 - the
    // square the king must pass through (e1->f1->g1) but never lands on or
    // starts from, isolating the "through" case from the already-covered
    // "into"/"out of" check cases.
    b[0][5] = B_ROOK;
    game.set_custom_state(b, 0);

    auto actions = game.get_legal_actions();
    int castle_action = -1;
    for (int act : actions) {
        ChessAction ca = Chess::decode_action(act);
        if (ca.r1 == 7 && ca.c1 == 4 && ca.r2 == 7 && ca.c2 == 6) {
            castle_action = act;
            break;
        }
    }
    CHECK_TRUE(castle_action == -1);
}

TEST(ChessTests, CastlingThroughAttackedSquareIsIllegalQueenside) {
    spdlog::info("Testing that castling queenside through an attacked square is illegal...");
    Chess game;
    Chess::board_t b{};
    for (auto &row : b)
        row.fill(EMPTY);
    b[7][4] = W_KING; // e1 - not itself in check
    b[7][0] = W_ROOK; // a1
    b[0][7] = B_KING; // h8, kept out of the way
    // Black rook on d8 attacks straight down the empty d-file to d1 - the
    // square the king must pass through (e1->d1->c1). b1 only needs to be
    // empty for the rook's path and isn't check-relevant, so it's left alone.
    b[0][3] = B_ROOK;
    game.set_custom_state(b, 0);

    auto actions = game.get_legal_actions();
    int castle_action = -1;
    for (int act : actions) {
        ChessAction ca = Chess::decode_action(act);
        if (ca.r1 == 7 && ca.c1 == 4 && ca.r2 == 7 && ca.c2 == 2) {
            castle_action = act;
            break;
        }
    }
    CHECK_TRUE(castle_action == -1);
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
        ChessAction ca = Chess::decode_action(act);
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
        ChessAction ca = Chess::decode_action(act);
        if (ca.r1 == 1 && ca.c1 == 0 && ca.r2 == 0 && ca.c2 == 0 && ca.promotion == 1) {
            found_promo = true;
            game.step(act);
            break;
        }
    }
    CHECK_TRUE(found_promo);
    CHECK_EQUAL(W_QUEEN, game.get_board_state()[0][0]);
}

// Regression test for a real cache-poisoning bug hit during chess training:
// write_canonical_state()'s en-passant plane used to mark the EP column
// whenever en_passant was set, even long after the capture right expired
// (en_passant/en_passant_move are never reset once set). Two positions
// identical except for EP *freshness* then produced the same canonical tensor
// while having different legal actions, breaking the inference transposition
// cache's core assumption (identical tensor => identical legal actions - see
// inference_cache.hpp) and letting cached results inject illegal EP captures
// into MCTS. The plane must be set exactly while the EP capture is timely -
// the same gate move_rules_P/p apply.
TEST(ChessTests, EnPassantPlaneOnlyMarkedWhileCaptureIsTimely) {
    spdlog::info("Testing en-passant plane timeliness...");
    Chess game;
    ChessEncoderV1 encoder;
    std::vector<float> tensor(19ULL * 8 * 8);

    // 1. d2-d4: EP on column 3 is live for black's reply.
    game.step(Chess::encode_action({6, 3, 4, 3, 0}));
    encoder.write_canonical_state(game, tensor.data());
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            CHECK_EQUAL(j == 3 ? 1.0f : 0.0f, tensor[18 * 64 + i * 8 + j]);

    // 1... a7-a6: black declined; the EP right expired, so the plane must be
    // all zeros again even though the en_passant member still holds column 3.
    game.step(Chess::encode_action({1, 0, 2, 0, 0}));
    encoder.write_canonical_state(game, tensor.data());
    for (int k = 0; k < 64; ++k)
        CHECK_EQUAL(0.0f, tensor[18 * 64 + k]);
}

// Regression test for a real "BAD ACTION INDEX" CUDA gather() crash hit during
// training. move_rules_P()/move_rules_p()'s diagonal-capture reads used to have
// no bound on the pawn's own rank: a pawn already sitting on its own promotion
// rank (row 0 for white, row 7 for black - it should always have promoted the
// move it got there, see move_piece()'s r2==0/r2==7 handling, so this can't
// happen through legal play) made those reads go one row past the board
// (board_state[-1] for white / board_state[8] for black), fabricating a
// phantom capture whose out-of-range destination is what corrupted an encoded
// action index (see the i==0/i==7 guards' comments in chess.cpp for the full
// mechanism). This constructs that unreachable-via-play state directly via
// set_custom_state() to exercise the guard; built with
// -DALPHAZERO_BUILD_ASAN_CHESS_TEST=ON, ASan turns the out-of-bounds read this
// guards against into a hard failure instead of silent UB.
TEST(ChessTests, PawnStuckOnPromotionRankDoesNotReadOutOfBounds) {
    spdlog::info("Testing pawn stuck on its own promotion rank...");
    Chess game;
    auto b = game.get_board_state();
    for (auto &row : b)
        row.fill(EMPTY);
    b[0][4] = B_KING;
    b[7][4] = W_KING;
    b[7][3] = B_PAWN; // illegal: black pawn already on its own promotion rank
    b[0][3] = W_PAWN; // illegal: white pawn already on its own promotion rank

    game.set_custom_state(b, 1); // black to move
    auto black_actions = game.get_legal_actions();
    for (int act : black_actions) {
        ChessAction ca = Chess::decode_action(act);
        CHECK_FALSE(ca.r1 == 7 && ca.c1 == 3); // stuck pawn must generate no moves
    }

    game.set_custom_state(b, 0); // white to move
    auto white_actions = game.get_legal_actions();
    for (int act : white_actions) {
        ChessAction ca = Chess::decode_action(act);
        CHECK_FALSE(ca.r1 == 0 && ca.c1 == 3); // stuck pawn must generate no moves
    }
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
    ChessEncoderV1 encoder;
    float buffer[19 * 64];
    encoder.write_canonical_state(game, buffer); // NOLINT

    for (int i = 0; i < 64; ++i) {
        CHECK_EQUAL(1.0f, buffer[(12 * 64) + i]); // NOLINT
    }
    CHECK_EQUAL(1.0f, buffer[0 * 64 + 6 * 8 + 0]);
    CHECK_EQUAL(1.0f, buffer[6 * 64 + 1 * 8 + 0]);

    auto board = game.get_board_state();
    game.set_custom_state(board, 1);
    encoder.write_canonical_state(game, buffer); // NOLINT

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
        ChessAction ca = Chess::decode_action(act);
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
        ChessAction ca = Chess::decode_action(act);
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

TEST(ChessTests, KnightCheckingKing) {
    spdlog::info("Testing available moves when king is checked by a knight...");
    Chess game;
    Chess::board_t b;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            b[i][j] = EMPTY;
        }
    }
    // White king at e1
    b[7][4] = W_KING;
    // Black knight at d3 checking the king
    b[5][3] = B_KNIGHT;

    // White rook at h1, could capture something if not in check
    b[7][7] = W_ROOK;
    // White pawn at c2, can capture the knight at d3
    b[6][2] = W_PAWN;

    game.set_custom_state(b, 0);
    auto actions = game.get_legal_actions();

    int valid_moves = 0;
    for (int act : actions) {
        ChessAction ca = Chess::decode_action(act);

        // Rook cannot move, because it doesn't resolve the knight check
        if (ca.r1 == 7 && ca.c1 == 7) {
            FAIL("Rook generated a move while in knight check!");
        }

        // Pawn capturing the knight
        if (ca.r1 == 6 && ca.c1 == 2) {
            CHECK_TRUE(ca.r2 == 5 && ca.c2 == 3);
            valid_moves++;
        }

        // King running away (d1, d2, e2, f1). f2 is attacked by the knight!
        if (ca.r1 == 7 && ca.c1 == 4) {
            valid_moves++;
        }
    }
    // Pawn capture (1) + King moves (4) = 5 moves
    CHECK_EQUAL(5, valid_moves);
}

TEST(ChessTests, BishopPinningPiece) {
    spdlog::info("Testing piece pinned by a bishop...");
    Chess game;
    Chess::board_t b;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            b[i][j] = EMPTY;
        }
    }
    // White king at h1
    b[7][7] = W_KING;
    // White rook at g2
    b[6][6] = W_ROOK;
    // Black bishop at a8 pinning the rook
    b[0][0] = B_BISHOP;

    game.set_custom_state(b, 0);
    auto actions = game.get_legal_actions();

    for (int act : actions) {
        ChessAction ca = Chess::decode_action(act);
        if (ca.r1 == 6 && ca.c1 == 6) {
            // Rook is pinned diagonally, but rooks can only move orthogonally.
            // Therefore, the pinned rook CANNOT move at all.
            FAIL("Orthogonal piece moved while diagonally pinned!");
        }
    }
}

TEST(ChessTests, FiftyMoveRuleTriggersAutomaticDrawAfterHundredPlies) {
    spdlog::info("Testing fifty-move rule...");
    Chess game;
    Chess::board_t b{};
    for (auto &row : b)
        row.fill(EMPTY);
    b[7][0] = W_KING;   // a1, never moves
    b[6][0] = W_BISHOP; // a2 - the piece that tours 50 distinct squares below
    b[3][3] = B_KING;   // d5, oscillates with row2/col3 (d6) below
    game.set_custom_state(b, 0);
    CHECK_FALSE(game.is_terminal());

    // 50 distinct destination squares for White's bishop, spanning rows 0-6 (row 7 is
    // White king's row). move_piece() doesn't validate move legality/reachability, so
    // these only need to avoid the White king's square, the bishop's own starting
    // square (landing back on it would recreate the very first position) and the two
    // squares Black's king oscillates between - not diagonal-move connectivity. Since
    // the bishop visits a fresh square every White move, the *composite* position
    // (bishop square + black king square + side to move) can never repeat, which
    // isolates this test from threefold repetition entirely.
    std::vector<std::pair<int, int>> bishop_squares;
    for (int r = 0; r <= 6 && bishop_squares.size() < 50; ++r) {
        for (int c = 0; c < 8 && bishop_squares.size() < 50; ++c) {
            bool is_start = (r == 6 && c == 0);
            bool is_black_osc = (r == 3 && c == 3) || (r == 2 && c == 3);
            if (is_start || is_black_osc)
                continue;
            bishop_squares.emplace_back(r, c);
        }
    }
    CHECK_EQUAL(50, static_cast<int>(bishop_squares.size()));

    int bishop_r = 6;
    int bishop_c = 0;
    int black_r = 3;
    for (int ply = 0; ply < 100; ++ply) {
        if (ply % 2 == 0) {
            auto [nr, nc] = bishop_squares[ply / 2]; // NOLINT
            game.move_piece(bishop_r, bishop_c, nr, nc);
            bishop_r = nr;
            bishop_c = nc;
        } else {
            int new_r = (black_r == 3) ? 2 : 3;
            game.move_piece(black_r, 3, new_r, 3);
            black_r = new_r;
        }
        if (ply < 99) {
            CHECK_FALSE(game.is_terminal());
        }
    }

    CHECK_TRUE(game.is_terminal());
    CHECK_EQUAL(0.0f, game.reward());
}

TEST(ChessTests, CaptureResetsFiftyMoveClock) {
    spdlog::info("Testing that a capture resets the fifty-move clock...");
    Chess game;
    Chess::board_t b{};
    for (auto &row : b)
        row.fill(EMPTY);
    b[7][0] = W_KING;
    b[6][0] = W_BISHOP;
    b[3][3] = B_KING;
    b[4][4] = B_BISHOP; // capturable by White's bishop from the right square
    game.set_custom_state(b, 0);

    // 90 non-progress plies (clock -> 90), well under the 100 threshold, using the
    // same distinct-squares-plus-oscillation trick as the test above so this doesn't
    // accidentally trip threefold repetition either.
    std::vector<std::pair<int, int>> bishop_squares;
    for (int r = 0; r <= 6 && bishop_squares.size() < 45; ++r) {
        for (int c = 0; c < 8 && bishop_squares.size() < 45; ++c) {
            bool is_start = (r == 6 && c == 0);
            bool is_black_osc = (r == 3 && c == 3) || (r == 2 && c == 3);
            bool is_black_bishop = (r == 4 && c == 4);
            if (is_start || is_black_osc || is_black_bishop)
                continue;
            bishop_squares.emplace_back(r, c);
        }
    }
    CHECK_EQUAL(45, static_cast<int>(bishop_squares.size()));

    int bishop_r = 6;
    int bishop_c = 0;
    int black_r = 3;
    for (int ply = 0; ply < 90; ++ply) {
        if (ply % 2 == 0) {
            auto [nr, nc] = bishop_squares[ply / 2]; // NOLINT
            game.move_piece(bishop_r, bishop_c, nr, nc);
            bishop_r = nr;
            bishop_c = nc;
        } else {
            int new_r = (black_r == 3) ? 2 : 3;
            game.move_piece(black_r, 3, new_r, 3);
            black_r = new_r;
        }
    }
    CHECK_FALSE(game.is_terminal());

    // Now White's bishop captures Black's spare bishop at (4,4) - this should reset
    // the clock to 0, not merely fail to increment it.
    game.move_piece(bishop_r, bishop_c, 4, 4);
    CHECK_EQUAL(W_BISHOP, game.get_board_state()[4][4]); // sanity: the capture landed
    CHECK_FALSE(game.is_terminal());

    // 99 more non-progress plies - if the capture above hadn't reset the clock, the
    // combined 90 (pre-capture) + 1 (capture, non-resetting under the bug) + 99 would
    // total 190, way past 100, and this would already be a draw. With the reset, the
    // clock is only at 99 here.
    bishop_r = 4;
    bishop_c = 4;
    std::vector<std::pair<int, int>> more_squares;
    for (int r = 0; r <= 6 && more_squares.size() < 50; ++r) {
        for (int c = 0; c < 8 && more_squares.size() < 50; ++c) {
            bool is_current = (r == 4 && c == 4);
            bool is_black_osc = (r == 3 && c == 3) || (r == 2 && c == 3);
            if (is_current || is_black_osc)
                continue;
            more_squares.emplace_back(r, c);
        }
    }
    CHECK_EQUAL(50, static_cast<int>(more_squares.size()));

    for (int ply = 0; ply < 99; ++ply) {
        if (ply % 2 == 0) {
            auto [nr, nc] = more_squares[ply / 2]; // NOLINT
            game.move_piece(bishop_r, bishop_c, nr, nc);
            bishop_r = nr;
            bishop_c = nc;
        } else {
            int new_r = (black_r == 3) ? 2 : 3;
            game.move_piece(black_r, 3, new_r, 3);
            black_r = new_r;
        }
    }
    // Clock is at 99 (the capture reset it 99 plies ago), not yet a draw.
    CHECK_FALSE(game.is_terminal());
}

TEST(ChessTests, PawnMoveResetsFiftyMoveClock) {
    spdlog::info("Testing that a pawn move resets the fifty-move clock...");
    Chess game;
    Chess::board_t b{};
    for (auto &row : b)
        row.fill(EMPTY);
    b[7][0] = W_KING;
    b[6][0] = W_BISHOP;
    b[3][3] = B_KING;
    b[5][5] = W_PAWN; // spare pawn, moves exactly once to reset the clock
    game.set_custom_state(b, 0);

    std::vector<std::pair<int, int>> bishop_squares;
    for (int r = 0; r <= 6 && bishop_squares.size() < 30; ++r) {
        for (int c = 0; c < 8 && bishop_squares.size() < 30; ++c) {
            bool is_start = (r == 6 && c == 0);
            bool is_black_osc = (r == 3 && c == 3) || (r == 2 && c == 3);
            bool is_pawn = (r == 5 && c == 5) || (r == 4 && c == 5); // pawn's square
                                                                     // before/after
            if (is_start || is_black_osc || is_pawn)
                continue;
            bishop_squares.emplace_back(r, c);
        }
    }
    CHECK_EQUAL(30, static_cast<int>(bishop_squares.size()));

    int bishop_r = 6;
    int bishop_c = 0;
    int black_r = 3;
    for (int ply = 0; ply < 60; ++ply) {
        if (ply % 2 == 0) {
            auto [nr, nc] = bishop_squares[ply / 2]; // NOLINT
            game.move_piece(bishop_r, bishop_c, nr, nc);
            bishop_r = nr;
            bishop_c = nc;
        } else {
            int new_r = (black_r == 3) ? 2 : 3;
            game.move_piece(black_r, 3, new_r, 3);
            black_r = new_r;
        }
    }
    CHECK_FALSE(game.is_terminal());

    // White's spare pawn advances one square - a pawn move, not a capture - should
    // still reset the clock.
    game.move_piece(5, 5, 4, 5);
    CHECK_FALSE(game.is_terminal());

    // 99 more non-progress plies (clock -> 99, not yet 100).
    black_r = (black_r == 3) ? 2 : 3; // it's Black's move now (White just moved the
                                      // pawn); keep oscillating from here
    std::vector<std::pair<int, int>> more_squares;
    for (int r = 0; r <= 6 && more_squares.size() < 50; ++r) {
        for (int c = 0; c < 8 && more_squares.size() < 50; ++c) {
            bool is_current = (r == bishop_r && c == bishop_c);
            bool is_black_osc = (r == 3 && c == 3) || (r == 2 && c == 3);
            bool is_pawn = (r == 4 && c == 5);
            if (is_current || is_black_osc || is_pawn)
                continue;
            more_squares.emplace_back(r, c);
        }
    }
    CHECK_EQUAL(50, static_cast<int>(more_squares.size()));

    for (int ply = 0; ply < 99; ++ply) {
        if (ply % 2 == 0) {
            game.move_piece(black_r, 3, black_r == 3 ? 2 : 3, 3);
            black_r = (black_r == 3) ? 2 : 3;
        } else {
            auto [nr, nc] = more_squares[ply / 2]; // NOLINT
            game.move_piece(bishop_r, bishop_c, nr, nc);
            bishop_r = nr;
            bishop_c = nc;
        }
    }
    CHECK_FALSE(game.is_terminal());
}

TEST(ChessTests, ThreefoldRepetitionTriggersDrawOnThirdOccurrence) {
    spdlog::info("Testing threefold repetition...");
    Chess game;
    Chess::board_t b{};
    for (auto &row : b)
        row.fill(EMPTY);
    b[7][0] = W_KING; // a1
    b[0][7] = B_KING; // h8
    game.set_custom_state(b, 0);
    CHECK_FALSE(game.is_terminal());
    CHECK_FALSE(game.is_threefold_repetition());

    // White's king shuffles a1<->a2, Black's shuffles h8<->h7, forever. Once each
    // side has moved its king for the first time (plies 1 and 2 respectively), that
    // side's castling rights are permanently gone - see
    // LosingCastlingRightsMakesPositionDistinctForRepetition below - so from ply 2
    // onward, every one of the 4 board+side-to-move combinations in this cycle
    // recurs with period 4: (a2,h7,White-to-move) first appears at ply 2, then again
    // at ply 6, then ply 10; (a1,h7,Black-to-move) first appears at ply 3, then ply
    // 7, then ply 11; and so on. The *earliest* of these to reach its third
    // occurrence is (a2,h7,White-to-move), at ply 10 - two plies into what would look
    // like a third "round trip", not at a round boundary. That's why this checks
    // is_terminal() after every single ply instead of only at round boundaries: the
    // rule fires mid-cycle, not when the board first looks like it's "back home".
    bool white_at_a1 = true;
    bool black_at_h8 = true;
    int white_row = 7;
    int black_row = 0;
    for (int ply = 1; ply <= 12; ++ply) {
        if (ply % 2 == 1) {
            int new_row = white_at_a1 ? 6 : 7;
            play_verified(game, white_row, 0, new_row, 0);
            white_row = new_row;
            white_at_a1 = !white_at_a1;
        } else {
            int new_row = black_at_h8 ? 1 : 0;
            play_verified(game, black_row, 7, new_row, 7);
            black_row = new_row;
            black_at_h8 = !black_at_h8;
        }
        if (ply < 10) {
            CHECK_FALSE(game.is_terminal());
            CHECK_FALSE(game.is_threefold_repetition());
        }
    }

    CHECK_TRUE(game.is_terminal());
    CHECK_TRUE(game.is_threefold_repetition());
    CHECK_EQUAL(0.0f, game.reward());
}

// A minimal, standalone regression test for a fact that broke several early drafts of
// the tests below while writing this feature: castling rights are lost the moment a
// king (or the relevant rook) first moves, *permanently* - moving it back to its
// original square does not restore them. It's easy to assume "the board looks
// identical to before, so it must BE the position from before", but it isn't:
// compute_position_hash() folds castling rights into a position's identity, so a
// king round trip changes the position even though the board and side to move don't.
TEST(ChessTests, CastlingRightsPermanentlyLostAfterKingMovesEvenIfItReturnsHome) {
    spdlog::info("Testing that castling-rights loss is permanent, even after the king "
                 "returns to its original square...");
    Chess game;
    Chess::board_t b{};
    for (auto &row : b)
        row.fill(EMPTY);
    b[7][0] = W_KING; // a1
    b[0][7] = B_KING; // h8
    game.set_custom_state(b, 0);
    CHECK_FALSE(game.is_terminal());

    auto shuffle_round = [&]() {
        play_verified(game, 7, 0, 6, 0); // Ka1-a2
        play_verified(game, 0, 7, 1, 7); // Kh8-h7
        play_verified(game, 6, 0, 7, 0); // Ka2-a1
        play_verified(game, 1, 7, 0, 7); // Kh7-h8
    };

    // Naive (wrong) expectation: ply 0 (the initial setup) is the position's first
    // occurrence; ply 4, with the board back to looking exactly like ply 0, is the
    // second; ply 8 would then be the third, and the draw would fire there.
    //
    // What actually happens: White's very first king move (ply 1) permanently drops
    // White's castling rights, and Black's first king move (ply 2) does the same for
    // Black - both round trips before ply 4 change the position's *identity* even
    // though the board returns to the same layout. So ply 0 (rights intact) can never
    // recur, and ply 4 is really the *first* occurrence of a different, "rights
    // gone" position - meaning ply 8 (its second occurrence) must NOT be terminal.
    // Only a third full round trip - ply 12 - reaches that variant's third
    // occurrence.
    shuffle_round(); // ply 4
    CHECK_FALSE(game.is_terminal());

    shuffle_round(); // ply 8 - the discriminating check: if castling rights weren't
                     // part of a position's identity, this would already be
                     // terminal (ply0 + ply4 + ply8 = 3 occurrences of the same,
                     // wrongly-merged position).
    CHECK_FALSE(game.is_terminal());
    CHECK_FALSE(game.is_threefold_repetition());

    shuffle_round(); // ply 12 - genuinely the third occurrence of the rights-gone
                     // position (ply4 = 1st, ply8 = 2nd, ply12 = 3rd).
    CHECK_TRUE(game.is_terminal());
    CHECK_TRUE(game.is_threefold_repetition());
}

TEST(ChessTests, LosingCastlingRightsMakesPositionDistinctForRepetition) {
    spdlog::info("Testing that castling rights are part of a position's identity...");
    Chess game;
    Chess::board_t b{};
    for (auto &row : b)
        row.fill(EMPTY);
    b[7][4] = W_KING; // e1, hasn't moved yet - White still has both castling rights
                      // (no rooks on the board, so - per compute_position_hash() -
                      // both of White's rights hinge purely on K_move_count==0)
    b[0][7] = B_KING; // h8, oscillates with h7
    game.set_custom_state(b, 0);
    CHECK_FALSE(game.is_terminal());

    // Same shuffle (Ke1<->f1, Kh8<->h7) and same reasoning as
    // ThreefoldRepetitionTriggersDrawOnThirdOccurrence above: after ply 1, White's
    // rights are gone for good, and the four board+side-to-move combinations in this
    // cycle recur with period 4 starting from ply 2, with (f1,h7,White-to-move) being
    // the first to reach a third occurrence, at ply 10.
    //
    // The discriminating check is *why* it isn't ply 8: the board at ply 8 looks
    // identical to the ply-0 starting position (White king back on e1, Black king
    // back on h8, White to move) - if castling rights were *not* part of a position's
    // identity, ply 0 would count as that same position's first occurrence, ply 4 its
    // second, and ply 8 would already be a third occurrence, well before ply 10.
    // Observing that it takes until ply 10 - not ply 8 - is itself the proof that the
    // rights-intact ply-0 position and the rights-lost ply-4/ply-8 positions are
    // being kept distinct.
    bool white_at_e1 = true;
    bool black_at_h8 = true;
    int white_col = 4;
    int black_row = 0;
    for (int ply = 1; ply <= 10; ++ply) {
        if (ply % 2 == 1) {
            int new_col = white_at_e1 ? 5 : 4;
            play_verified(game, 7, white_col, 7, new_col);
            white_col = new_col;
            white_at_e1 = !white_at_e1;
        } else {
            int new_row = black_at_h8 ? 1 : 0;
            play_verified(game, black_row, 7, new_row, 7);
            black_row = new_row;
            black_at_h8 = !black_at_h8;
        }
        if (ply < 10) {
            CHECK_FALSE(game.is_terminal());
            CHECK_FALSE(game.is_threefold_repetition());
        }
    }

    CHECK_TRUE(game.is_terminal());
    CHECK_TRUE(game.is_threefold_repetition());
}

TEST(ChessTests, RepetitionHistoryResetsOnNewGame) {
    spdlog::info("Testing that repetition history doesn't leak across games...");
    Chess game;
    Chess::board_t b{};
    for (auto &row : b)
        row.fill(EMPTY);
    b[7][0] = W_KING;
    b[0][7] = B_KING;
    game.set_custom_state(b, 0);

    auto shuffle_round = [&]() {
        play_verified(game, 7, 0, 6, 0);
        play_verified(game, 0, 7, 1, 7);
        play_verified(game, 6, 0, 7, 0);
        play_verified(game, 1, 7, 0, 7);
    };
    shuffle_round(); // 2nd occurrence of the starting position
    CHECK_FALSE(game.is_terminal());

    // Re-establishing the *same* board via set_custom_state should start a fresh
    // history, not treat this as the position's 3rd occurrence.
    game.set_custom_state(b, 0);
    CHECK_FALSE(game.is_terminal());
    CHECK_FALSE(game.is_threefold_repetition());

    // Same check for reset().
    game.reset();
    game.set_custom_state(b, 0);
    shuffle_round();
    CHECK_FALSE(game.is_terminal());
    game.set_custom_state(b, 0);
    CHECK_FALSE(game.is_threefold_repetition());
}

TEST(ChessTests, ClonePreservesRepetitionHistoryIndependently) {
    spdlog::info("Testing that clone() carries repetition history forward for MCTS...");
    Chess game;
    Chess::board_t b{};
    for (auto &row : b)
        row.fill(EMPTY);
    b[7][0] = W_KING;
    b[0][7] = B_KING;
    game.set_custom_state(b, 0);

    auto move = [&](Game &g, int r1, int c1, int r2, int c2) {
        int action = Chess::encode_action({static_cast<int8_t>(r1), static_cast<int8_t>(c1),
                                           static_cast<int8_t>(r2), static_cast<int8_t>(c2), 0});
        g.step(action);
    };

    // Same cycle and periodicity as ThreefoldRepetitionTriggersDrawOnThirdOccurrence
    // above: White king a1<->a2, Black king h8<->h7, King a2/h7/White-to-move first
    // recurring at ply 2, then ply 6, then ply 10 - the earliest of the cycle's four
    // sub-positions to reach a third occurrence.
    move(game, 7, 0, 6, 0); // ply1: Ka1-a2
    move(game, 0, 7, 1, 7); // ply2: Kh8-h7
    move(game, 6, 0, 7, 0); // ply3: Ka2-a1
    move(game, 1, 7, 0, 7); // ply4: Kh7-h8
    CHECK_FALSE(game.is_terminal());

    auto clone = game.clone();
    CHECK_FALSE(clone->is_terminal());

    // Push the clone 6 more plies (to ply 10 total) - this must only affect the
    // clone, and the clone must correctly recognize the resulting position as
    // terminal (proving position_history and repetition_count survive clone(), which
    // matters because MCTS clones the game once per simulation and relies on that
    // history being carried forward correctly - see engine/mcts/mcts.cpp's search()).
    move(*clone, 7, 0, 6, 0); // ply5
    move(*clone, 0, 7, 1, 7); // ply6
    move(*clone, 6, 0, 7, 0); // ply7
    move(*clone, 1, 7, 0, 7); // ply8
    move(*clone, 7, 0, 6, 0); // ply9
    move(*clone, 0, 7, 1, 7); // ply10 - third occurrence of (a2,h7,White-to-move)
    CHECK_TRUE(clone->is_terminal());
    CHECK_EQUAL(0.0f, clone->reward());

    // The original, untouched since the clone was taken, must still be exactly where
    // it was left (ply 4, no repetitions yet) - clone() must be a deep, independent
    // copy, not sharing state with the original.
    CHECK_FALSE(game.is_terminal());
}

TEST(ChessTests, CheckmateTakesPrecedenceOverCoincidingFiftyMoveRule) {
    spdlog::info("Testing checkmate takes precedence when it coincides with the fifty-move "
                 "threshold...");
    // Same smothered-mate setup used in engine/tests/test_mcts.cpp's
    // ForcedMateGetsMostVisitsUnderUniformPrior: White King c1, Knight e5; Black King
    // h8, Rook g8, Pawns g7/h7, Rook a2, Queen e2. White's Ne5-f7 is checkmate.
    // Two spare bishops (White's tours 49 distinct squares, Black's oscillates
    // between two fixed squares) pad out exactly 99 non-progress plies first, so the
    // mating move itself becomes the ply that pushes the halfmove clock to exactly
    // 100 - this tests that reward() reports checkmate (-1), not a fifty-move-rule
    // draw (0), when both conditions are met by the same move. The padding moves use
    // the raw move_piece() mutator directly (like
    // FiftyMoveRuleTriggersAutomaticDrawAfterHundredPlies above) rather than
    // play_verified()'s legality-checked path: only the destination squares matter
    // for what this test is actually exercising (the halfmove clock and the
    // checkmate/draw precedence rule), not whether each individual padding hop is a
    // geometrically valid bishop move.
    Chess game;
    Chess::board_t b{};
    for (auto &row : b)
        row.fill(EMPTY);
    b[7][2] = W_KING;   // c1
    b[3][4] = W_KNIGHT; // e5 - delivers the mate
    b[0][7] = B_KING;   // h8
    b[0][6] = B_ROOK;   // g8
    b[1][6] = B_PAWN;   // g7
    b[1][7] = B_PAWN;   // h7
    b[6][0] = B_ROOK;   // a2
    b[6][4] = B_QUEEN;  // e2
    b[4][2] = W_BISHOP; // spare padding piece, starts at c4 - its own reach doesn't
                        // matter (padding uses the unvalidated move_piece() path), it
                        // just needs to visit distinct squares
    b[3][3] = B_KNIGHT; // spare padding piece, oscillates with (2,3). Unlike a
                        // bishop, a knight's short/sparse attack pattern makes it easy
                        // to verify it can never end up threatening or interposing on
                        // the mate square (1,5)/f7: from (3,3) the offset to (1,5) is
                        // (-2,+2), not a knight move, and after an even number (50) of
                        // oscillations it's back at (3,3), not (2,3) - which *would*
                        // knight-attack f7 - so this is the square it must end on.

    // Padding starts with Black to move: 99 total padding plies (50 Black + 49 White)
    // leaves it White's turn for move #100, the mate.
    game.set_custom_state(b, /*active_player=*/1);
    CHECK_FALSE(game.is_terminal());

    // Candidate squares for White's spare bishop's 49-move tour. Excludes: the 8
    // "real" mate-setup squares, Black's two oscillation squares, its own starting
    // square, and the only two squares from which a piece would check Black's king at
    // h8 ((1,5) and (2,6), a knight's-move away) - landing on either would force
    // Black to respond to check instead of freely oscillating. (The bishop's own
    // actual attack pattern doesn't matter here since move_piece() doesn't validate
    // check status either, but avoiding them keeps this robust regardless.)
    std::vector<std::pair<int, int>> used = {{7, 2}, {3, 4}, {0, 7}, {0, 6}, {1, 6}, {1, 7}, {6, 0},
                                             {6, 4}, {4, 2}, {3, 3}, {2, 3}, {1, 5}, {2, 6}};
    std::vector<std::pair<int, int>> tour;
    for (int r = 0; r < 8 && tour.size() < 49; ++r) {
        for (int c = 0; c < 8 && tour.size() < 49; ++c) {
            if (std::find(used.begin(), used.end(), std::make_pair(r, c)) != used.end())
                continue;
            tour.emplace_back(r, c);
        }
    }
    CHECK_EQUAL(49, static_cast<int>(tour.size()));

    int white_r = 4;
    int white_c = 2;
    int black_r = 3;
    for (int ply = 0; ply < 99; ++ply) {
        if (ply % 2 == 0) {
            // Black's turn (padding starts with Black to move).
            int new_r = (black_r == 3) ? 2 : 3;
            game.move_piece(black_r, 3, new_r, 3);
            black_r = new_r;
        } else {
            auto [nr, nc] = tour[ply / 2]; // NOLINT
            game.move_piece(white_r, white_c, nr, nc);
            white_r = nr;
            white_c = nc;
        }
    }
    // 99 non-progress plies played; clock is at 99, one below the automatic-draw
    // threshold, and it's now White's turn to deliver the mate.
    CHECK_FALSE(game.is_terminal());
    CHECK_EQUAL(0, game.get_current_player());

    // Ne5-f7# - the mating move. This is also the move that pushes the halfmove clock
    // to exactly 100.
    play_verified(game, 3, 4, 1, 5);

    CHECK_TRUE(game.is_terminal());
    CHECK_TRUE(game.is_fifty_move_draw()); // the clock genuinely did reach 100...
    CHECK_EQUAL(-1.0f, game.reward());     // ...but checkmate still wins.
}
