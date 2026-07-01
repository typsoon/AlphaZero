import { ChessBoard } from './chess.js';
import * as assert from 'assert';

function testStandardMove() {
  console.log('--- Test Standard Move ---');
  const board = new ChessBoard();

  // E2 to E4 (White Pawn)
  // E2 = file 4, rank 2 -> row 6, col 4
  // E4 = file 4, rank 4 -> row 4, col 4
  const from = 6 * 8 + 4;
  const to = 4 * 8 + 4;
  const action = (from * 64 + to) * 5 + 0;

  board.step(action);
  const state = board.get_board_state().board;

  assert.strictEqual(state[4]![4]!, 'P');
  assert.strictEqual(state[6]![4]!, ' ');
  assert.strictEqual(board.get_current_player(), 1); // Black to move
  console.log('Standard move passed.');
}

function runAllTests() {
  testStandardMove();
  console.log('All tests passed successfully.');
}

runAllTests();
