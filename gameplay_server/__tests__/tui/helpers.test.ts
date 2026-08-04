import { Chess } from 'chess.js';
import { describe, it, expect } from '@jest/globals';
import {
  BIG_BOARD,
  COMPACT_BOARD,
  describeResult,
  pickBestLegalAction,
  renderBoard,
} from '../../src/tui/helpers.js';
import { ChessBoard } from '../../src/game/chess.js';

// Build the ANSI-escape matcher without a literal control character in the
// regex source (which eslint's no-control-regex rightly forbids).
const ANSI = new RegExp(`${String.fromCharCode(27)}\\[[0-9;]*m`, 'g');
const stripAnsi = (s: string) => s.replace(ANSI, '');

describe('pickBestLegalAction', () => {
  it('picks the highest-mass legal action from a sparse policy', () => {
    const policy = [
      { index: 10, value: 0.9 }, // illegal - must be ignored
      { index: 5, value: 0.3 },
      { index: 7, value: 0.5 },
    ];
    expect(pickBestLegalAction(policy, [5, 7])).toBe(7);
  });

  it('picks the highest-mass legal action from a dense policy', () => {
    const policy = new Array<number>(20).fill(0);
    policy[3] = 0.2;
    policy[12] = 0.8; // illegal - must be ignored
    policy[15] = 0.4;
    expect(pickBestLegalAction(policy, [3, 15])).toBe(15);
  });

  it('returns null when no legal action carries mass', () => {
    expect(pickBestLegalAction([{ index: 9, value: 1 }], [1, 2])).toBeNull();
    expect(pickBestLegalAction([], [])).toBeNull();
  });
});

describe('ChessBoard.decodeMove', () => {
  it('round-trips every legal starting move through encode/decode', () => {
    const board = new ChessBoard();
    const reference = new Chess();
    const legalSquares = new Set(
      reference.moves({ verbose: true }).map((m) => `${m.from}${m.to}`),
    );
    const decoded = board
      .get_legal_actions()
      .map((a) => board.decodeMove(a))
      .map((m) => `${m.from}${m.to}`);
    expect(new Set(decoded)).toEqual(legalSquares);
  });

  it('decodes promotions', () => {
    // White pawn on a7 promotes: a7a8=q.
    const board = new ChessBoard('8/P7/8/8/8/8/8/K6k w - - 0 1');
    const actions = board.get_legal_actions();
    const promotions = actions
      .map((a) => board.decodeMove(a))
      .filter((m) => m.promotion !== undefined);
    expect(promotions.length).toBe(4); // q, r, n, b
    expect(promotions.every((m) => m.from === 'a7' && m.to === 'a8')).toBe(
      true,
    );
  });
});

describe('ChessBoard.encodeMove', () => {
  // Regression coverage for the UCI wrapper (tui/az-uci.ts): it receives
  // moves as UCI strings from a GUI/opponent engine's `position ... moves
  // ...` command and needs to turn them back into the same action indices
  // get_legal_actions() would produce, i.e. encodeMove must be decodeMove's
  // exact inverse.
  it('is the inverse of decodeMove for every legal starting move', () => {
    const board = new ChessBoard();
    for (const action of board.get_legal_actions()) {
      const move = board.decodeMove(action);
      const uci = move.from + move.to + (move.promotion ?? '');
      expect(board.encodeMove(uci)).toBe(action);
    }
  });

  it('encodes promotion suffixes to the matching decodeMove action', () => {
    // White pawn on a7 promotes: a7a8=q/r/n/b.
    const board = new ChessBoard('8/P7/8/8/8/8/8/K6k w - - 0 1');
    const legal = new Set(board.get_legal_actions());
    for (const suffix of ['q', 'r', 'n', 'b']) {
      const action = board.encodeMove(`a7a8${suffix}`);
      expect(legal.has(action)).toBe(true);
      expect(board.decodeMove(action)).toEqual({
        from: 'a7',
        to: 'a8',
        promotion: suffix,
      });
    }
  });

  it('a non-promotion UCI move (no 5th character) encodes to promotion 0', () => {
    const board = new ChessBoard();
    const action = board.encodeMove('e2e4');
    expect(board.decodeMove(action)).toEqual({ from: 'e2', to: 'e4' });
  });
});

describe('renderBoard / describeResult', () => {
  it('renders the big board with cellHeight rows per rank plus file labels', () => {
    const lines = renderBoard(new Chess(), false, BIG_BOARD).split('\n');
    expect(lines.length).toBe(8 * BIG_BOARD.cellHeight + 1);
    // Rank labels sit on each rank's piece row.
    expect(lines.some((l) => l.startsWith('8 '))).toBe(true);
    const labels = lines[lines.length - 1]!.replace(/\s+/g, ' ').trim();
    expect(labels).toBe('a b c d e f g h');
  });

  it('big pieces are aligned multi-row figurines with a gap between ranks', () => {
    const boardRows = renderBoard(new Chess(), false, BIG_BOARD)
      .split('\n')
      .slice(0, 8 * BIG_BOARD.cellHeight) // drop the file-label line
      .map(stripAnsi);
    // Every board row has the same visible width: the block glyphs must stay
    // single-column, or the grid would shear on the piece rows.
    expect(new Set(boardRows.map((l) => l.length)).size).toBe(1);
    // Rank 8's figurine sits on the bottom row of its cell (the ▟█▙ pedestal)...
    expect(boardRows[BIG_BOARD.cellHeight - 1]).toContain('█');
    // ...and the first row of rank 7's cell is a blank spacer, so vertically
    // adjacent pieces never touch.
    expect(boardRows[BIG_BOARD.cellHeight]!.trim()).toBe('');
  });

  it('renders the compact board with one row per rank', () => {
    const lines = renderBoard(new Chess(), false, COMPACT_BOARD).split('\n');
    expect(lines.length).toBe(9);
    expect(lines[0]!.startsWith('8 ')).toBe(true);
    expect(lines[8]!.replace(/\s+/g, ' ').trim()).toBe('a b c d e f g h');
  });

  it('centers a single glyph in an odd-width cell', () => {
    // Strip ANSI, split into cells, and confirm equal padding around the piece.
    const line = stripAnsi(
      renderBoard(new Chess(), false, {
        cellWidth: 5,
        cellHeight: 1,
      }).split('\n')[0]!,
    );
    const firstCell = line.slice(2, 7); // after the "8 " rank label
    expect(firstCell).toBe('  ♜  '); // 2 left, glyph, 2 right - centered
  });

  it('ascii mode renders pieces as case-coded letters', () => {
    const lines = renderBoard(new Chess(), false, COMPACT_BOARD, true).split(
      '\n',
    );
    const rank1 = stripAnsi(lines[7]!);
    const rank8 = stripAnsi(lines[0]!);
    expect(rank1).toContain('K'); // white king, uppercase
    expect(rank8).toContain('k'); // black king, lowercase
    expect(rank1).not.toContain('♚'); // no unicode glyphs
  });

  it('reports checkmate for the fools mate position', () => {
    const chess = new Chess();
    for (const san of ['f3', 'e5', 'g4', 'Qh4#']) chess.move(san);
    expect(describeResult(chess)).toBe('Checkmate — Black wins.');
  });
});
