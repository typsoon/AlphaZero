import { describe, expect, test } from 'vitest';
import {
  squareToRowCol,
  rowColToSquare,
  getLegalMoves,
  resolveMyPlayerNumber,
  computeIsMyTurn,
  isMyPieceAt,
  nextPremove,
  pickPremoveAction,
  toggleArrow,
} from './chessInteraction';

describe('squareToRowCol / rowColToSquare', () => {
  test('a8 is row 0, col 0 (top-left, White orientation)', () => {
    expect(squareToRowCol('a8')).toEqual({ row: 0, col: 0 });
  });

  test('h1 is row 7, col 7 (bottom-right)', () => {
    expect(squareToRowCol('h1')).toEqual({ row: 7, col: 7 });
  });

  test('e4 round-trips through row/col and back to the same square', () => {
    const { row, col } = squareToRowCol('e4');
    expect(rowColToSquare(row, col)).toBe('e4');
  });

  test.each(['a1', 'b1', 'g1', 'h8', 'd4'])('round-trips %s', (square) => {
    const { row, col } = squareToRowCol(square);
    expect(rowColToSquare(row, col)).toBe(square);
  });
});

describe('getLegalMoves', () => {
  // action = (from_idx*64 + to_idx)*5 + promo; from/to_idx = row*8+col.
  function encode(
    fromRow: number,
    fromCol: number,
    toRow: number,
    toCol: number,
    promo = 0,
  ) {
    const from = fromRow * 8 + fromCol;
    const to = toRow * 8 + toCol;
    return (from * 64 + to) * 5 + promo;
  }

  test('filters to only actions matching the exact from/to square pair', () => {
    const legalActions = [
      encode(7, 1, 5, 2), // b1-c3
      encode(7, 1, 5, 0), // b1-a3
      encode(7, 6, 5, 5), // g1-f3
    ];
    expect(getLegalMoves(legalActions, 7, 1, 5, 2)).toEqual([
      encode(7, 1, 5, 2),
    ]);
  });

  test('returns an empty array when no action matches', () => {
    const legalActions = [encode(7, 1, 5, 2)];
    expect(getLegalMoves(legalActions, 6, 4, 4, 4)).toEqual([]);
  });

  test('returns every promotion option for the same square pair', () => {
    const legalActions = [
      encode(1, 0, 0, 0, 1), // =Q
      encode(1, 0, 0, 0, 2), // =R
      encode(1, 0, 0, 0, 3), // =N
      encode(1, 0, 0, 0, 4), // =B
    ];
    expect(getLegalMoves(legalActions, 1, 0, 0, 0)).toHaveLength(4);
  });
});

describe('resolveMyPlayerNumber', () => {
  test('matches p1 -> player 0', () => {
    expect(
      resolveMyPlayerNumber({
        playerId: 'abc',
        p1Id: 'abc',
        p2Id: 'xyz',
        currentPlayer: 0,
      }),
    ).toBe(0);
  });

  test('matches p2 -> player 1', () => {
    expect(
      resolveMyPlayerNumber({
        playerId: 'xyz',
        p1Id: 'abc',
        p2Id: 'xyz',
        currentPlayer: 1,
      }),
    ).toBe(1);
  });

  test('no match -> null (spectator or AI seat)', () => {
    expect(
      resolveMyPlayerNumber({
        playerId: 'someone-else',
        p1Id: 'abc',
        p2Id: 'xyz',
        currentPlayer: 0,
      }),
    ).toBeNull();
  });

  test('explicit "spectator" sentinel -> null even if it happened to match an id', () => {
    expect(
      resolveMyPlayerNumber({
        playerId: 'spectator',
        p1Id: 'spectator',
        p2Id: 'xyz',
        currentPlayer: 0,
      }),
    ).toBeNull();
  });

  test('null playerId -> null', () => {
    expect(
      resolveMyPlayerNumber({
        playerId: null,
        p1Id: 'abc',
        p2Id: 'xyz',
        currentPlayer: 0,
      }),
    ).toBeNull();
  });

  // Regression test: pass-and-play (both seats human) shares one identity
  // server-side (routes/game.ts's /game/create reuses p1Id as p2Id). The
  // first implementation of this function always resolved to player 0 in
  // that case, which made every one of black's own turns look like "the
  // opponent's turn" and caused premoves to misfire against yourself.
  describe('pass-and-play (p1Id === p2Id, shared identity)', () => {
    test('tracks currentPlayer rather than pinning to white', () => {
      const params = {
        playerId: 'shared-id',
        p1Id: 'shared-id',
        p2Id: 'shared-id',
      };
      expect(resolveMyPlayerNumber({ ...params, currentPlayer: 0 })).toBe(0);
      expect(resolveMyPlayerNumber({ ...params, currentPlayer: 1 })).toBe(1);
    });
  });
});

describe('computeIsMyTurn', () => {
  test('true when myPlayerNumber matches currentPlayer', () => {
    expect(computeIsMyTurn(0, 0)).toBe(true);
    expect(computeIsMyTurn(1, 1)).toBe(true);
  });

  test('false when they differ', () => {
    expect(computeIsMyTurn(0, 1)).toBe(false);
    expect(computeIsMyTurn(1, 0)).toBe(false);
  });

  test('false for a spectator (myPlayerNumber null)', () => {
    expect(computeIsMyTurn(null, 0)).toBe(false);
    expect(computeIsMyTurn(null, 1)).toBe(false);
  });
});

describe('isMyPieceAt', () => {
  test('white piece belongs to player 0', () => {
    expect(isMyPieceAt('N', 0)).toBe(true);
    expect(isMyPieceAt('N', 1)).toBe(false);
  });

  test('black piece belongs to player 1', () => {
    expect(isMyPieceAt('n', 1)).toBe(true);
    expect(isMyPieceAt('n', 0)).toBe(false);
  });

  test('empty square never belongs to anyone', () => {
    expect(isMyPieceAt(' ', 0)).toBe(false);
    expect(isMyPieceAt(' ', 1)).toBe(false);
    expect(isMyPieceAt(undefined, 0)).toBe(false);
  });

  test('spectator (null) never owns a piece, even a real one', () => {
    expect(isMyPieceAt('N', null)).toBe(false);
    expect(isMyPieceAt('n', null)).toBe(false);
  });
});

describe('nextPremove', () => {
  test('returns the {from, to} pair for two different squares', () => {
    expect(nextPremove({ row: 7, col: 1 }, { row: 5, col: 2 })).toEqual({
      from: { row: 7, col: 1 },
      to: { row: 5, col: 2 },
    });
  });

  test('same square for from and to cancels (returns null)', () => {
    expect(nextPremove({ row: 7, col: 1 }, { row: 7, col: 1 })).toBeNull();
  });
});

describe('pickPremoveAction', () => {
  test('returns null when there are no candidate actions', () => {
    expect(pickPremoveAction([])).toBeNull();
  });

  test('returns the single action when there is exactly one', () => {
    expect(pickPremoveAction([42])).toBe(42);
  });

  // action % 5: 0 = plain move, 1 = queen, 2 = rook, 3 = knight, 4 = bishop.
  test('prefers queen promotion over other candidates', () => {
    const knightPromo = 10 * 5 + 3;
    const queenPromo = 10 * 5 + 1;
    const rookPromo = 10 * 5 + 2;
    expect(pickPremoveAction([knightPromo, rookPromo, queenPromo])).toBe(
      queenPromo,
    );
  });

  test('falls back to the first action when no promotion option exists', () => {
    const plainMove = 5 * 5 + 0;
    expect(pickPremoveAction([plainMove])).toBe(plainMove);
  });
});

describe('toggleArrow', () => {
  test('adds a new arrow to an empty list', () => {
    expect(toggleArrow([], 'e2', 'e4')).toEqual([{ from: 'e2', to: 'e4' }]);
  });

  test('adding a second, different arrow keeps both', () => {
    const withOne = toggleArrow([], 'e2', 'e4');
    const withTwo = toggleArrow(withOne, 'g1', 'f3');
    expect(withTwo).toEqual([
      { from: 'e2', to: 'e4' },
      { from: 'g1', to: 'f3' },
    ]);
  });

  test('drawing the exact same arrow again removes it (toggle off)', () => {
    const withOne = toggleArrow([], 'e2', 'e4');
    expect(toggleArrow(withOne, 'e2', 'e4')).toEqual([]);
  });

  test('toggling one arrow off leaves other arrows untouched', () => {
    const arrows = [
      { from: 'e2', to: 'e4' },
      { from: 'g1', to: 'f3' },
    ];
    expect(toggleArrow(arrows, 'e2', 'e4')).toEqual([{ from: 'g1', to: 'f3' }]);
  });

  test('does not mutate the input array', () => {
    const arrows = [{ from: 'e2', to: 'e4' }];
    const result = toggleArrow(arrows, 'g1', 'f3');
    expect(arrows).toHaveLength(1);
    expect(result).toHaveLength(2);
  });
});
