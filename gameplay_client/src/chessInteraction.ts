// Pure, framework-agnostic chess-UI interaction logic factored out of
// ChessView.vue so it's unit-testable without mounting the component (which
// would otherwise require mocking chessboard.js, jQuery, WebSocket, and
// fetch). Anything here is a plain function of its arguments - no refs, no
// DOM, no network.

export type RowCol = { row: number; col: number };
export type Arrow = { from: string; to: string };

/** Our internal board is a string[][] grid with row 0 = rank 8, col 0 = file a. */
export function squareToRowCol(square: string): RowCol {
  const col = square.charCodeAt(0) - 97; // 'a' -> 0
  const rank = parseInt(square[1], 10);
  return { row: 8 - rank, col };
}

export function rowColToSquare(row: number, col: number): string {
  const file = String.fromCharCode(97 + col);
  const rank = 8 - row;
  return `${file}${rank}`;
}

// action = (from_idx*64 + to_idx)*5 + promo, from/to_idx = row*8+col - matches
// engine/game/chess.cpp's Chess::encode_action (row 0 = rank 8, col 0 = file a).
export function getLegalMoves(
  legalActions: number[],
  fromRow: number,
  fromCol: number,
  toRow: number,
  toCol: number,
): number[] {
  const from = fromRow * 8 + fromCol;
  const to = toRow * 8 + toCol;
  return legalActions.filter((action) => {
    const act = Math.floor(action / 5);
    const actTo = act % 64;
    const actFrom = Math.floor(act / 64);
    return actFrom === from && actTo === to;
  });
}

/**
 * Which of the game's two player ids this browser tab is. null = spectator
 * or an AI seat. Matches routes/game.ts's own `.includes()` comparison style
 * (its isP1/isP2 checks), not strict equality.
 *
 * Pass-and-play (both seats human) is a special case: routes/game.ts's
 * /game/create reuses p1Id as p2Id when both are human, since there's one
 * shared identity controlling both sides on the same device. There's no real
 * opponent to premove against in that mode, so whichever side is to move IS
 * "my turn" - track currentPlayer directly rather than pinning to whichever
 * id happened to match first (which would otherwise permanently resolve to
 * white, making black's own turns look like an opponent's).
 */
export function resolveMyPlayerNumber(params: {
  playerId: string | null;
  p1Id: string | null;
  p2Id: string | null;
  currentPlayer: number;
}): 0 | 1 | null {
  const { playerId, p1Id, p2Id, currentPlayer } = params;
  if (!playerId || playerId === 'spectator') return null;
  const matchesP1 = !!p1Id && playerId.includes(p1Id);
  const matchesP2 = !!p2Id && playerId.includes(p2Id);
  if (matchesP1 && matchesP2) return currentPlayer as 0 | 1;
  if (matchesP1) return 0;
  if (matchesP2) return 1;
  return null;
}

export function computeIsMyTurn(
  myPlayerNumber: 0 | 1 | null,
  currentPlayer: number,
): boolean {
  return myPlayerNumber !== null && myPlayerNumber === currentPlayer;
}

/** cell is a board cell string ('N', 'n', ' ', etc.) - uppercase = white = player 0. */
export function isMyPieceAt(
  cell: string | undefined,
  myPlayerNumber: 0 | 1 | null,
): boolean {
  if (myPlayerNumber === null) return false;
  if (!cell || cell === ' ') return false;
  const isWhitePiece = cell === cell.toUpperCase();
  return myPlayerNumber === 0 ? isWhitePiece : !isWhitePiece;
}

/** Dragging/clicking a premove's own source square again cancels it (returns null). */
export function nextPremove(
  from: RowCol,
  to: RowCol,
): { from: RowCol; to: RowCol } | null {
  if (from.row === to.row && from.col === to.col) return null;
  return { from, to };
}

/**
 * Picks which legal action a fired premove should use. A premove fires
 * instantly with no chance to show the promotion picker, so this prefers
 * queen promotion (action % 5 === 1) - the overwhelming common case, same as
 * lichess/chess.com's premove auto-queen. Returns null if none matched (the
 * opponent didn't cooperate - the premove is just dropped).
 */
export function pickPremoveAction(actions: number[]): number | null {
  if (actions.length === 0) return null;
  return actions.find((a) => a % 5 === 1) ?? actions[0];
}

/**
 * Adds a new arrow, or removes it if the exact same arrow already exists -
 * drawing the same line twice erases it (the lichess/chess.com convention).
 */
export function toggleArrow(
  arrows: Arrow[],
  from: string,
  to: string,
): Arrow[] {
  const existingIdx = arrows.findIndex((a) => a.from === from && a.to === to);
  if (existingIdx !== -1) {
    return [...arrows.slice(0, existingIdx), ...arrows.slice(existingIdx + 1)];
  }
  return [...arrows, { from, to }];
}
