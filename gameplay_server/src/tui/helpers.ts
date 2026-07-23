import fs from 'node:fs';
import path from 'node:path';
import type { Chess } from 'chess.js';
import type { Evaluation, EvaluationPolicyEntry } from '../agent.js';

/**
 * Picks the legal action with the highest policy mass from an inference-server
 * evaluation. The server's policy may be dense (number[] over the full action
 * space) or sparse ([{index, value}]); either way only indices present in
 * `legalActions` are considered, so an off-policy spike on an illegal index
 * can never produce an illegal move. Returns null if no legal action carries
 * any readable mass (callers should treat that as "pick any legal move").
 */
export function pickBestLegalAction(
  policy: Evaluation['policy'],
  legalActions: number[],
): number | null {
  if (legalActions.length === 0) return null;

  const mass = new Map<number, number>();
  if (policy.length > 0 && typeof policy[0] === 'object') {
    for (const entry of policy as EvaluationPolicyEntry[]) {
      mass.set(entry.index, entry.value);
    }
  } else {
    const dense = policy as number[];
    for (const action of legalActions) {
      const value = dense[action];
      if (value !== undefined) mass.set(action, value);
    }
  }

  let best: number | null = null;
  let bestValue = -Infinity;
  for (const action of legalActions) {
    const value = mass.get(action);
    if (value !== undefined && value > bestValue) {
      bestValue = value;
      best = action;
    }
  }
  return best;
}

const UNICODE_PIECES: Record<string, string> = {
  K: '♔',
  Q: '♕',
  R: '♖',
  B: '♗',
  N: '♘',
  P: '♙',
  k: '♚',
  q: '♛',
  r: '♜',
  b: '♝',
  n: '♞',
  p: '♟',
};

// Multi-row figurines. In a tall cell a single glyph occupies only the middle
// row and reads as a tiny piece; drawing these three rows fills the cell
// vertically so the piece looks ~3x larger without widening the board. Drawn
// with Unicode block elements (▙▟█▄▌▐▀ etc.), which are single-column - unlike
// the ♟-family chess glyphs that render double-width and break alignment - so
// they stay put on a phone terminal. Same art for both colors (color is the
// foreground escape). Distinct silhouettes: King's cornered crown, Queen's
// points, Rook's battlements, Bishop's mitre, Knight's head, small Pawn, all on
// a common ▟█▙ pedestal. `--ascii` is the ultra-safe single-letter fallback for
// terminals without block-glyph support. Rows are authored 5 wide to fill a BIG
// cell; renderBoard re-centers to the actual cellWidth.
const PIECE_ART: Record<string, string[]> = {
  P: ['     ', '  ▄  ', ' ▟█▙ '],
  R: [' ▙▄▟ ', ' ▐█▌ ', ' ▟█▙ '],
  N: [' ▄▀▜ ', ' ▘█▌ ', ' ▟█▙ '],
  B: ['  ▄  ', ' ▝█▘ ', ' ▟█▙ '],
  Q: [' ▀▄▀ ', ' ▐█▌ ', ' ▟█▙ '],
  K: [' ▟▄▙ ', ' ▐█▌ ', ' ▟█▙ '],
};
const PIECE_ART_HEIGHT = 3;

/** Centers a single art row to exactly `width` columns (pad/trim as needed). */
function fitRow(row: string, width: number): string {
  if (row.length === width) return row;
  if (row.length > width) {
    const start = Math.floor((row.length - width) / 2);
    return row.slice(start, start + width);
  }
  const total = width - row.length;
  const left = Math.floor(total / 2);
  return ' '.repeat(left) + row + ' '.repeat(total - left);
}

export type BoardStyle = {
  /** Characters per square. The piece glyph is centered inside. */
  cellWidth: number;
  /** Terminal rows per square (piece row is vertically centered). */
  cellHeight: number;
  /**
   * Draw pieces as multi-row ASCII figurines (see PIECE_ART) instead of a
   * single centered glyph, so pieces fill a tall cell. Ignored when `ascii` is
   * requested (letters stay single-glyph) or the cell is too short to hold the
   * art. Defaults to off.
   */
  bigPieces?: boolean;
};

// Odd cell dimensions are deliberate: a single-column glyph can only be
// centered in a box with an odd width and odd height (even sizes force one
// extra pad on the right and put the glyph on the top row, which reads as an
// off-center piece). 5x3 keeps the board within ~42 columns so it still fits a
// landscape phone terminal; COMPACT is the original dense layout.
// cellHeight is 4 (one taller than the 3-row figurine) so the piece rests on
// the bottom of its square and the spare row sits on top as breathing room -
// without it a 3-row piece fills a 3-row tile and touches the piece on the
// rank above, with no gap between them.
export const BIG_BOARD: BoardStyle = {
  cellWidth: 5,
  cellHeight: 4,
  bigPieces: true,
};
export const COMPACT_BOARD: BoardStyle = { cellWidth: 3, cellHeight: 1 };

/**
 * Renders the position as a fixed-width ANSI board. `flip` draws it from
 * Black's point of view (rank 1 at the top) for users playing Black. `ascii`
 * renders pieces as letters (uppercase = White, lowercase = Black) instead of
 * Unicode chess symbols; letters are guaranteed single-width, so they stay
 * centered on terminals that render the ♟-family glyphs as double-width.
 */
export function renderBoard(
  chess: Chess,
  flip: boolean,
  style: BoardStyle = BIG_BOARD,
  ascii = false,
): string {
  const { cellWidth, cellHeight } = style;
  const grid = chess.board();
  const lines: string[] = [];
  const ranks = flip
    ? [7, 6, 5, 4, 3, 2, 1, 0].reverse()
    : [0, 1, 2, 3, 4, 5, 6, 7];
  const files = flip ? [7, 6, 5, 4, 3, 2, 1, 0] : [0, 1, 2, 3, 4, 5, 6, 7];
  // The glyph sits in the exact middle of the cell (odd dimensions required).
  const pieceRow = Math.floor(cellHeight / 2);
  const padLeft = Math.floor(cellWidth / 2);
  const padRight = cellWidth - 1 - padLeft;
  // Multi-row figurines need a cell at least as tall as the art and are only
  // used in glyph mode (ascii mode keeps single letters).
  const useArt =
    style.bigPieces === true && !ascii && cellHeight >= PIECE_ART_HEIGHT;
  // Bottom-align the figurine: the piece rests on the square's bottom edge and
  // any spare rows fall above it as a gap from the piece on the rank above.
  const artStartRow = cellHeight - PIECE_ART_HEIGHT;

  for (const r of ranks) {
    for (let row = 0; row < cellHeight; row++) {
      let line = row === pieceRow ? `${8 - r} ` : '  ';
      for (const f of files) {
        const square = grid[r]![f];
        const dark = (r + f) % 2 === 1;
        const bg = dark ? '\x1b[48;5;94m' : '\x1b[48;5;180m';
        let cell = ' '.repeat(cellWidth);
        if (square) {
          const fg =
            square.color === 'w' ? '\x1b[1m\x1b[97m' : '\x1b[1m\x1b[30m';
          const art = useArt ? PIECE_ART[square.type.toUpperCase()] : undefined;
          if (art) {
            const artRowIdx = row - artStartRow;
            if (artRowIdx >= 0 && artRowIdx < PIECE_ART_HEIGHT) {
              cell = `${fg}${fitRow(art[artRowIdx] ?? '', cellWidth)}`;
            }
          } else if (row === pieceRow) {
            const letter =
              square.color === 'w' ? square.type.toUpperCase() : square.type;
            const glyph = ascii ? letter : (UNICODE_PIECES[letter] ?? letter);
            cell = `${' '.repeat(padLeft)}${fg}${glyph}${' '.repeat(padRight)}`;
          }
        }
        line += `${bg}${cell}\x1b[0m`;
      }
      lines.push(line);
    }
  }
  const fileLabels = files
    .map((f) =>
      String.fromCharCode('a'.charCodeAt(0) + f)
        .padStart(padLeft + 1)
        .padEnd(cellWidth),
    )
    .join('');
  lines.push(`  ${fileLabels}`);
  return lines.join('\n');
}

/** One-line, human-readable verdict for a finished chess.js game. */
export function describeResult(chess: Chess): string {
  if (chess.isCheckmate()) {
    return chess.turn() === 'w'
      ? 'Checkmate — Black wins.'
      : 'Checkmate — White wins.';
  }
  if (chess.isStalemate()) return 'Draw — stalemate.';
  if (chess.isThreefoldRepetition()) return 'Draw — threefold repetition.';
  if (chess.isInsufficientMaterial()) return 'Draw — insufficient material.';
  if (chess.isDraw()) return 'Draw.';
  return 'Game over.';
}

export type NetworkChoice = { label: string; path: string };

/**
 * Enumerates playable chess networks: the live checkpoint (TensorRT, falling
 * back to TorchScript) plus every timestamped TensorRT archive under
 * old_checkpoint/, newest first. Paths that don't exist are skipped, so the
 * list is always directly usable.
 */
export function listNetworks(projRoot: string): NetworkChoice[] {
  const choices: NetworkChoice[] = [];
  const chessDir = path.join(projRoot, 'checkpoints', 'chess');

  for (const [label, rel] of [
    ['current (TensorRT)', path.join('tensorrt', 'chess_AZNetwork_0.pt_trt')],
    [
      'current (TorchScript)',
      path.join('scripted', 'chess_AZNetwork_0.pt_scripted'),
    ],
  ] as const) {
    const full = path.join(chessDir, rel);
    if (fs.existsSync(full)) {
      choices.push({ label, path: full });
      break; // Prefer TensorRT; only fall back when it is missing.
    }
  }

  const archiveDir = path.join(chessDir, 'old_checkpoint');
  if (fs.existsSync(archiveDir)) {
    const archives = fs
      .readdirSync(archiveDir)
      .filter((f) => f.endsWith('.pt_trt'))
      .sort()
      .reverse();
    for (const f of archives) {
      const stamp = f.replace('chess_AZNetwork_', '').replace('.pt_trt', '');
      choices.push({
        label: `archive ${stamp}`,
        path: path.join(archiveDir, f),
      });
    }
  }
  return choices;
}
