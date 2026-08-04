import { Chess, Move } from 'chess.js';
import type Game from './game.js';

export class ChessBoard implements Game {
  private chess: Chess;
  // Ordered actions this instance has actually stepped through, from either
  // the starting position (no fen given) or an arbitrary fen (puzzle/editor
  // positions - no real game history, so this stays empty for those). Sent
  // to the inference server alongside the current position so it can replay
  // the same moves and correctly seed threefold-repetition/history-encoder
  // state instead of always evaluating as if the game just started - see
  // get_inference_state().
  private actionHistory: number[] = [];

  constructor(fen?: string) {
    this.chess = new Chess();
    if (fen) {
      this.chess.load(fen);
    }
  }

  reset(): void {
    this.chess.reset();
    this.actionHistory = [];
  }

  getActionSize(): number {
    return 64 * 64 * 5;
  }

  get_legal_actions(): number[] {
    const moves = this.chess.moves({ verbose: true });
    return moves.map((m) => this.encodeAction(m));
  }

  step(action: number): void {
    const move = this.decodeAction(action);
    try {
      this.chess.move(move);
      this.actionHistory.push(action);
    } catch (error) {
      console.error(
        'Invalid move attempted: ',
        move,
        ' action: ',
        action,
        ' error: ',
        error,
      );
    }
  }

  get_board_state(): { board: string[][]; fen: string } {
    return {
      board: this.chess.board().map((row) =>
        row.map((square) => {
          if (!square) return ' ';
          return square.color === 'w' ? square.type.toUpperCase() : square.type;
        }),
      ),
      fen: this.chess.fen(),
    };
  }

  get_inference_state(): {
    board: number[][];
    player: number;
    en_passant: number;
    castling: number[];
    history: number[];
  } {
    const board = this.chess.board().map((row) =>
      row.map((square) => {
        if (!square) return 0;
        const color = square.color === 'w' ? 1 : -1;
        switch (square.type) {
          case 'p':
            return 1 * color;
          case 'n':
            return 2 * color;
          case 'b':
            return 3 * color;
          case 'r':
            return 4 * color;
          case 'q':
            return 5 * color;
          case 'k':
            return 6 * color;
          default:
            return 0;
        }
      }),
    );
    const fenTokens = this.chess.fen().split(' ');
    const castlingFen = fenTokens[2] || '-';
    const enPassantFen = fenTokens[3] || '-';

    let en_passant = -1;
    if (enPassantFen !== '-') {
      en_passant = enPassantFen.charCodeAt(0) - 'a'.charCodeAt(0);
    }

    const k_mc =
      !castlingFen.includes('k') && !castlingFen.includes('q') ? 1 : 0;
    const r1_mc = !castlingFen.includes('q') ? 1 : 0;
    const r2_mc = !castlingFen.includes('k') ? 1 : 0;

    const K_mc =
      !castlingFen.includes('K') && !castlingFen.includes('Q') ? 1 : 0;
    const R1_mc = !castlingFen.includes('Q') ? 1 : 0;
    const R2_mc = !castlingFen.includes('K') ? 1 : 0;

    return {
      board,
      player: this.get_current_player(),
      en_passant,
      castling: [k_mc, r1_mc, r2_mc, K_mc, R1_mc, R2_mc],
      history: this.actionHistory,
    };
  }

  get_current_player(): number {
    return this.chess.turn() === 'w' ? 0 : 1;
  }

  /**
   * Public decoding of an engine action index into a from/to(/promotion) move,
   * for callers (e.g. the terminal UI) that drive their own chess.js instance
   * and only need the mapping, not this board's state.
   */
  decodeMove(action: number): { from: string; to: string; promotion?: string } {
    return this.decodeAction(action);
  }

  /**
   * The inverse of decodeMove: encodes a UCI move string ("e2e4", "e7e8q")
   * into an engine action index, for callers (e.g. a UCI engine wrapper)
   * that receive moves as UCI text - from an opponent engine, a GUI's
   * `position ... moves ...` command, etc. - rather than from this board's
   * own get_legal_actions().
   */
  encodeMove(uci: string): number {
    const fromSquare = this.squareToRowCol(uci.slice(0, 2));
    const toSquare = this.squareToRowCol(uci.slice(2, 4));
    const promotionChar = uci.length > 4 ? uci[4] : undefined;

    const from = fromSquare.row * 8 + fromSquare.col;
    const to = toSquare.row * 8 + toSquare.col;

    let promotion = 0;
    if (promotionChar === 'q') promotion = 1;
    else if (promotionChar === 'r') promotion = 2;
    else if (promotionChar === 'n') promotion = 3;
    else if (promotionChar === 'b') promotion = 4;

    return (from * 64 + to) * 5 + promotion;
  }

  is_terminal(): boolean {
    return this.chess.isGameOver();
  }

  /**
   * Winner/reason for a finished game, derived from this instance's own
   * chess.js state (which has seen the full move history) rather than a
   * fresh Chess() reloaded from just the current FEN - reloading from FEN
   * alone loses the position-repetition count, so isThreefoldRepetition()
   * (and therefore isDraw()/isGameOver()) would never be able to fire on it.
   */
  get_game_over_reason(): { winner: number; reason: string } | null {
    if (this.chess.isCheckmate()) {
      // The side to move is the one in checkmate, i.e. the loser.
      return {
        winner: this.chess.turn() === 'w' ? -1 : 1,
        reason: 'checkmate',
      };
    }
    if (
      this.chess.isStalemate() ||
      this.chess.isInsufficientMaterial() ||
      this.chess.isThreefoldRepetition() ||
      this.chess.isDrawByFiftyMoves()
    ) {
      return { winner: 0, reason: 'draw' };
    }
    return null;
  }

  /** Full move history in PGN notation, as recorded by chess.js's own SAN
   * tracking (accumulated incrementally in step() via this.chess.move()). */
  getPgn(): string {
    return this.chess.pgn();
  }

  // --- Internal Mapping Helpers ---

  private squareToRowCol(square: string): { row: number; col: number } {
    const file = square.charCodeAt(0) - 'a'.charCodeAt(0);
    const rank = parseInt(square[1]!, 10);
    return { row: 8 - rank, col: file };
  }

  private rowColToSquare(row: number, col: number): string {
    const file = String.fromCharCode('a'.charCodeAt(0) + col);
    const rank = 8 - row;
    return `${file}${rank}`;
  }

  private encodeAction(move: Move): number {
    const fromSquare = this.squareToRowCol(move.from);
    const toSquare = this.squareToRowCol(move.to);

    const from = fromSquare.row * 8 + fromSquare.col;
    const to = toSquare.row * 8 + toSquare.col;

    let promotion = 0;
    if (move.promotion) {
      if (move.promotion === 'q') promotion = 1;
      else if (move.promotion === 'r') promotion = 2;
      else if (move.promotion === 'n') promotion = 3;
      else if (move.promotion === 'b') promotion = 4;
    }

    return (from * 64 + to) * 5 + promotion;
  }

  private decodeAction(action: number): {
    from: string;
    to: string;
    promotion?: string;
  } {
    const promotionVal = action % 5;
    action = Math.floor(action / 5);
    const toIndex = action % 64;
    action = Math.floor(action / 64);
    const fromIndex = action;

    const fromRow = Math.floor(fromIndex / 8);
    const fromCol = fromIndex % 8;
    const toRow = Math.floor(toIndex / 8);
    const toCol = toIndex % 8;

    const move: { from: string; to: string; promotion?: string } = {
      from: this.rowColToSquare(fromRow, fromCol),
      to: this.rowColToSquare(toRow, toCol),
    };

    if (promotionVal !== 0) {
      if (promotionVal === 1) move.promotion = 'q';
      else if (promotionVal === 2) move.promotion = 'r';
      else if (promotionVal === 3) move.promotion = 'n';
      else if (promotionVal === 4) move.promotion = 'b';
    }

    return move;
  }
}
