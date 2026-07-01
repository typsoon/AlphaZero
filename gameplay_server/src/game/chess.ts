import { Chess, Move } from 'chess.js';
import type Game from './game.js';

export class ChessBoard implements Game {
  private chess: Chess;

  constructor(fen?: string) {
    this.chess = new Chess();
    if (fen) {
      this.chess.load(fen);
    }
  }

  reset(): void {
    this.chess.reset();
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
    };
  }

  get_current_player(): number {
    return this.chess.turn() === 'w' ? 0 : 1;
  }

  is_terminal(): boolean {
    return this.chess.isGameOver();
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
