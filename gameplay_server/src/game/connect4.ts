import type Game from './game.js'

type Connect4StateJson = {
  board: number[][]
}

class Connect4 implements Game {
  private static readonly ROWS = 6
  private static readonly COLS = 7

  private board: number[][]
  private currentPlayer: number
  private finished: boolean
  private rewardValue: number

  constructor(initialBoard?: number[][]) {
    this.board = []
    this.currentPlayer = 1
    this.finished = false
    this.rewardValue = 0.0

    if (initialBoard === undefined) {
      this.reset()
      return
    }

    this.validateBoardShape(initialBoard)
    this.board = initialBoard.map((row) => row.map((cell) => this.normalizeCellValue(cell)))

    let player1Count = 0
    let player2Count = 0
    for (const row of this.board) {
      for (const cell of row) {
        if (cell === 1) player1Count += 1
        else if (cell === -1) player2Count += 1
      }
    }

    this.currentPlayer = player1Count === player2Count ? 1 : -1
    this.finished = false
    this.rewardValue = 0.0

    for (let row = 0; row < Connect4.ROWS; row += 1) {
      for (let col = 0; col < Connect4.COLS; col += 1) {
        const cell = this.board[row]?.[col]
        if (cell !== undefined && cell !== 0 && this.checkWin(row, col)) {
          this.finished = true
          this.rewardValue = cell === this.currentPlayer ? 1.0 : -1.0
          return
        }
      }
    }

    let boardFull = true
    for (let col = 0; col < Connect4.COLS; col += 1) {
      const topCell = this.board[0]?.[col]
      if (topCell === 0) {
        boardFull = false
        break
      }
    }

    if (boardFull) {
      this.finished = true
      this.rewardValue = 0.0
    }
  }

  reset(): void {
    this.board = Array.from(
      { length: Connect4.ROWS },
      () => Array<number>(Connect4.COLS).fill(0),
    )
    this.currentPlayer = 1
    this.finished = false
    this.rewardValue = 0.0
  }

  getActionSize(): number {
    return Connect4.COLS
  }

  get_legal_actions(): number[] {
    const legalActions: number[] = []
    for (let col = 0; col < Connect4.COLS; col += 1) {
      if (this.board[0]?.[col] === 0) {
        legalActions.push(col)
      }
    }
    return legalActions
  }

  step(action: number): void {
    if (
      this.finished ||
      action < 0 ||
      action >= Connect4.COLS ||
      this.board[0]?.[action] !== 0
    ) {
      throw new Error(`Invalid action ${this.finished} ${action}`)
    }

    let placedRow = -1
    const placedCol = action
    for (let row = Connect4.ROWS - 1; row >= 0; row -= 1) {
      if (this.board[row]?.[action] === 0) {
        const rowRef = this.board[row]
        if (rowRef === undefined) {
          throw new Error('Invalid board state')
        }
        rowRef[action] = this.currentPlayer
        placedRow = row
        break
      }
    }

    if (placedRow === -1) {
      throw new Error(`Invalid action ${this.finished} ${action}`)
    }

    if (this.checkWin(placedRow, placedCol)) {
      this.finished = true
      this.rewardValue = 1.0
    } else if (this.get_legal_actions().length === 0) {
      this.finished = true
      this.rewardValue = 0.0
    } else {
      this.currentPlayer = -this.currentPlayer
    }
  }

  get_board_state(): Connect4StateJson {
    const serializedBoard = this.board.map((row) =>
      row.map((cell) => {
        if (cell === -1) return 2
        return cell
      }),
    )

    return { board: serializedBoard }
  }

  get_current_player(): number {
    return this.currentPlayer
  }

  is_terminal(): boolean {
    return this.finished
  }

  reward(): number {
    if (!this.finished) return 0.0
    return this.rewardValue
  }

  private validateBoardShape(board: number[][]): void {
    if (board.length !== Connect4.ROWS) {
      throw new Error(`Board must have ${Connect4.ROWS} rows`)
    }

    for (const row of board) {
      if (row.length !== Connect4.COLS) {
        throw new Error(`Board must have ${Connect4.COLS} columns`)
      }
    }
  }

  private normalizeCellValue(cell: number): number {
    if (cell === 2) return -1
    if (cell === -1 || cell === 0 || cell === 1) return cell
    throw new Error(`Invalid cell value: ${cell}`)
  }

  private checkWin(row: number, col: number): boolean {
    return (
      this.checkDirection(row, col, 1, 0) ||
      this.checkDirection(row, col, 0, 1) ||
      this.checkDirection(row, col, 1, 1) ||
      this.checkDirection(row, col, 1, -1)
    )
  }

  private checkDirection(row: number, col: number, dRow: number, dCol: number): boolean {
    let count = 0
    for (let i = -3; i <= 3; i += 1) {
      const r = row + i * dRow
      const c = col + i * dCol
      const inBounds = r >= 0 && r < Connect4.ROWS && c >= 0 && c < Connect4.COLS
      if (inBounds && this.board[r]?.[c] === this.currentPlayer) {
        count += 1
        if (count === 4) return true
      } else {
        count = 0
      }
    }
    return false
  }
}

export default Connect4
