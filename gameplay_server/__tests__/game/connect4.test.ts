import { describe, expect, test } from '@jest/globals'
import Connect4 from '../../src/game/connect4.js'

type BoardState = {
  board: number[][]
}

function expectBoardSchema(state: BoardState): void {
  expect(state).toEqual(
    expect.objectContaining({
      board: expect.any(Array),
    }),
  )
  expect(Object.keys(state)).toEqual(['board'])
  expect(state.board).toHaveLength(6)

  for (const row of state.board) {
    expect(Array.isArray(row)).toBe(true)
    expect(row).toHaveLength(7)
    for (const cell of row) {
      expect([0, 1, 2]).toContain(cell)
    }
  }
}

describe('Connect4', () => {
  test('starts as empty board and valid schema', () => {
    const game = new Connect4()
    const state = game.get_board_state() as BoardState

    expect(game.getActionSize()).toBe(7)
    expect(game.get_current_player()).toBe(1)
    expect(game.is_terminal()).toBe(false)
    expect(game.get_legal_actions()).toEqual([0, 1, 2, 3, 4, 5, 6])
    expectBoardSchema(state)
    expect(state.board.every((row) => row.every((cell) => cell === 0))).toBe(true)
  })

  test('drops pieces from bottom and serializes second player as 2', () => {
    const game = new Connect4()

    game.step(3) // player 1
    game.step(3) // player -1 (serialized as 2)

    const state = game.get_board_state() as BoardState
    expect(state.board[5]?.[3]).toBe(1)
    expect(state.board[4]?.[3]).toBe(2)
    expect(game.get_current_player()).toBe(1)
  })

  test('rejects invalid actions and full-column moves', () => {
    const game = new Connect4()

    expect(() => game.step(-1)).toThrow('Invalid action')
    expect(() => game.step(7)).toThrow('Invalid action')

    for (let i = 0; i < 6; i += 1) {
      game.step(0)
    }
    expect(game.get_legal_actions()).not.toContain(0)
    expect(() => game.step(0)).toThrow('Invalid action')
  })

  test('detects horizontal win like C++ logic', () => {
    const game = new Connect4()
    game.step(0)
    game.step(6)
    game.step(1)
    game.step(6)
    game.step(2)
    game.step(6)
    game.step(3)

    expect(game.is_terminal()).toBe(true)
    expect(game.get_current_player()).toBe(1)
    expect(game.reward()).toBe(1.0)
  })

  test('accepts initial board and preserves JSON enum format', () => {
    const initial = [
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 2, 0, 0, 0, 0],
      [0, 1, 2, 0, 0, 0, 0],
    ]
    const game = new Connect4(initial)

    const state = game.get_board_state() as BoardState
    expectBoardSchema(state)
    expect(state.board).toEqual(initial)
    expect(game.get_current_player()).toBe(-1)
  })

  test('rejects invalid initial board shape', () => {
    expect(() => new Connect4([[0]])).toThrow('Board must have 6 rows')
    const badCols = Array.from({ length: 6 }, () => [0, 0, 0, 0, 0, 0])
    expect(() => new Connect4(badCols)).toThrow('Board must have 7 columns')
  })
})
