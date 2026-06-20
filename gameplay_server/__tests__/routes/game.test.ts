import { describe, expect, test, beforeAll, afterAll } from '@jest/globals'
import fastify from 'fastify'
import gameRoutes from '../../src/routes/game.js'

describe('Game Routes Integration', () => {
  const app = fastify()

  beforeAll(async () => {
    app.register(gameRoutes)
    await app.ready()
  })

  afterAll(async () => {
    await app.close()
  })

  test('GET /game/status returns valid game state schema', async () => {
    const response = await app.inject({
      method: 'GET',
      url: '/game/status'
    })

    expect(response.statusCode).toBe(200)
    const body = JSON.parse(response.payload)
    expect(body.status).toBe('ok')
    expect(body.is_terminal).toBe(false)
    expect(body.current_player).toBe(1)
    expect(body.legal_actions).toEqual([0, 1, 2, 3, 4, 5, 6])
    expect(body.board).toBeInstanceOf(Array)
    expect(body.board).toHaveLength(6)
  })

  test('POST /game/move with invalid column returns 400 Bad Request', async () => {
    const response = await app.inject({
      method: 'POST',
      url: '/game/move',
      payload: { column: 99 }
    })

    expect(response.statusCode).toBe(400)
    const body = JSON.parse(response.payload)
    expect(body.status).toBe('error')
    expect(body.message).toBe('Invalid column')
  })

  test('POST /game/move with missing column returns 400 Bad Request', async () => {
    const response = await app.inject({
      method: 'POST',
      url: '/game/move',
      payload: {}
    })

    expect(response.statusCode).toBe(400)
  })

  test('GET /game/reset resets the game successfully', async () => {
    const response = await app.inject({
      method: 'GET',
      url: '/game/reset'
    })

    expect(response.statusCode).toBe(200)
    const body = JSON.parse(response.payload)
    expect(body.status).toBe('ok')
    expect(body.message).toBe('Game reset')
  })

  test('POST /game/move handles both offline and online AI server gracefully', async () => {
    const response = await app.inject({
      method: 'POST',
      url: '/game/move',
      payload: { column: 3 }
    })

    if (response.statusCode === 200) {
      const body = JSON.parse(response.payload)
      expect(body.status).toBe('ok')
      expect(body.player_column).toBe(3)
      expect(typeof body.ai_column).toBe('number')
      expect(body.board).toBeInstanceOf(Array)
    } else {
      expect(response.statusCode).toBe(500)
      const body = JSON.parse(response.payload)
      expect(body.status).toBe('error')
      expect(body.message).toMatch(/ENOENT|Move failed/)
    }
  })
})
