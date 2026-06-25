import { describe, expect, test, beforeAll, afterAll } from '@jest/globals';
import fastify from 'fastify';
import gameRoutes from '../../src/routes/game.js';

describe('Game Routes Integration', () => {
  const app = fastify();
  let gameId = '';
  let playerId = '';

  beforeAll(async () => {
    app.register(gameRoutes);
    await app.ready();
  });

  afterAll(async () => {
    await app.close();
  });

  test('POST /game/create creates a game', async () => {
    const response = await app.inject({
      method: 'POST',
      url: '/game/create',
    });

    expect(response.statusCode).toBe(200);
    const body = JSON.parse(response.payload);
    expect(body.status).toBe('ok');
    expect(body.game_id).toBeDefined();
    expect(body.p1_id).toBeDefined();
    gameId = body.game_id;
    playerId = body.p1_id;
  });

  test('GET /game/:id/status returns valid game state schema', async () => {
    const response = await app.inject({
      method: 'GET',
      url: `/game/${gameId}/status`,
    });

    expect(response.statusCode).toBe(200);
    const body = JSON.parse(response.payload);
    expect(body.status).toBe('ok');
    expect(body.is_terminal).toBe(false);
    expect(body.current_player).toBe(1);
    expect(body.legal_actions).toEqual([0, 1, 2, 3, 4, 5, 6]);
    expect(body.board).toBeInstanceOf(Array);
    expect(body.board).toHaveLength(6);
  });

  test('POST /game/:id/move with invalid column returns 400 Bad Request', async () => {
    const response = await app.inject({
      method: 'POST',
      url: `/game/${gameId}/move`,
      payload: { column: 99, player_id: playerId },
    });

    expect(response.statusCode).toBe(400);
    const body = JSON.parse(response.payload);
    expect(body.status).toBe('error');
    expect(body.message).toBe('Invalid column');
  });

  test('POST /game/:id/move with missing player_id returns 401', async () => {
    const response = await app.inject({
      method: 'POST',
      url: `/game/${gameId}/move`,
      payload: { column: 3 },
    });

    expect(response.statusCode).toBe(401);
  });

  test('POST /game/:id/reset resets the game successfully', async () => {
    const response = await app.inject({
      method: 'POST',
      url: `/game/${gameId}/reset`,
      payload: { player_id: playerId },
    });

    expect(response.statusCode).toBe(200);
    const body = JSON.parse(response.payload);
    expect(body.status).toBe('ok');
    expect(body.message).toBe('Game reset');
  });

  test('POST /game/:id/move handles both offline and online AI server gracefully', async () => {
    const response = await app.inject({
      method: 'POST',
      url: `/game/${gameId}/move`,
      payload: { column: 3, player_id: playerId },
    });

    if (response.statusCode === 200) {
      const body = JSON.parse(response.payload);
      expect(body.status).toBe('ok');
      expect(body.player_column).toBe(3);
      expect(body.board).toBeInstanceOf(Array);
    } else {
      expect(response.statusCode).toBe(500);
      const body = JSON.parse(response.payload);
      expect(body.status).toBe('error');
      expect(body.message).toMatch(/ENOENT|Move failed/);
    }
  });
});
