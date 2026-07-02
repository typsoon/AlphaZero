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
    expect(body.message).toBe('Invalid action');
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
      expect(body.player_action).toBe(3);
      expect(body.board).toBeInstanceOf(Array);
    } else {
      expect(response.statusCode).toBe(500);
      const body = JSON.parse(response.payload);
      expect(body.status).toBe('error');
      expect(body.message).toMatch(/ENOENT|Move failed/);
    }
  });
  describe('AI vs human player ID assignment', () => {
    test('POST /game/create assigns a real p2_id when p1 is AI and p2 is human', async () => {
      // Regression test: p2_id used to be derived from p1_id (`p2Id = p2Type ===
      // 'human' ? p1Id : null`), so when p1 was AI (p1Id === null) but p2 was
      // human, p2 silently got null too - the human Player 2/Yellow could never
      // get a valid session id to make moves with.
      const response = await app.inject({
        method: 'POST',
        url: '/game/create',
        payload: {
          p1_type: 'ai',
          p2_type: 'human',
        },
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.payload);
      expect(body.status).toBe('ok');
      expect(body.p1_id).toBeNull();
      expect(body.p2_id).toBeDefined();
      expect(body.p2_id).not.toBeNull();
      expect(typeof body.p2_id).toBe('string');
      expect(body.p2_id.length).toBeGreaterThan(0);
    });

    test('POST /game/create still shares one id between two human players', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/game/create',
        payload: {
          p1_type: 'human',
          p2_type: 'human',
        },
      });

      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.payload);
      expect(body.status).toBe('ok');
      expect(body.p1_id).toBeDefined();
      expect(body.p2_id).toBe(body.p1_id);
    });
  });

  describe('Chess Game Tests', () => {
    let chessGameId = '';
    let chessPlayerId = '';

    test('POST /game/create creates a chess game', async () => {
      const response = await app.inject({
        method: 'POST',
        url: '/game/create',
        payload: {
          game_type: 'chess',
          p1_type: 'human',
          p2_type: 'human',
        },
      });
      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.payload);
      expect(body.status).toBe('ok');
      chessGameId = body.game_id;
      chessPlayerId = body.p1_id;
    });

    test('GET /game/:id/status returns valid chess state schema', async () => {
      const response = await app.inject({
        method: 'GET',
        url: `/game/${chessGameId}/status`,
      });
      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.payload);
      expect(body.status).toBe('ok');
      expect(body.gameType).toBe('chess');
      expect(body.board).toBeInstanceOf(Array);
      expect(body.board).toHaveLength(8); // Chess is 8x8
      expect(body.fen).toBeDefined();
    });

    test('POST /game/:id/move accepts a valid chess action', async () => {
      // Get legal actions first to ensure we pick a valid one dynamically
      const statusRes = await app.inject({
        method: 'GET',
        url: `/game/${chessGameId}/status`,
      });
      const statusBody = JSON.parse(statusRes.payload);
      const validAction = statusBody.legal_actions[0];

      const response = await app.inject({
        method: 'POST',
        url: `/game/${chessGameId}/move`,
        payload: { action: validAction, player_id: chessPlayerId },
      });
      expect(response.statusCode).toBe(200);
      const body = JSON.parse(response.payload);
      expect(body.status).toBe('ok');
      expect(body.player_action).toBe(validAction);
      expect(body.board).toBeInstanceOf(Array);
    });

    test('POST /game/:id/move rejects an invalid chess action out of bounds', async () => {
      const response = await app.inject({
        method: 'POST',
        url: `/game/${chessGameId}/move`,
        payload: { action: 999999, player_id: chessPlayerId },
      });
      expect(response.statusCode).toBe(400);
      const body = JSON.parse(response.payload);
      expect(body.status).toBe('error');
      expect(body.message).toBe('Invalid action');
    });
    test("plays a fool's mate and terminates the game", async () => {
      // Create a new game
      const createRes = await app.inject({
        method: 'POST',
        url: '/game/create',
        payload: {
          game_type: 'chess',
          p1_type: 'human',
          p2_type: 'human',
        },
      });
      const createBody = JSON.parse(createRes.payload);
      const gameId = createBody.game_id;
      const playerId = createBody.p1_id;

      // 1. f3 e5 2. g4 Qh4#
      const getChessAction = (
        fromRow: number,
        fromCol: number,
        toRow: number,
        toCol: number,
        promotion = 0,
      ) => {
        const fromIndex = fromRow * 8 + fromCol;
        const toIndex = toRow * 8 + toCol;
        return (fromIndex * 64 + toIndex) * 5 + promotion;
      };

      const moves = [
        getChessAction(6, 5, 5, 5), // f3 (white)
        getChessAction(1, 4, 3, 4), // e5 (black)
        getChessAction(6, 6, 4, 6), // g4 (white)
        getChessAction(0, 3, 4, 7), // Qh4# (black)
      ];

      for (const action of moves) {
        const moveRes = await app.inject({
          method: 'POST',
          url: `/game/${gameId}/move`,
          payload: { action, player_id: playerId },
        });
        expect(moveRes.statusCode).toBe(200);
      }

      const statusRes = await app.inject({
        method: 'GET',
        url: `/game/${gameId}/status`,
      });
      expect(statusRes.statusCode).toBe(200);
      const body = JSON.parse(statusRes.payload);
      expect(body.is_terminal).toBe(true);
      expect(body.surrender_winner).toBe(-1);
    });
  });
});
