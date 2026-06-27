import type { FastifyInstance } from 'fastify';
import Connect4 from '../game/connect4.js';
import { AlphaZeroAgent } from '../agent.js';
import {
  createGame,
  getGame,
  updateGame,
  deleteGame,
  getGames,
  updateGamePlayers,
} from '../db.js';
import { v4 as uuidv4 } from 'uuid';
import fs from 'fs/promises';
import path from 'path';

function toCppInferenceGameState(board: number[][]) {
  const convertedBoard = board.map((row) =>
    row.map((cell) => (cell === -1 ? 2 : cell)),
  );
  return { board: convertedBoard };
}

const subscriptions = new Map<string, Set<WebSocket>>();

export function broadcastState(gameId: string, state: Record<string, unknown>) {
  const clients = subscriptions.get(gameId);
  if (clients) {
    for (const client of clients) {
      if (client.readyState === 1) {
        client.send(JSON.stringify({ type: 'state_update', data: state }));
      }
    }
  }
}

export default async function gameRoutes(server: FastifyInstance) {
  server.get('/games', async () => {
    try {
      return { status: 'ok', games: getGames() };
    } catch (e: unknown) {
      server.log.error(e);
      return { status: 'error', games: [] };
    }
  });

  server.get('/agents', async () => {
    try {
      const baseDir = '/tmp';
      const files = await fs.readdir(baseDir);
      const dirs = files.filter((f) => f.startsWith('alphazero-inference-'));

      const agents = new Set<string>();
      for (const dir of dirs) {
        const fullDir = path.join(baseDir, dir);
        try {
          const subFiles = await fs.readdir(fullDir, { withFileTypes: true });
          for (const f of subFiles) {
            if (f.isDirectory()) {
              agents.add(f.name);
            } else if (f.name.endsWith('.sock')) {
              // Legacy/fallback: if socket is directly in the inference folder without a network subfolder
              agents.add(f.name);
            }
          }
        } catch {
          /* ignore */
        }
      }
      return { status: 'ok', agents: Array.from(agents) };
    } catch (e: unknown) {
      server.log.error(e);
      return { status: 'error', agents: [] };
    }
  });

  server.get(
    '/game/:id/ws',
    { websocket: true },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (connection: any /* WebSocket */, req /* FastifyRequest */) => {
      const { id } = req.params as { id: string };
      if (!subscriptions.has(id)) {
        subscriptions.set(id, new Set());
      }
      subscriptions.get(id)!.add(connection);

      connection.on('close', () => {
        const subs = subscriptions.get(id);
        if (subs) {
          subs.delete(connection);
          if (subs.size === 0) {
            subscriptions.delete(id);
            const dbGame = getGame(id);
            if (dbGame && dbGame.finished) {
              deleteGame(id);
            }
          }
        }
      });
    },
  );

  async function playAITurn(id: string) {
    const dbGame = getGame(id);
    if (!dbGame || dbGame.finished) return;

    const currentPlayer = dbGame.currentPlayer;
    const isP1 = currentPlayer === 1;
    const currentType = isP1 ? dbGame.p1Type : dbGame.p2Type;
    const currentAgent = isP1 ? dbGame.p1Agent : dbGame.p2Agent;

    if (currentType === 'ai' && currentAgent) {
      let resolvedAgent = currentAgent;
      if (!currentAgent.endsWith('.sock')) {
        // Resolve network name (e.g. "default") to an active socket path
        const baseDir = '/tmp';
        try {
          const files = await fs.readdir(baseDir);
          const dirs = files.filter((f) =>
            f.startsWith('alphazero-inference-'),
          );
          for (const dir of dirs) {
            const networkDir = path.join(baseDir, dir, currentAgent);
            try {
              const sockets = await fs.readdir(networkDir);
              const validSockets = sockets.filter((s) => s.endsWith('.sock'));
              if (validSockets.length > 0) {
                // Pick one randomly
                const chosen = validSockets[
                  Math.floor(Math.random() * validSockets.length)
                ] as string;
                resolvedAgent = path.join(networkDir, chosen);
                break;
              }
            } catch {
              /* ignore */
            }
          }
        } catch {
          /* ignore */
        }
      }

      if (!resolvedAgent.endsWith('.sock')) {
        server.log.warn(
          `No active inference server found for AI agent '${currentAgent}' in game ${id}. Ensure the inference server is running.`,
        );
        return;
      }

      const game = new Connect4(dbGame.board);
      const agent = new AlphaZeroAgent(resolvedAgent);
      try {
        const aiGameState = toCppInferenceGameState(
          game.get_board_state().board,
        );
        const aiMove = await agent.act(aiGameState);
        game.step(aiMove);

        const history = dbGame.history;
        history.push(game.get_board_state().board);

        const isTerm = game.is_terminal();
        updateGame(
          id,
          JSON.stringify(game.get_board_state().board),
          game.get_current_player(),
          isTerm,
          JSON.stringify(history),
        );
        const newState = {
          board: game.get_board_state().board,
          legal_actions: game.get_legal_actions(),
          is_terminal: isTerm,
          current_player: game.get_current_player(),
          history: history,
          status: 'ok',
          surrender_winner: null,
        };
        broadcastState(id, newState);

        if (!isTerm) {
          setTimeout(() => playAITurn(id), 500); // Wait slightly for visual
        } else {
          const subs = subscriptions.get(id);
          if (!subs || subs.size === 0) {
            deleteGame(id);
          }
        }
      } catch (e: unknown) {
        const err = e as { code?: string; message?: string };
        if (err.code === 'ENOENT') {
          server.log.warn(
            `Inference server socket not found at ${resolvedAgent}. Ensure the inference server is running.`,
          );
        } else {
          server.log.error(`AI turn failed: ${err.message || String(e)}`);
        }
      }
    }
  }

  server.post('/game/create', async (req) => {
    const body = req.body as {
      p1_type: string;
      p1_agent: string | null;
      p2_type: string;
      p2_agent: string | null;
    } | null;
    const p1Type = body?.p1_type || 'human';
    const p1Agent = body?.p1_agent || null;
    const p2Type = body?.p2_type || 'human';
    const p2Agent = body?.p2_agent || null;

    const game = new Connect4();
    const p1Id = p1Type === 'human' ? uuidv4() : null;
    const p2Id = p2Type === 'human' ? p1Id : null;

    const history = [game.get_board_state().board];
    const gameId = createGame(
      JSON.stringify(game.get_board_state().board),
      game.get_current_player(),
      game.is_terminal(),
      p1Type,
      p1Agent,
      p1Id,
      p2Type,
      p2Agent,
      p2Id,
      JSON.stringify(history),
    );

    // Start AI loop if P1 is AI
    if (p1Type === 'ai') {
      setTimeout(() => playAITurn(gameId), 100);
    }

    return { status: 'ok', game_id: gameId, p1_id: p1Id, p2_id: p2Id };
  });

  server.get('/game/:id/status', async (req, reply) => {
    const { id } = req.params as { id: string };
    const dbGame = getGame(id);
    if (!dbGame) {
      reply.code(404);
      return { status: 'error', message: 'Game not found' };
    }

    const game = new Connect4(dbGame.board);

    return {
      status: 'ok',
      board: game.get_board_state().board,
      legal_actions: game.get_legal_actions(),
      is_terminal: game.is_terminal(),
      current_player: game.get_current_player(),
      history: dbGame.history,
      surrender_winner: dbGame.winner,
      p1_type: dbGame.p1Type,
      p1_agent: dbGame.p1Agent,
      p2_type: dbGame.p2Type,
      p2_agent: dbGame.p2Agent,
    };
  });

  server.post('/game/:id/players', async (req, reply) => {
    const { id } = req.params as { id: string };
    const body = req.body as {
      p1_type: string;
      p1_agent: string | null;
      p2_type: string;
      p2_agent: string | null;
    };
    const dbGame = getGame(id);
    if (!dbGame) {
      reply.code(404);
      return { status: 'error', message: 'Game not found' };
    }

    const p1Id = body.p1_type === 'human' ? dbGame.p1Id || uuidv4() : null;
    const p2Id =
      body.p2_type === 'human'
        ? body.p1_type === 'human'
          ? p1Id
          : dbGame.p2Id || uuidv4()
        : null;

    updateGamePlayers(
      id,
      body.p1_type,
      body.p1_agent,
      p1Id,
      body.p2_type,
      body.p2_agent,
      p2Id,
    );

    const game = new Connect4(dbGame.board);
    const currentPlayerType =
      game.get_current_player() === 1 ? body.p1_type : body.p2_type;
    if (currentPlayerType === 'ai' && !game.is_terminal()) {
      setTimeout(() => playAITurn(id), 100);
    }

    return { status: 'ok', p1_id: p1Id, p2_id: p2Id };
  });

  server.post('/game/:id/reset', async (req, reply) => {
    const { id } = req.params as { id: string };
    const body = req.body as { player_id?: string };
    const playerId = body?.player_id;

    if (!playerId) {
      reply.code(401);
      return { status: 'error', message: 'Player ID is required' };
    }

    const dbGame = getGame(id);
    if (!dbGame) {
      reply.code(404);
      return { status: 'error', message: 'Game not found' };
    }

    if (playerId !== dbGame.p1Id && playerId !== dbGame.p2Id) {
      reply.code(403);
      return { status: 'error', message: 'Not a player in this game' };
    }

    const game = new Connect4();
    const history = [game.get_board_state().board];
    updateGame(
      id,
      JSON.stringify(game.get_board_state().board),
      game.get_current_player(),
      game.is_terminal(),
      JSON.stringify(history),
    );

    const newState = {
      board: game.get_board_state().board,
      legal_actions: game.get_legal_actions(),
      is_terminal: game.is_terminal(),
      current_player: game.get_current_player(),
      history: history,
      status: 'ok',
      surrender_winner: null,
    };
    broadcastState(id, newState);

    if (dbGame.p1Type === 'ai') {
      setTimeout(() => playAITurn(id), 100);
    }

    return { status: 'ok', message: 'Game reset' };
  });

  server.post('/game/:id/move', async (req, reply) => {
    const { id } = req.params as { id: string };
    const body = req.body as { column?: number; player_id?: string };
    const column = body?.column;
    const playerId = body?.player_id;

    if (!playerId) {
      reply.code(401);
      return { status: 'error', message: 'Player ID is required' };
    }

    const dbGame = getGame(id);
    if (!dbGame) {
      reply.code(404);
      return { status: 'error', message: 'Game not found' };
    }

    const game = new Connect4(dbGame.board);

    if (game.is_terminal()) {
      reply.code(400);
      return { status: 'error', message: 'Game is already finished' };
    }

    const currentPlayer = game.get_current_player();
    const expectedPlayerId = currentPlayer === 1 ? dbGame.p1Id : dbGame.p2Id;

    if (!expectedPlayerId || playerId !== expectedPlayerId) {
      reply.code(403);
      return { status: 'error', message: 'Not your turn or invalid player ID' };
    }

    if (
      column === undefined ||
      typeof column !== 'number' ||
      column < 0 ||
      column > 6
    ) {
      reply.code(400);
      return { status: 'error', message: 'Invalid column' };
    }

    const legalActions = game.get_legal_actions();
    if (!legalActions.includes(column)) {
      reply.code(400);
      return {
        status: 'error',
        message: `Illegal move: column ${column} not available`,
      };
    }

    try {
      game.step(column);

      const history = dbGame.history;
      history.push(game.get_board_state().board);

      const isTerm = game.is_terminal();
      updateGame(
        id,
        JSON.stringify(game.get_board_state().board),
        game.get_current_player(),
        isTerm,
        JSON.stringify(history),
      );

      const newState = {
        board: game.get_board_state().board,
        legal_actions: game.get_legal_actions(),
        is_terminal: isTerm,
        current_player: game.get_current_player(),
        history: history,
        status: 'ok',
        player_column: column,
        surrender_winner: null,
      };
      broadcastState(id, newState);

      if (!isTerm) {
        const nextPlayer = game.get_current_player();
        const nextType = nextPlayer === 1 ? dbGame.p1Type : dbGame.p2Type;
        if (nextType === 'ai') {
          setTimeout(() => playAITurn(id), 100);
        }
      } else {
        const subs = subscriptions.get(id);
        if (!subs || subs.size === 0) {
          deleteGame(id);
        }
      }

      return newState;
    } catch (e: unknown) {
      server.log.error(e);
      reply.code(500);
      return {
        status: 'error',
        message: (e as Error).message || 'Move failed',
      };
    }
  });

  server.post('/game/:id/surrender', async (req, reply) => {
    const { id } = req.params as { id: string };
    const body = req.body as { player_id?: string };
    const playerId = body?.player_id;

    if (!playerId) {
      reply.code(401);
      return { status: 'error', message: 'Player ID is required' };
    }

    const dbGame = getGame(id);
    if (!dbGame) {
      reply.code(404);
      return { status: 'error', message: 'Game not found' };
    }

    if (dbGame.finished) {
      reply.code(400);
      return { status: 'error', message: 'Game is already finished' };
    }

    let surrenderingPlayer = 0;
    if (playerId === dbGame.p1Id) surrenderingPlayer = 1;
    else if (playerId === dbGame.p2Id) surrenderingPlayer = -1;
    else {
      reply.code(403);
      return { status: 'error', message: 'Not your game or invalid player ID' };
    }

    const winner = surrenderingPlayer === 1 ? -1 : 1;

    updateGame(
      id,
      JSON.stringify(dbGame.board),
      dbGame.currentPlayer,
      true,
      JSON.stringify(dbGame.history),
      winner,
    );

    const newState = {
      board: dbGame.board,
      legal_actions: [],
      is_terminal: true,
      current_player: dbGame.currentPlayer,
      history: dbGame.history,
      status: 'ok',
      surrender_winner: winner,
    };
    broadcastState(id, newState);

    const subs = subscriptions.get(id);
    if (!subs || subs.size === 0) {
      deleteGame(id);
    }

    return newState;
  });

  server.post('/game/:id/resume', async (req, reply) => {
    const { id } = req.params as { id: string };
    const body = req.body as { history_index?: number };
    const index = body?.history_index;

    if (index === undefined || typeof index !== 'number' || index < 0) {
      reply.code(400);
      return { status: 'error', message: 'Invalid history index' };
    }

    const dbGame = getGame(id);
    if (!dbGame) {
      reply.code(404);
      return { status: 'error', message: 'Game not found' };
    }

    if (index >= dbGame.history.length) {
      reply.code(400);
      return { status: 'error', message: 'History index out of bounds' };
    }

    const history = dbGame.history.slice(0, index + 1);
    const newBoard = history[history.length - 1];
    const game = new Connect4(newBoard);

    updateGame(
      id,
      JSON.stringify(game.get_board_state().board),
      game.get_current_player(),
      game.is_terminal(),
      JSON.stringify(history),
    );

    const newState = {
      board: game.get_board_state().board,
      legal_actions: game.get_legal_actions(),
      is_terminal: game.is_terminal(),
      current_player: game.get_current_player(),
      history: history,
      status: 'ok',
      surrender_winner: null,
    };
    broadcastState(id, newState);

    const currentPlayer = game.get_current_player();
    const currentType = currentPlayer === 1 ? dbGame.p1Type : dbGame.p2Type;
    if (currentType === 'ai' && !game.is_terminal()) {
      setTimeout(() => playAITurn(id), 100);
    }

    return newState;
  });

  // On server startup, resume AI for all active games if it is currently their turn
  try {
    const activeGames = getGames().filter((g) => !g.finished);
    for (const g of activeGames) {
      const dbGame = getGame(g.id);
      if (dbGame) {
        const isP1 = dbGame.currentPlayer === 1;
        const currentType = isP1 ? dbGame.p1Type : dbGame.p2Type;
        if (currentType === 'ai') {
          setTimeout(() => playAITurn(g.id), 100);
        }
      }
    }
  } catch (e: unknown) {
    server.log.error(e);
  }
}
