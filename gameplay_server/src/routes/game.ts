/* eslint-disable @typescript-eslint/no-explicit-any */
import type { FastifyInstance, FastifyRequest } from 'fastify';
import Connect4 from '../game/connect4.js';
import { ChessBoard } from '../game/chess.js';
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

function getGameInstance(dbGame: any) {
  if (dbGame.gameType === 'chess') {
    const state = dbGame.board;
    return new ChessBoard(state.fen);
  } else {
    return new Connect4(dbGame.board);
  }
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

  server.get(
    '/agents',
    async (request: FastifyRequest<{ Querystring: { game?: string } }>) => {
      try {
        const queryGame = request.query.game;
        const baseDir = '/tmp';
        const files = await fs.readdir(baseDir);
        const dirs = files.filter((f) => f.startsWith('alphazero-inference-'));

        const agents = new Set<string>();
        for (const dir of dirs) {
          const fullDir = path.join(baseDir, dir);
          try {
            if (queryGame) {
              const gameDir = path.join(fullDir, queryGame);
              const subFiles = await fs.readdir(gameDir, {
                withFileTypes: true,
              });
              for (const f of subFiles) {
                if (f.isDirectory()) agents.add(f.name);
              }
            } else {
              const games = await fs.readdir(fullDir, { withFileTypes: true });
              for (const g of games) {
                if (g.isDirectory()) {
                  const subFiles = await fs.readdir(
                    path.join(fullDir, g.name),
                    { withFileTypes: true },
                  );
                  for (const f of subFiles) {
                    if (f.isDirectory()) agents.add(f.name);
                  }
                }
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
    },
  );

  server.get(
    '/game/:id/ws',
    { websocket: true },
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
    const isP1 =
      dbGame.gameType === 'chess' ? currentPlayer === 0 : currentPlayer === 1;
    const currentType = isP1 ? dbGame.p1Type : dbGame.p2Type;
    const currentAgent = isP1 ? dbGame.p1Agent : dbGame.p2Agent;

    if (currentType === 'ai' && currentAgent) {
      let resolvedAgent = currentAgent;
      if (!currentAgent.endsWith('.sock')) {
        // Resolve network name (e.g. "AZNetwork_0") to a socket path.
        const baseDir = '/tmp';
        try {
          const files = await fs.readdir(baseDir);
          const dirs = files.filter((f) =>
            f.startsWith('alphazero-inference-'),
          );
          for (const dir of dirs) {
            const networkDir = path.join(
              baseDir,
              dir,
              dbGame.gameType,
              currentAgent,
            );
            try {
              const sockets = await fs.readdir(networkDir);
              const validSockets = sockets.filter((s) => s.endsWith('.sock'));
              if (validSockets.length > 0) {
                // Pick most recently modified socket (most likely to be alive)
                const socketsWithStats = await Promise.all(
                  validSockets.map(async (s) => {
                    const fullPath = path.join(networkDir, s);
                    try {
                      const stat = await fs.stat(fullPath);
                      return { path: fullPath, mtime: stat.mtime.getTime() };
                    } catch {
                      return { path: fullPath, mtime: 0 };
                    }
                  }),
                );
                socketsWithStats.sort((a, b) => b.mtime - a.mtime);
                resolvedAgent = socketsWithStats[0]!.path;
                server.log.info(
                  `Resolved AI agent '${currentAgent}' to socket: ${resolvedAgent}`,
                );
                break;
              }
            } catch {
              /* ignore missing dir */
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

      const game = getGameInstance(dbGame);

      const agent = new AlphaZeroAgent(resolvedAgent);
      try {
        const aiGameState =
          dbGame.gameType === 'chess'
            ? (game as any).get_inference_state()
            : toCppInferenceGameState((game.get_board_state() as any).board);
        const aiMove = await agent.act(aiGameState);
        game.step(aiMove);

        const history = dbGame.history;
        const bs =
          dbGame.gameType === 'chess'
            ? game.get_board_state()
            : (game.get_board_state() as any).board;
        if (dbGame.gameType === 'chess') {
          (bs as any).last_action = aiMove;
        }
        history.push(bs);

        const isTerm = game.is_terminal();
        let winner: number | null = dbGame.winner ?? null;
        let winReason: string | null = dbGame.winReason ?? null;

        if (isTerm && dbGame.gameType === 'chess') {
          const boardState = game.get_board_state() as any;
          const ChessModule = await import('chess.js');
          const chessObj = new ChessModule.Chess(boardState.fen);
          if (chessObj.isCheckmate()) {
            winner = chessObj.turn() === 'w' ? -1 : 1;
            winReason = 'checkmate';
          } else if (
            chessObj.isDraw() ||
            chessObj.isStalemate() ||
            chessObj.isThreefoldRepetition() ||
            chessObj.isInsufficientMaterial()
          ) {
            winner = 0; // Draw
            winReason = 'draw';
          }
        }

        updateGame(
          id,
          JSON.stringify(
            dbGame.gameType === 'chess'
              ? game.get_board_state()
              : (game.get_board_state() as any).board,
          ),
          game.get_current_player(),
          isTerm,
          JSON.stringify(history),
          winner,
          winReason,
        );
        const newState = {
          board: (game.get_board_state() as any).board,
          fen:
            dbGame.gameType === 'chess'
              ? (game.get_board_state() as any).fen
              : undefined,
          legal_actions: game.get_legal_actions(),
          is_terminal: isTerm,
          current_player: game.get_current_player(),
          history: history,
          status: 'ok',
          surrender_winner: winner,
          win_reason: winReason,
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
        const err = e as { code?: string; message?: string; stack?: string };
        if (err.code === 'ENOENT') {
          server.log.warn(
            `Inference server socket not found at ${resolvedAgent}. Ensure the inference server is running.`,
          );
        } else {
          server.log.error(
            { resolvedAgent, code: err.code, stack: err.stack },
            `AI turn failed: ${err.message || String(e)}`,
          );
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
    const gameType = (body as any)?.game_type || 'connect4';

    let game;
    if (gameType === 'chess') {
      game = new ChessBoard();
    } else {
      game = new Connect4();
    }

    const p1Id = p1Type === 'human' ? uuidv4() : null;
    const p2Id = p2Type === 'human' ? (p1Type === 'human' ? p1Id : uuidv4()) : null;

    const history = [
      gameType === 'chess'
        ? game.get_board_state()
        : (game.get_board_state() as any).board,
    ];
    const gameId = createGame(
      JSON.stringify(
        gameType === 'chess'
          ? game.get_board_state()
          : (game.get_board_state() as any).board,
      ),
      game.get_current_player(),
      game.is_terminal(),
      p1Type,
      p1Agent,
      p1Id,
      p2Type,
      p2Agent,
      p2Id,
      gameType,
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

    const game = getGameInstance(dbGame);

    return {
      status: 'ok',
      board: dbGame.gameType === 'chess' ? dbGame.board.board : dbGame.board,
      fen: dbGame.gameType === 'chess' ? dbGame.board.fen : undefined,
      gameType: dbGame.gameType,
      legal_actions: game.get_legal_actions(),
      is_terminal: game.is_terminal(),
      current_player: game.get_current_player(),
      history: dbGame.history,
      surrender_winner: dbGame.winner,
      win_reason: dbGame.winReason,
      p1_type: dbGame.p1Type,
      p1_agent: dbGame.p1Agent,
      p2_type: dbGame.p2Type,
      p2_agent: dbGame.p2Agent,
      player_action:
        dbGame.gameType === 'chess' && dbGame.history.length > 0
          ? (dbGame.history[dbGame.history.length - 1] as any).last_action
          : undefined,
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

    const game = getGameInstance(dbGame);
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

    const isP1 = dbGame.p1Id && playerId.includes(dbGame.p1Id);
    const isP2 = dbGame.p2Id && playerId.includes(dbGame.p2Id);
    if (!isP1 && !isP2) {
      reply.code(403);
      return { status: 'error', message: 'Not a player in this game' };
    }

    let game;
    if (dbGame.gameType === 'chess') {
      game = new ChessBoard();
    } else {
      game = new Connect4();
    }
    const history = [
      dbGame.gameType === 'chess'
        ? game.get_board_state()
        : (game.get_board_state() as any).board,
    ];
    updateGame(
      id,
      JSON.stringify(
        dbGame.gameType === 'chess'
          ? game.get_board_state()
          : (game.get_board_state() as any).board,
      ),
      game.get_current_player(),
      false,
      JSON.stringify(history),
    );
    const newState = {
      board:
        dbGame.gameType === 'chess'
          ? (game.get_board_state() as any).board
          : (game.get_board_state() as any).board,
      fen:
        dbGame.gameType === 'chess'
          ? (game.get_board_state() as any).fen
          : undefined,
      legal_actions: game.get_legal_actions(),
      is_terminal: false,
      current_player: game.get_current_player(),
      history,
      status: 'ok',
      surrender_winner: null,
      win_reason: null,
    };
    broadcastState(id, newState);

    if (dbGame.p1Type === 'ai') {
      setTimeout(() => playAITurn(id), 100);
    }

    return { status: 'ok', message: 'Game reset' };
  });

  server.post('/game/:id/move', async (req, reply) => {
    const { id } = req.params as { id: string };
    const body = req.body as {
      column?: number;
      action?: number;
      player_id?: string;
    };
    const action = body?.action ?? body?.column;
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

    const game = getGameInstance(dbGame);

    if (game.is_terminal()) {
      reply.code(400);
      return { status: 'error', message: 'Game is already finished' };
    }

    const currentPlayer = game.get_current_player();
    const isP1 =
      dbGame.gameType === 'chess' ? currentPlayer === 0 : currentPlayer === 1;
    const expectedPlayerId = isP1 ? dbGame.p1Id : dbGame.p2Id;

    if (!expectedPlayerId || !playerId.includes(expectedPlayerId)) {
      reply.code(403);
      return { status: 'error', message: 'Not your turn or invalid player ID' };
    }

    if (
      action === undefined ||
      typeof action !== 'number' ||
      action < 0 ||
      action >= game.getActionSize()
    ) {
      reply.code(400);
      return { status: 'error', message: 'Invalid action' };
    }

    const legalActions = game.get_legal_actions();
    if (!legalActions.includes(action)) {
      reply.code(400);
      return {
        status: 'error',
        message: `Illegal move: action ${action} not available`,
      };
    }

    try {
      game.step(action);

      const history = dbGame.history;
      const bs =
        dbGame.gameType === 'chess'
          ? game.get_board_state()
          : (game.get_board_state() as any).board;
      if (dbGame.gameType === 'chess') {
        (bs as any).last_action = action;
      }
      history.push(bs);
      const isTerm = game.is_terminal();
      let winner: number | null = dbGame.winner ?? null;
      let winReason: string | null = dbGame.winReason ?? null;

      if (isTerm && dbGame.gameType === 'chess') {
        const boardState = game.get_board_state() as any;
        const ChessModule = await import('chess.js');
        const chessObj = new ChessModule.Chess(boardState.fen);
        if (chessObj.isCheckmate()) {
          // If it's checkmate, the player whose turn it is lost.
          winner = chessObj.turn() === 'w' ? -1 : 1;
          winReason = 'checkmate';
        } else if (
          chessObj.isDraw() ||
          chessObj.isStalemate() ||
          chessObj.isThreefoldRepetition() ||
          chessObj.isInsufficientMaterial()
        ) {
          winner = 0; // Draw
          winReason = 'draw';
        }
      }

      updateGame(
        id,
        JSON.stringify(
          dbGame.gameType === 'chess'
            ? game.get_board_state()
            : (game.get_board_state() as any).board,
        ),
        game.get_current_player(),
        isTerm,
        JSON.stringify(history),
        winner,
        winReason,
      );

      const newState = {
        board:
          dbGame.gameType === 'chess'
            ? (game.get_board_state() as any).board
            : (game.get_board_state() as any).board,
        fen:
          dbGame.gameType === 'chess'
            ? (game.get_board_state() as any).fen
            : undefined,
        gameType: dbGame.gameType,
        legal_actions: game.get_legal_actions(),
        is_terminal: isTerm,
        current_player: game.get_current_player(),
        history: history,
        status: 'ok',
        player_action: action,
        surrender_winner: winner,
        win_reason: winReason,
      };
      broadcastState(id, newState);

      if (!isTerm) {
        const nextPlayer = game.get_current_player();
        // Chess: 0=white=p1, 1=black=p2. Connect4: 1=p1, -1=p2.
        const nextIsP1 =
          dbGame.gameType === 'chess' ? nextPlayer === 0 : nextPlayer === 1;
        const nextType = nextIsP1 ? dbGame.p1Type : dbGame.p2Type;
        if (nextType === 'ai') {
          setTimeout(() => playAITurn(id), 100);
        }
      } else {
        // Do not immediately delete the game to allow fetching status
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
    if (dbGame.p1Id && playerId.includes(dbGame.p1Id)) surrenderingPlayer = 1;
    else if (dbGame.p2Id && playerId.includes(dbGame.p2Id))
      surrenderingPlayer = -1;
    else {
      reply.code(403);
      return { status: 'error', message: 'Not your game or invalid player ID' };
    }

    const winner = surrenderingPlayer === 1 ? -1 : 1;
    const winReason = 'surrender';

    updateGame(
      id,
      JSON.stringify(dbGame.board), // dbGame.board is already an object now! Wait, no, dbGame.board in the object is an object because we used JSON.parse(row.board) in getGame!
      // WAIT! If dbGame.board is an object, then JSON.stringify(dbGame.board) is correct!
      dbGame.currentPlayer,
      true,
      JSON.stringify(dbGame.history),
      winner,
      winReason,
    );

    const newState = {
      board: dbGame.gameType === 'chess' ? dbGame.board.board : dbGame.board,
      fen: dbGame.gameType === 'chess' ? dbGame.board.fen : undefined,
      gameType: dbGame.gameType,
      legal_actions: [],
      is_terminal: true,
      current_player: dbGame.currentPlayer,
      history: dbGame.history,
      status: 'ok',
      surrender_winner: winner,
      win_reason: winReason,
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
    let game;
    if (dbGame.gameType === 'chess') {
      game = new ChessBoard(newBoard.fen);
    } else {
      game = new Connect4(newBoard);
    }

    updateGame(
      id,
      JSON.stringify(
        dbGame.gameType === 'chess'
          ? game.get_board_state()
          : (game.get_board_state() as any).board,
      ),
      game.get_current_player(),
      false,
      JSON.stringify(history),
    );

    const newState = {
      board:
        dbGame.gameType === 'chess'
          ? (game.get_board_state() as any).board
          : (game.get_board_state() as any).board,
      fen:
        dbGame.gameType === 'chess'
          ? (game.get_board_state() as any).fen
          : undefined,
      gameType: dbGame.gameType,
      legal_actions: game.get_legal_actions(),
      is_terminal: game.is_terminal(),
      current_player: game.get_current_player(),
      history: history,
      status: 'ok',
      surrender_winner: null,
      win_reason: null,
      player_action:
        dbGame.gameType === 'chess' ? (newBoard as any).last_action : undefined,
    };
    broadcastState(id, newState);

    const currentPlayer = game.get_current_player();
    const isP1 =
      dbGame.gameType === 'chess' ? currentPlayer === 0 : currentPlayer === 1;
    const currentType = isP1 ? dbGame.p1Type : dbGame.p2Type;
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
        const isP1 =
          dbGame.gameType === 'chess'
            ? dbGame.currentPlayer === 0
            : dbGame.currentPlayer === 1;
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
